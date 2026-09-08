"""LLM access, kept deliberately thin and provider-agnostic.

Any OpenAI-compatible endpoint works -- Groq, Cerebras, NVIDIA NIM -- and
nothing above this module knows which is in use. Four behaviours here came from
measuring real providers rather than reading their documentation:

1. Structured output is negotiated, not assumed. NVIDIA's documented
   ``nvext.guided_json`` extension is rejected by its own hosted endpoint (it
   exists only for self-hosted containers), so the chain starts at OpenAI-style
   ``response_format`` and degrades to prompting plus a repair retry. A provider
   that rejects an optional parameter has it dropped and the call retried.

2. Reasoning models emit their thinking before their answer and charge it
   against ``max_tokens``. A budget sized for the answer alone returns an empty
   string with ``finish_reason='length'`` and no error at all. Budgets are
   therefore generous, and an empty completion is retried with more room.

3. The SDK's default of two internal retries silently triples every timeout and
   makes a slow endpoint indistinguishable from a hung one. Retrying is done
   here instead, with backoff.

4. Responses are cached on a content hash, so re-running the corpus after a
   change to non-LLM code costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

from app import config

DEFAULT_MAX_TOKENS = 8000
RETRY_MAX_TOKENS = 16000
MAX_ATTEMPTS = 4
REQUEST_TIMEOUT = 600.0


class LLMError(RuntimeError):
    """Raised when a completion could not be obtained or parsed."""


class TokenBudget:
    """Client-side sliding-window rate limit, shared across worker threads.

    Hosted endpoints meter tokens per minute, not requests, and exceeding the
    budget returns 413 or 429 rather than queueing. Groq's free tier allows
    8,000 tokens per minute against 1,000 requests, so tokens bind long before
    requests do and eight parallel extractions breach it immediately.

    Waiting here is strictly better than being rejected there: a refused request
    still costs a round trip and loses whatever the model had already generated.
    """

    def __init__(
        self,
        tokens_per_minute: int,
        requests_per_minute: int,
        window_seconds: float = 60.0,
    ) -> None:
        self._tpm = max(1, tokens_per_minute)
        self._rpm = max(1, requests_per_minute)
        self._window_seconds = max(0.01, window_seconds)
        # Entries are mutable so a caller can correct its own reservation once
        # the true cost is known. Correcting by position instead would let one
        # thread's overage be charged to another thread's request.
        self._window: deque[list[float]] = deque()
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        cutoff = now - self._window_seconds
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()

    def acquire(self, estimated_tokens: int) -> list[float]:
        """Block until this request fits inside the trailing minute.

        Returns the caller's own reservation, to be passed back to ``settle``.
        """
        estimated = float(max(1, min(estimated_tokens, self._tpm)))
        while True:
            with self._lock:
                now = time.monotonic()
                self._prune(now)
                used = sum(entry[1] for entry in self._window)
                if used + estimated <= self._tpm and len(self._window) < self._rpm:
                    entry = [now, estimated]
                    self._window.append(entry)
                    return entry
                oldest = self._window[0][0] if self._window else now
                wait = self._window_seconds - (now - oldest)

            # Poll rather than sleeping out the whole window: entries expire
            # continuously, so capacity usually frees up well before the oldest
            # one does, and several waiting threads would otherwise all sleep
            # the maximum and then wake together.
            time.sleep(max(0.01, min(wait, 2.0)))

    def settle(self, reservation: list[float], actual_tokens: int) -> None:
        """Replace an estimate with the real cost, once the API reports it."""
        with self._lock:
            reservation[1] = float(max(0, min(actual_tokens, self._tpm)))

    def release(self, reservation: list[float]) -> None:
        """Give back a reservation for a request that never ran.

        A rejected or failed call spends nothing, so charging it against the
        window punishes the next caller for a request the provider refused.
        """
        with self._lock:
            reservation[1] = 0.0


@dataclass
class Completion:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cached: bool = False


def _cache_key(**parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


class LLMClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        cache_enabled: bool | None = None,
    ) -> None:
        self._client = OpenAI(
            api_key=api_key or config.LLM_API_KEY,
            base_url=base_url or config.LLM_BASE_URL,
            timeout=REQUEST_TIMEOUT,
            # This class does its own retrying with backoff; the SDK's default
            # of 2 silently multiplies every timeout by three and makes a slow
            # endpoint look like a hung one.
            max_retries=0,
        )
        self._cache_enabled = (
            config.LLM_CACHE_ENABLED if cache_enabled is None else cache_enabled
        )
        if self._cache_enabled:
            config.LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._budget = TokenBudget(config.LLM_TPM_LIMIT, config.LLM_RPM_LIMIT)

    # --- caching ----------------------------------------------------------

    def _cache_read(self, key: str) -> dict | None:
        if not self._cache_enabled:
            return None
        path = config.LLM_CACHE_DIR / f"{key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def _cache_write(self, key: str, payload: dict) -> None:
        if not self._cache_enabled:
            return
        path = config.LLM_CACHE_DIR / f"{key}.json"
        try:
            path.write_text(json.dumps(payload, ensure_ascii=False))
        except OSError:
            pass  # a cache miss is not worth failing an ingest over

    # --- raw completion ---------------------------------------------------

    def complete(
        self,
        messages: Sequence[dict],
        *,
        model: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
        response_format: dict | None = None,
    ) -> Completion:
        key = _cache_key(
            model=model,
            messages=list(messages),
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
        )
        hit = self._cache_read(key)
        if hit is not None:
            return Completion(
                text=hit["text"],
                model=model,
                prompt_tokens=hit.get("prompt_tokens", 0),
                completion_tokens=hit.get("completion_tokens", 0),
                cached=True,
            )

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        if config.REASONING_EFFORT:
            # Not every provider accepts this; an unsupported value is dropped
            # below rather than failing the request.
            kwargs["reasoning_effort"] = config.REASONING_EFFORT

        estimated = _estimate_tokens(messages, max_tokens)
        last_error: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            reservation = self._budget.acquire(estimated)
            try:
                response = self._client.chat.completions.create(**kwargs)
            except (RateLimitError, APIConnectionError) as exc:
                # A rejected request spends no tokens, so its reservation must
                # go back. Holding it for the full window meant each retry ate
                # budget it never used, and a few rejections in a row starved
                # the limiter into waiting minutes for capacity that was
                # already free.
                self._budget.release(reservation)
                last_error = exc
                self._backoff(attempt)
                continue
            except APIStatusError as exc:
                self._budget.release(reservation)
                # 413 "request too large" is how some providers report a
                # tokens-per-minute breach. It clears with time, so it is
                # retryable rather than fatal.
                if exc.status_code in (408, 413, 429) or exc.status_code >= 500:
                    last_error = exc
                    self._backoff(attempt)
                    continue
                # Providers reject parameters they do not implement. Drop the
                # optional ones and retry rather than failing the whole run.
                if exc.status_code == 400 and _drop_unsupported(kwargs, str(exc)):
                    continue
                raise LLMError(f"{model}: {exc.status_code} {str(exc)[:200]}") from exc

            choice = response.choices[0]
            text = choice.message.content or ""

            # A reasoning model can spend the whole budget thinking and return
            # nothing. Retry once with more room before giving up.
            if not text.strip() and choice.finish_reason == "length":
                if response.usage:
                    self._budget.settle(reservation, response.usage.total_tokens)
                if kwargs["max_tokens"] < RETRY_MAX_TOKENS:
                    kwargs["max_tokens"] = RETRY_MAX_TOKENS
                    continue
                raise LLMError(f"{model}: empty completion, budget exhausted by reasoning")

            if response.usage:
                self._budget.settle(reservation, response.usage.total_tokens)

            payload = {
                "text": text,
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            }
            self._cache_write(key, payload)
            return Completion(
                text=text,
                model=model,
                prompt_tokens=payload["prompt_tokens"],
                completion_tokens=payload["completion_tokens"],
            )

        raise LLMError(f"{model}: failed after {MAX_ATTEMPTS} attempts ({last_error})")

    @staticmethod
    def _backoff(attempt: int) -> None:
        time.sleep(min(2**attempt + random.random(), 30.0))

    # --- structured output ------------------------------------------------

    def complete_json(
        self,
        messages: Sequence[dict],
        *,
        model: str,
        schema: dict,
        schema_name: str = "result",
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Any:
        """Return a JSON object, trying the most reliable mechanism available.

        Order: OpenAI-style ``response_format`` -> plain prompting -> a repair
        retry that shows the model its own invalid output and the parse error.
        """
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": schema, "strict": True},
        }

        try:
            completion = self.complete(
                messages, model=model, max_tokens=max_tokens, response_format=response_format
            )
            return _loads(completion.text)
        except (LLMError, json.JSONDecodeError):
            pass

        completion = self.complete(messages, model=model, max_tokens=max_tokens)
        try:
            return _loads(completion.text)
        except json.JSONDecodeError as exc:
            return self._repair(messages, completion.text, str(exc), model=model,
                                max_tokens=max_tokens)

    def _repair(
        self, messages: Sequence[dict], broken: str, error: str, *, model: str, max_tokens: int
    ) -> Any:
        repair_messages = [
            *messages,
            {"role": "assistant", "content": broken[:4000]},
            {
                "role": "user",
                "content": (
                    f"That response was not valid JSON ({error}). "
                    "Reply with the corrected JSON object only, no prose, no code fences."
                ),
            },
        ]
        completion = self.complete(repair_messages, model=model, max_tokens=max_tokens)
        try:
            return _loads(completion.text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"{model}: unparseable JSON after repair ({exc})") from exc


# Optional request parameters, in the order they are given up when a provider
# rejects them. Structured output is surrendered last because losing it costs
# the most.
_OPTIONAL_PARAMS = ("reasoning_effort", "response_format")


def _drop_unsupported(kwargs: dict[str, Any], message: str) -> bool:
    """Remove one rejected optional parameter. True if something was dropped."""
    lowered = message.lower()
    for param in _OPTIONAL_PARAMS:
        if param in kwargs and param in lowered:
            kwargs.pop(param)
            return True
    # The error did not name a parameter; drop the least essential one present.
    for param in _OPTIONAL_PARAMS:
        if param in kwargs:
            kwargs.pop(param)
            return True
    return False


def _estimate_tokens(messages: Sequence[dict], max_tokens: int) -> int:
    """Rough budget reservation: prompt size plus a modest output allowance.

    Deliberately not exact. Reserving the full ``max_tokens`` would throttle to
    a fraction of the real limit, since extractions rarely use their whole
    budget; the reservation is corrected from reported usage once known.
    """
    prompt_chars = sum(len(str(m.get("content", ""))) for m in messages)
    # The output allowance is capped well below max_tokens on purpose: an
    # extraction rarely uses its whole ceiling, and reserving the ceiling would
    # throttle throughput to a fraction of the real limit. The reservation is
    # corrected from reported usage as soon as the call returns.
    return int(prompt_chars / 3.5) + min(max_tokens, 1500)


def _loads(text: str) -> Any:
    """Parse JSON, tolerating code fences and surrounding prose.

    Returns whatever the model produced -- an object or a bare array. Callers
    decide what shape they expect, because models honour a requested wrapper
    inconsistently even under a strict schema.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        parts = stripped.split("```")
        if len(parts) > 1:
            stripped = parts[1]
            stripped = stripped.removeprefix("json")
            stripped = stripped.strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Fall back to the outermost object or array embedded in the text.
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = stripped.find(opener), stripped.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                continue

    raise json.JSONDecodeError("no json object or array found", stripped, 0)


def map_concurrent[T, R](
    fn: Callable[[T], R], items: Iterable[T], *, workers: int = 8
) -> list[R]:
    """Run ``fn`` over ``items`` in parallel, preserving input order.

    Throughput here is latency-bound, not compute-bound: a reasoning model can
    take minutes on a single region, so concurrency is what makes a full corpus
    run finish in reasonable time.
    """
    items = list(items)
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(items)))) as pool:
        return list(pool.map(fn, items))
