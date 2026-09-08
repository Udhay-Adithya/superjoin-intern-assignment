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
import re
import sys
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
MAX_ATTEMPTS = 5
REQUEST_TIMEOUT = 600.0

# Longer than the 60-second window a per-minute quota resets on.
MAX_BACKOFF = 75.0

# Starting guess for completion size, before any real usage is seen.
DEFAULT_COMPLETION_ESTIMATE = 2500


class LLMError(RuntimeError):
    """Raised when a completion could not be obtained or parsed."""


class DailyQuotaExhausted(LLMError):
    """The account's daily token allowance is gone.

    Distinguished from a per-minute limit because the remedy is different:
    a per-minute breach clears in under a minute and is worth waiting for,
    while a daily one will not clear during the run. Retrying it burns time
    and tells the operator nothing.
    """


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
        # Reservations start from a guess and converge on what requests really
        # cost. Guessing low is what produced repeated 429s: several workers
        # would each reserve well under their true usage, burst past the quota
        # together, and then spend a minute backing off.
        self._observed_completion = float(DEFAULT_COMPLETION_ESTIMATE)

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

    def observe_completion(self, completion_tokens: int) -> None:
        """Fold a real completion size into the running estimate."""
        if completion_tokens <= 0:
            return
        with self._lock:
            # Weighted towards recent history so the estimate tracks the kind of
            # work currently being done rather than the whole run's average.
            self._observed_completion = (
                0.7 * self._observed_completion + 0.3 * float(completion_tokens)
            )

    def completion_estimate(self) -> int:
        with self._lock:
            return int(self._observed_completion)

    def headroom(self) -> float:
        """Tokens still available in the trailing window, right now."""
        with self._lock:
            now = time.monotonic()
            self._prune(now)
            return self._tpm - sum(entry[1] for entry in self._window)

    def release(self, reservation: list[float]) -> None:
        """Give back a reservation for a request that never ran.

        A rejected or failed call spends nothing, so charging it against the
        window punishes the next caller for a request the provider refused.
        """
        with self._lock:
            reservation[1] = 0.0


@dataclass
class _Endpoint:
    """One API key, with the quota that belongs to it.

    Rate limits are per key, so a pool multiplies both the per-minute and the
    per-day allowance. Each key therefore needs its own budget: sharing one
    would throttle the pool to a single key's rate.
    """

    label: str
    client: OpenAI
    budget: TokenBudget
    exhausted: bool = False


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
        keys = [api_key] if api_key else list(config.LLM_API_KEYS)
        if not keys:
            keys = [""]

        self._endpoints = [
            _Endpoint(
                label=f"key {index}",
                client=OpenAI(
                    api_key=key,
                    base_url=base_url or config.LLM_BASE_URL,
                    timeout=REQUEST_TIMEOUT,
                    # This class does its own retrying with backoff; the SDK's
                    # default of 2 silently multiplies every timeout by three
                    # and makes a slow endpoint look like a hung one.
                    max_retries=0,
                ),
                budget=TokenBudget(config.LLM_TPM_LIMIT, config.LLM_RPM_LIMIT),
            )
            for index, key in enumerate(keys, start=1)
        ]
        self._pool_lock = threading.Lock()
        self._cache_enabled = (
            config.LLM_CACHE_ENABLED if cache_enabled is None else cache_enabled
        )
        if self._cache_enabled:
            config.LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

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

    # --- key pool ---------------------------------------------------------

    def _pick_endpoint(self) -> _Endpoint | None:
        """The live key with the most room right now, or None if all are spent.

        Choosing by headroom spreads load without needing a round-robin cursor,
        and naturally favours a key whose window has drained while another is
        still waiting one out.
        """
        with self._pool_lock:
            live = [e for e in self._endpoints if not e.exhausted]
        if not live:
            return None
        return max(live, key=lambda e: e.budget.headroom())

    def _retire(self, endpoint: _Endpoint, detail: str) -> None:
        """Take a key out of the pool for the rest of the run."""
        with self._pool_lock:
            if endpoint.exhausted:
                return
            endpoint.exhausted = True
            remaining = sum(1 for e in self._endpoints if not e.exhausted)
        print(
            f"  [llm] {endpoint.label} exhausted its daily quota ({detail}); "
            f"{remaining} key(s) left",
            file=sys.stderr,
            flush=True,
        )

    @property
    def live_keys(self) -> int:
        with self._pool_lock:
            return sum(1 for e in self._endpoints if not e.exhausted)

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

        last_error: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            endpoint = self._pick_endpoint()
            if endpoint is None:
                raise DailyQuotaExhausted(
                    f"{model}: every key's daily quota is exhausted. Waiting will "
                    f"not help within this run; add another key or resume "
                    f"tomorrow. Work already done is cached and will not be "
                    f"repeated."
                )

            estimated = _estimate_tokens(
                messages, max_tokens, endpoint.budget.completion_estimate()
            )
            reservation = endpoint.budget.acquire(estimated)

            try:
                response = endpoint.client.chat.completions.create(**kwargs)
            except (RateLimitError, APIConnectionError) as exc:
                endpoint.budget.release(reservation)
                if _is_daily_quota(exc):
                    # This key is done for the day, but the others may not be.
                    # Retiring it rather than failing is the whole point of a
                    # pool; the attempt is not counted against the retry budget
                    # because no key was actually given a chance to answer.
                    self._retire(endpoint, _quota_detail(exc))
                    continue
                last_error = exc
                self._backoff(attempt, exc)
                continue
            except APIStatusError as exc:
                endpoint.budget.release(reservation)
                if exc.status_code == 429 and _is_daily_quota(exc):
                    self._retire(endpoint, _quota_detail(exc))
                    continue
                # 413 "request too large" is how some providers report a
                # tokens-per-minute breach. It clears with time, so it is
                # retryable rather than fatal.
                if exc.status_code in (408, 413, 429) or exc.status_code >= 500:
                    last_error = exc
                    self._backoff(attempt, exc)
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
                    endpoint.budget.settle(reservation, response.usage.total_tokens)
                if kwargs["max_tokens"] < RETRY_MAX_TOKENS:
                    kwargs["max_tokens"] = RETRY_MAX_TOKENS
                    continue
                raise LLMError(f"{model}: empty completion, budget exhausted by reasoning")

            if response.usage:
                endpoint.budget.settle(reservation, response.usage.total_tokens)
                endpoint.budget.observe_completion(response.usage.completion_tokens)

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
    def _retry_after(exc: Exception | None) -> float | None:
        """The provider's own advice on when to try again, if it gave any."""
        if exc is None:
            return None
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None) or {}
        raw = headers.get("retry-after") or headers.get("x-ratelimit-reset-tokens")
        if not raw:
            return None
        text = str(raw).strip()
        if text.endswith("s") and "m" not in text:
            text = text[:-1]
        try:
            return float(text)
        except ValueError:
            return None

    @classmethod
    def _backoff(cls, attempt: int, exc: Exception | None = None) -> None:
        """Wait before retrying, preferring the provider's stated delay.

        Exponential backoff alone was too short to help against a per-minute
        quota: 1+2+4+8 seconds all fall inside the same blocked window, so every
        attempt failed for the same reason and whole regions were lost. The
        ceiling is now longer than the window itself.
        """
        advised = cls._retry_after(exc)
        delay = advised + 0.5 if advised is not None else 2**attempt + random.random()
        time.sleep(min(delay, MAX_BACKOFF))

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


_DAILY_MARKERS = ("tokens per day", "tpd", "requests per day", "rpd")


def _is_daily_quota(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _DAILY_MARKERS)


def _quota_detail(exc: Exception) -> str:
    """Pull the useful numbers out of a provider's rate-limit message."""
    match = re.search(r"Limit (\d+), Used (\d+)", str(exc))
    if match:
        return f"limit {int(match.group(1)):,}, used {int(match.group(2)):,}"
    return "limit reached"


def _estimate_tokens(
    messages: Sequence[dict], max_tokens: int, completion_estimate: int
) -> int:
    """Budget reservation: prompt size plus what completions actually cost.

    Reserving the full ``max_tokens`` would throttle throughput to a fraction of
    the real limit, since an extraction rarely uses its whole ceiling. Reserving
    a fixed small allowance was the opposite mistake -- it under-reserved, so
    workers burst past the quota together and lost a minute to backing off.
    The allowance is therefore measured rather than assumed.
    """
    prompt_chars = sum(len(str(m.get("content", ""))) for m in messages)
    allowance = min(max(completion_estimate, DEFAULT_COMPLETION_ESTIMATE), max_tokens)
    return int(prompt_chars / 3.5) + allowance


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
