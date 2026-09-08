"""LLM access, kept deliberately thin and provider-agnostic.

Backed by NVIDIA NIM's OpenAI-compatible endpoint, but nothing above this module
knows that. Three behaviours here were driven by measurement rather than by the
documentation:

1. ``nvext.guided_json`` -- NVIDIA's documented structured-output extension -- is
   rejected by the hosted endpoint (it exists for self-hosted NIM containers).
   So the strategy chain starts at OpenAI-style ``response_format`` and falls
   back to prompting plus a repair retry.

2. Several catalogue models are reasoning models that emit ``reasoning_content``
   before ``content``, and that reasoning is charged against ``max_tokens``. A
   budget sized for the answer alone returns an empty string with
   ``finish_reason='length'`` and no error. Budgets here are therefore generous,
   and an empty completion is retried with a larger one.

3. Responses are cached on a content hash, so re-running the corpus after a
   change to non-LLM code costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, TypeVar

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

from app import config

T = TypeVar("T")
R = TypeVar("R")

DEFAULT_MAX_TOKENS = 8000
RETRY_MAX_TOKENS = 16000
MAX_ATTEMPTS = 4
REQUEST_TIMEOUT = 600.0


class LLMError(RuntimeError):
    """Raised when a completion could not be obtained or parsed."""


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
            api_key=api_key or config.NIM_API_KEY,
            base_url=base_url or config.NIM_BASE_URL,
            timeout=REQUEST_TIMEOUT,
        )
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

        last_error: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = self._client.chat.completions.create(**kwargs)
            except (RateLimitError, APIConnectionError) as exc:
                last_error = exc
                self._backoff(attempt)
                continue
            except APIStatusError as exc:
                if exc.status_code >= 500:
                    last_error = exc
                    self._backoff(attempt)
                    continue
                raise LLMError(f"{model}: {exc.status_code} {str(exc)[:200]}") from exc

            choice = response.choices[0]
            text = choice.message.content or ""

            # A reasoning model can spend the whole budget thinking and return
            # nothing. Retry once with more room before giving up.
            if not text.strip() and choice.finish_reason == "length":
                if kwargs["max_tokens"] < RETRY_MAX_TOKENS:
                    kwargs["max_tokens"] = RETRY_MAX_TOKENS
                    continue
                raise LLMError(f"{model}: empty completion, budget exhausted by reasoning")

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
        time.sleep(min(2**attempt + random.random(), 30.0))  # noqa: S311 - jitter, not crypto

    # --- structured output ------------------------------------------------

    def complete_json(
        self,
        messages: Sequence[dict],
        *,
        model: str,
        schema: dict,
        schema_name: str = "result",
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> dict:
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
    ) -> dict:
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


def _loads(text: str) -> dict:
    """Parse JSON, tolerating code fences and surrounding prose."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("```")[1]
        if stripped.startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(stripped[start : end + 1])


def map_concurrent(
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
