"""Key pool behaviour.

Rate limits are charged per key, so several keys multiply both the per-minute
and the per-day allowance. The pool exists because one key's daily cap stopped
a run dead: 200,000 tokens covers a fraction of a real corpus.
"""

from __future__ import annotations

from app.llm.client import DailyQuotaExhausted, LLMClient, _is_daily_quota


def _client(monkeypatch, keys: list[str]) -> LLMClient:
    monkeypatch.setattr("app.config.LLM_API_KEYS", keys)
    monkeypatch.setattr("app.config.LLM_CACHE_ENABLED", False)
    return LLMClient(cache_enabled=False)


def test_every_key_gets_its_own_budget(monkeypatch) -> None:
    """Sharing one budget would throttle the pool to a single key's rate."""
    client = _client(monkeypatch, ["a", "b", "c"])
    assert client.live_keys == 3

    budgets = {id(e.budget) for e in client._endpoints}  # noqa: SLF001
    assert len(budgets) == 3, "keys must not share a budget"


def test_load_spreads_towards_the_key_with_room(monkeypatch) -> None:
    client = _client(monkeypatch, ["a", "b"])
    first, second = client._endpoints  # noqa: SLF001

    # Fill the first key's window; the pool should prefer the second.
    first.budget.acquire(first.budget.headroom())
    assert client._pick_endpoint() is second  # noqa: SLF001


def test_an_exhausted_key_is_retired_but_the_others_continue(monkeypatch) -> None:
    client = _client(monkeypatch, ["a", "b", "c"])
    first = client._endpoints[0]  # noqa: SLF001

    client._retire(first, "limit 200,000, used 200,000")  # noqa: SLF001

    assert client.live_keys == 2
    assert client._pick_endpoint() is not first  # noqa: SLF001


def test_retiring_the_same_key_twice_is_harmless(monkeypatch) -> None:
    client = _client(monkeypatch, ["a", "b"])
    first = client._endpoints[0]  # noqa: SLF001

    client._retire(first, "x")  # noqa: SLF001
    client._retire(first, "x")  # noqa: SLF001

    assert client.live_keys == 1


def test_pool_reports_empty_once_every_key_is_spent(monkeypatch) -> None:
    client = _client(monkeypatch, ["a", "b"])
    for endpoint in client._endpoints:  # noqa: SLF001
        client._retire(endpoint, "spent")  # noqa: SLF001

    assert client.live_keys == 0
    assert client._pick_endpoint() is None  # noqa: SLF001


def test_exhausting_the_pool_raises_rather_than_hanging(monkeypatch) -> None:
    """The failure has to be loud: waiting cannot help once every key is spent."""
    client = _client(monkeypatch, ["a"])
    client._retire(client._endpoints[0], "spent")  # noqa: SLF001

    try:
        client.complete([{"role": "user", "content": "hi"}], model="m")
    except DailyQuotaExhausted as exc:
        assert "every key" in str(exc)
    else:  # pragma: no cover - the call must not succeed
        raise AssertionError("an exhausted pool should raise")


def test_a_single_key_still_works(monkeypatch) -> None:
    """The common case must not need a list."""
    client = _client(monkeypatch, ["only-one"])
    assert client.live_keys == 1
    assert client._pick_endpoint() is not None  # noqa: SLF001


def test_daily_and_per_minute_limits_are_told_apart() -> None:
    daily = Exception("429 ... on tokens per day (TPD): Limit 200000, Used 199000")
    minute = Exception("429 ... on tokens per minute (TPM): Limit 8000, Used 7900")
    assert _is_daily_quota(daily)
    assert not _is_daily_quota(minute)
