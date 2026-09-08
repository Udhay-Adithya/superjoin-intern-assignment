"""Rate limiter behaviour.

Written after a real starvation bug: reservations were corrected by position in
the window, so one thread's underestimate was charged to whichever request had
most recently been admitted. Under concurrency that inflated arbitrary entries
until nothing fit, and every worker sat waiting for a full minute to elapse.
"""

from __future__ import annotations

import threading
import time

from app.llm.client import TokenBudget


def test_requests_within_budget_do_not_block() -> None:
    budget = TokenBudget(tokens_per_minute=10_000, requests_per_minute=1000)
    started = time.monotonic()
    for _ in range(4):
        budget.acquire(2000)
    assert time.monotonic() - started < 0.5


def test_settle_corrects_the_callers_own_reservation() -> None:
    """The bug: settle must not touch a different thread's entry."""
    budget = TokenBudget(tokens_per_minute=10_000, requests_per_minute=1000)

    first = budget.acquire(1000)
    second = budget.acquire(1000)

    # The first request turned out to be far more expensive than estimated.
    budget.settle(first, 5000)

    assert first[1] == 5000, "the caller's own reservation was not corrected"
    assert second[1] == 1000, "another request's reservation was altered"


def test_over_budget_request_waits_then_proceeds() -> None:
    budget = TokenBudget(tokens_per_minute=1000, requests_per_minute=1000, window_seconds=30)
    budget.acquire(900)

    done = threading.Event()

    def second() -> None:
        budget.acquire(900)  # cannot fit until the first entry expires
        done.set()

    thread = threading.Thread(target=second, daemon=True)
    thread.start()

    assert not done.wait(timeout=0.5), "an over-budget request was admitted immediately"

    # Settling the first request down to a smaller real cost frees capacity.
    budget._window[0][1] = 50.0
    assert done.wait(timeout=5), "capacity freed but the waiter was never admitted"


def test_a_single_request_larger_than_the_limit_still_runs() -> None:
    """Clamped rather than deadlocked: refusing forever would be worse."""
    budget = TokenBudget(tokens_per_minute=500, requests_per_minute=1000)
    started = time.monotonic()
    budget.acquire(50_000)
    assert time.monotonic() - started < 0.5


def test_concurrent_workers_all_finish() -> None:
    """Saturating the budget delays workers; it must never strand them.

    A short window keeps the test fast; the behaviour under a 60-second window
    is the same, just slower.
    """
    budget = TokenBudget(tokens_per_minute=4000, requests_per_minute=1000, window_seconds=0.5)
    finished: list[int] = []
    lock = threading.Lock()

    def worker(index: int) -> None:
        reservation = budget.acquire(1000)
        budget.settle(reservation, 1500)  # every call costs more than estimated
        with lock:
            finished.append(index)

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert len(finished) == 4, f"only {len(finished)} of 4 workers completed"
