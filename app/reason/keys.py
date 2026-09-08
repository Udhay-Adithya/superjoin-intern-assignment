"""Comparison keys: the mechanism that decides what may be compared with what.

Two levels, and the difference between them produces every verdict the system
can reach.

  core key = (entity, metric, period)
      Groups facts that are *about the same thing*. Two facts sharing a core key
      are worth comparing; two that do not are unrelated and never compared.

  full key = core key + (basis, modality, variant)
      Groups facts that are *directly comparable*. Sharing a full key means any
      difference in value is a real disagreement.

Facts that share a core key but differ on the full key are the interesting case:
they look contradictory to a naive system and are in fact measuring different
things. The qualifier on which their full keys diverge is, by construction, the
explanation.
"""

from __future__ import annotations

from app.normalize.periods import Period

# A missing qualifier is a wildcard rather than a distinct value: an earnings
# deck that says "8,142" without stating "consolidated" should still be able to
# corroborate an annual report figure that does.
WILDCARD = "*"


def _slot(value: str | None) -> str:
    cleaned = (value or "").strip().lower()
    return cleaned or WILDCARD


def core_key(*, entity: str, metric: str, period: Period | None) -> str:
    """Identity of the thing being measured, over the period measured."""
    period_part = period.key if period is not None else WILDCARD
    return f"{_slot(entity)}|{_slot(metric)}|{period_part}"


def full_key(
    core: str, *, basis: str | None, modality: str | None, variant: str | None
) -> str:
    """Comparability key: same value expected unless something is wrong."""
    return f"{core}|{_slot(basis)}|{_slot(modality)}|{_slot(variant)}"


def qualifiers_differ(a: dict, b: dict) -> list[str]:
    """Which qualifiers separate two facts, ignoring wildcards.

    A qualifier that is absent on one side does not count as a difference; it is
    unstated, not contradictory. This is what lets an under-specified fact
    corroborate a fully specified one while still recording that we inferred it.
    """
    differing = []
    for field in ("basis", "modality", "variant"):
        left, right = _slot(a.get(field)), _slot(b.get(field))
        if WILDCARD in (left, right):
            continue
        if left != right:
            differing.append(field)
    return differing


def has_inferred_qualifier(a: dict, b: dict) -> bool:
    """True when the pairing relies on treating a missing qualifier as a match."""
    for field in ("basis", "modality", "variant"):
        left, right = _slot(a.get(field)), _slot(b.get(field))
        if (left == WILDCARD) != (right == WILDCARD):
            return True
    return False
