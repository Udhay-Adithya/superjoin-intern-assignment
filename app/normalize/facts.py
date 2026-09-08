"""Turn an extracted candidate into a normalized, comparable fact.

This is where the model's output stops being text and becomes something the
comparison engine can reason over: scales resolved, periods turned into
intervals, entities and metrics mapped onto canonical registry entries, and both
comparison keys computed.

Nothing here calls a language model. Everything is deterministic and testable,
which is the point -- a weaker extractor loses recall, not correctness.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from app.extract.extractor import CandidateFact
from app.normalize.canonical import (
    Registry,
    canonical_entity_name,
    canonical_metric_name,
    normalize_phrase,
)
from app.normalize.numbers import Quantity, normalize_quantity
from app.normalize.periods import Period, parse_period
from app.reason.keys import core_key, full_key

# Scope qualifiers seen across filings. Matched as substrings of whatever the
# document wrote, so "Consolidated - FY ended" resolves to "consolidated".
_BASIS_TERMS = ("consolidated", "standalone", "unconsolidated", "combined", "segment")

_MEASURE_VARIANTS = (
    "real", "nominal", "constant prices", "current prices",
    "gross", "net", "adjusted", "reported",
)


@dataclass
class NormalizedFact:
    """A candidate fact after normalization, ready to be stored and compared."""

    candidate: CandidateFact

    entity_id: int | None
    entity_name: str
    metric_id: int | None
    metric_name: str

    quantity: Quantity
    period: Period | None

    basis: str | None
    modality: str
    variant: str | None

    core_key: str
    full_key: str

    @property
    def value_norm(self) -> float:
        return self.quantity.value

    @property
    def period_inferred(self) -> bool:
        return bool(self.period and self.period.inferred)


class NormalizationError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def canonical_basis(raw: str | None) -> str | None:
    """Reduce a basis qualifier to a canonical term.

    Documents write the qualifier as part of a longer header, for example
    "Consolidated - FY ended", so the term is matched inside the string rather
    than against the whole of it.
    """
    if not raw:
        return None
    lowered = raw.lower()
    for term in _BASIS_TERMS:
        if term in lowered:
            return term
    return None


# Financial statements name a measure and then qualify what it counts with a
# prepositional tail: "revenue from operations", "revenue for services",
# "provision for tax".
_VARIANT_SEPARATORS = (" from ", " for ")

# A tail that starts with a determiner is grammar, not a qualifier: "loss for
# the year" names a period, whereas "revenue for services" names what is being
# counted. Splitting the former would strip "loss" of the only word that says
# which loss it is.
_TAIL_STOPWORDS = frozenset({"the", "a", "an", "this", "that", "these", "those", "each"})


def split_metric_variant(metric: str) -> tuple[str, str | None]:
    """Separate a measure from the tail that says what it counts.

    "revenue from operations" becomes ("revenue", "operations") and "revenue
    from customers" becomes ("revenue", "customers"). Both then share a metric,
    so they land in the same cluster and can be compared -- while the differing
    variant keeps them from being treated as interchangeable.

    Leaving the tail inside the metric was why the annual report's
    "revenue from operations" never met the earnings deck's "revenue from
    customers", even though both report the same 8,142 crore for FY24.
    """
    lowered = normalize_phrase(metric)
    for separator in _VARIANT_SEPARATORS:
        if separator in lowered:
            head, tail = lowered.split(separator, 1)
            head, tail = head.strip(), tail.strip()
            # Only the leading word of the tail: "customers (A+B)" and
            # "customers" must reduce to the same variant.
            if head and tail:
                first = tail.split(" ")[0]
                if first in _TAIL_STOPWORDS:
                    return lowered, None
                return head, first
    return lowered, None


def canonical_variant(raw: str | None, metric: str = "") -> str | None:
    """Reduce a measure variant to a canonical term.

    A variant changes what is counted, so it must not be normalized away.
    "Revenue from services" excludes traded goods while "revenue from customers"
    includes them; treating them as one metric would manufacture a contradiction
    out of a definitional difference.
    """
    haystack = f"{raw or ''} {metric or ''}".lower()
    for term in _MEASURE_VARIANTS:
        if term in haystack:
            return term

    explicit = normalize_phrase(raw or "")
    if explicit:
        return explicit.split(" ")[0]

    _, tail = split_metric_variant(metric)
    return tail


def is_entity_like(subject: str, metric: str) -> bool:
    """Is this really the thing being measured, or the measurement itself?

    Models routinely fill ``subject`` with the metric when a sentence has no
    explicit actor -- the IMF's projection came back with subject "real GDP
    growth" rather than "India". That silently creates a bogus entity, and
    because the entity is part of the core key it stops the fact ever being
    compared with the same figure from another source.
    """
    subject_key = canonical_metric_name(subject)
    metric_key = canonical_metric_name(metric)
    if not subject_key:
        return False
    if subject_key == metric_key:
        return False
    # "revenue from operations" as a subject when the metric is "revenue" -- one
    # phrase contains the other, so it is a restatement of the measure.
    return not (subject_key in metric_key or metric_key in subject_key)


def resolve_entity(
    raw: str, *, registry: Registry, default_entity: str = "", metric: str = ""
) -> tuple[int | None, str]:
    """Resolve an entity mention, falling back to the document's subject.

    Filings refer to their own subject as "the Company" or "your Company".
    Those carry no identity of their own, so they resolve to whichever entity
    the document is about -- as does a subject that merely restates the metric.
    """
    canonical = canonical_entity_name(raw) if is_entity_like(raw, metric) else ""
    if not canonical:
        canonical = canonical_entity_name(default_entity)
    if not canonical:
        return None, ""

    resolution = registry.resolve(raw or default_entity, canonical=canonical, kind="other")
    if resolution is None:
        return None, canonical
    return resolution.id, resolution.canonical_name


def resolve_metric(
    raw: str, *, registry: Registry, unit_class: str
) -> tuple[int | None, str]:
    canonical = canonical_metric_name(raw)
    if not canonical:
        return None, ""
    resolution = registry.resolve(raw, canonical=canonical, unit_class=unit_class)
    if resolution is None:
        return None, canonical
    return resolution.id, resolution.canonical_name


def normalize_candidate(
    candidate: CandidateFact,
    *,
    conn: sqlite3.Connection,
    default_entity: str = "",
) -> NormalizedFact:
    """Normalize one grounded candidate. Raises on anything uncomparable."""
    quantity = normalize_quantity(candidate.value_raw, candidate.unit_raw)
    if quantity is None:
        raise NormalizationError(f"unparseable value {candidate.value_raw!r}")

    period = parse_period(candidate.period_raw)

    entity_registry = Registry(conn, "entities")
    metric_registry = Registry(conn, "metrics")

    entity_id, entity_name = resolve_entity(
        candidate.subject,
        registry=entity_registry,
        default_entity=default_entity,
        metric=candidate.metric,
    )
    if not entity_name:
        raise NormalizationError("no resolvable entity")

    # The qualifying tail moves into `variant`, so the measure itself is what
    # gets resolved and clustered.
    metric_head, _ = split_metric_variant(candidate.metric)
    metric_id, metric_name = resolve_metric(
        metric_head or candidate.metric,
        registry=metric_registry,
        unit_class=quantity.unit_class,
    )
    if not metric_name:
        raise NormalizationError("no resolvable metric")

    basis = canonical_basis(candidate.basis_raw)
    variant = canonical_variant(candidate.variant_raw, candidate.metric)

    core = core_key(
        entity=entity_name,
        metric=metric_name,
        period=period,
        period_raw=candidate.period_raw,
    )
    full = full_key(core, basis=basis, modality=candidate.modality, variant=variant)

    return NormalizedFact(
        candidate=candidate,
        entity_id=entity_id,
        entity_name=entity_name,
        metric_id=metric_id,
        metric_name=metric_name,
        quantity=quantity,
        period=period,
        basis=basis,
        modality=candidate.modality,
        variant=variant,
        core_key=core,
        full_key=full,
    )


def scale_multiplier(quantity: Quantity, value_raw: str) -> float:
    """Recover the multiplier applied during normalization.

    The comparison engine needs it to work out what precision the written figure
    claimed: "8,142" meaning crore implies a far coarser measurement than
    "8,142" meaning rupees.
    """
    from app.normalize.numbers import parse_number

    raw = parse_number(value_raw)
    if not raw:
        return 1.0
    ratio = quantity.value / raw
    return ratio if ratio > 0 else 1.0
