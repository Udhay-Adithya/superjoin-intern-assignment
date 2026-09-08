"""Decide how two facts relate. No language model is involved in the verdict.

Tolerance is derived from the documents rather than tuned by hand. A figure
written as "8,142" in crore claims precision to the nearest crore; one written
as "81,415.38" in million claims precision to the nearest ten thousand rupees.
Two values agree when their difference fits inside the rounding interval that
their own notation implies. That is why the annual report's 81,415.38 million
and the earnings deck's 8,142 crore corroborate: they differ by 4.6 million
rupees, which is less than the half-crore the coarser figure was rounded to.

Contradiction is reserved for claims that purport to describe the same settled
fact. Two institutions forecasting different growth rates for a year that has
not happened yet are not contradicting each other -- they disagree, which is a
different thing, and the system says so rather than picking a winner.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from app.normalize.periods import Period
from app.reason.keys import full_key, has_inferred_qualifier, qualifiers_differ

CORROBORATES = "corroborates"
CONTRADICTS = "contradicts"
RECONCILED = "reconciled"

# Absorbs differences too small to be meaningful even when notation implies
# more precision than the source really has.
RELATIVE_FLOOR = 0.001  # 0.1%

_DECIMALS = re.compile(r"\.(\d+)")


@dataclass(frozen=True)
class ComparableFact:
    """A normalized fact, ready to be compared."""

    id: int
    entity: str
    metric: str
    period: Period | None
    value_norm: float
    unit: str
    unit_class: str  # currency | percent | count
    value_raw: str
    scale_multiplier: float
    basis: str | None
    modality: str
    variant: str | None
    doc_id: int
    core_key: str
    block_id: int = 0
    unit_raw: str = ""

    # provenance, used for explanation and trust ordering
    publisher: str = ""
    published_date: str | None = None
    source_tier: int = 50
    page_no: int = 0
    quote: str = ""

    @property
    def full_key(self) -> str:
        return full_key(
            self.core_key, basis=self.basis, modality=self.modality, variant=self.variant
        )

    def as_qualifier_dict(self) -> dict:
        return {"basis": self.basis, "modality": self.modality, "variant": self.variant}


@dataclass(frozen=True)
class Relation:
    fact_a: int
    fact_b: int
    verdict: str
    rule: str
    explanation: str
    delta: float | None = None
    tolerance: float | None = None
    qualifier_inferred: bool = False
    needs_adjudication: bool = False


def implied_precision(value_raw: str, scale_multiplier: float) -> float:
    """Half-width of the rounding interval a written number implies.

    "8,142" in crore was rounded to the nearest crore, so it stands for anything
    in a half-crore band either side. "81,415.38" in million claims four more
    digits of precision and gets a correspondingly narrow band.
    """
    match = _DECIMALS.search(value_raw or "")
    decimals = len(match.group(1)) if match else 0
    return 0.5 * (10.0**-decimals) * max(scale_multiplier, 1.0)


def implied_step(value_raw: str, scale_multiplier: float) -> float:
    """The size of the last place a number was written to."""
    return 2.0 * implied_precision(value_raw, scale_multiplier)


def tolerance_between(a: ComparableFact, b: ComparableFact) -> float:
    """The coarser of the two notations -- the precision a comparison can claim."""
    return max(
        implied_step(a.value_raw, a.scale_multiplier),
        implied_step(b.value_raw, b.scale_multiplier),
    )


def values_agree(a: ComparableFact, b: ComparableFact) -> tuple[bool, float, float]:
    """Return (agrees, absolute difference, tolerance applied).

    Two values agree when, rounded to the coarser of the two precisions, they
    are the same number. Comparing rounding *bands* for overlap instead would
    make any two adjacent one-decimal percentages agree -- 6.5% and 6.6% both
    touch 6.55 -- which would quietly erase the difference between two
    institutions' published forecasts. Rounding to a common precision also
    keeps the comparison in integers, so it does not turn on floating-point
    representation.
    """
    delta = abs(a.value_norm - b.value_norm)
    step = tolerance_between(a, b)

    if step > 0 and round(a.value_norm / step) == round(b.value_norm / step):
        return True, delta, step

    # Absorb differences too small to be meaningful even when the notation
    # claims more precision than the underlying figure really has.
    magnitude = max(abs(a.value_norm), abs(b.value_norm))
    if magnitude and delta <= magnitude * RELATIVE_FLOOR:
        return True, delta, step

    return False, delta, step


def _with_unit(value: str, unit: str) -> str:
    """Append a unit unless the value already carries one.

    Extractors often include the unit in the value -- "6.6 percent" rather than
    "6.6" -- and appending blindly produced "6.6 percent %" in explanations
    a reader is meant to trust.
    """
    value = (value or "").strip()
    unit = (unit or "").strip()
    if not unit:
        return value
    if unit.lower() in value.lower():
        return value
    if unit == "%" and re.search(r"per\s*cent|percent|%", value, re.IGNORECASE):
        return value
    return f"{value} {unit}"


def _format(fact: ComparableFact) -> str:
    unit = "%" if fact.unit_class == "percent" else (fact.unit or "")
    return _with_unit(fact.value_raw, unit)


def _written(fact: ComparableFact) -> str:
    """The figure as the document wrote it, including its scale.

    Reporting only the normalized magnitude would hide the very thing that makes
    a corroboration interesting -- that "81,415.38 million" and "8,142 Cr" are
    the same quantity in different notation.
    """
    unit = (fact.unit_raw or "").strip()
    if not unit and fact.unit_class == "percent":
        unit = "%"
    return _with_unit(fact.value_raw, unit)


def _qualifier_phrase(fact: ComparableFact, field: str) -> str:
    return str(getattr(fact, field, None) or "unstated")


def compare(a: ComparableFact, b: ComparableFact) -> Relation | None:
    """Classify the relationship between two facts sharing a core key.

    Returns ``None`` when the two are not meaningfully comparable at all.
    """
    if a.id == b.id or a.core_key != b.core_key:
        return None

    # Comparing a percentage against a rupee amount is a category error, not a
    # contradiction.
    if a.unit_class != b.unit_class:
        return None
    if a.unit_class == "currency" and a.unit != b.unit:
        return None  # cross-currency needs an FX rate and an as-of date

    differing = qualifiers_differ(a.as_qualifier_dict(), b.as_qualifier_dict())
    inferred = has_inferred_qualifier(a.as_qualifier_dict(), b.as_qualifier_dict())
    agrees, delta, tolerance = values_agree(a, b)

    # --- same thing, measured differently -------------------------------
    if differing:
        if agrees:
            return Relation(
                a.id, b.id, CORROBORATES, "agrees_despite_differing_qualifiers",
                explanation=(
                    f"Same quantity, written differently: {_written(a)} and {_written(b)} "
                    f"are the same figure once scaled, for the same period, "
                    f"despite differing on {', '.join(differing)}."
                ),
                delta=delta, tolerance=tolerance, qualifier_inferred=inferred,
            )

        field = differing[0]
        return Relation(
            a.id, b.id, RECONCILED, f"different_{field}",
            explanation=(
                f"Not a contradiction: these measure different things. "
                f"{_format(a)} is {_qualifier_phrase(a, field)} while "
                f"{_format(b)} is {_qualifier_phrase(b, field)}."
            ),
            delta=delta, tolerance=tolerance, qualifier_inferred=inferred,
        )

    # A document does not contradict itself inside a single extracted region.
    # When two figures from one block share a full key and disagree, the metric
    # labels collapsed two different things -- a before-and-after pair in one
    # sentence, or two share classes distinguished only by face value. Reporting
    # that as a contradiction would blame the document for our own resolution.
    if not agrees and a.block_id and a.block_id == b.block_id:
        return Relation(
            a.id, b.id, RECONCILED, "metric_label_collision",
            explanation=(
                f"Same source passage, so not a document conflict: {_format(a)} and "
                f"{_format(b)} were both read as '{a.metric}'. They are different "
                f"quantities whose distinguishing detail was lost in extraction."
            ),
            delta=delta, tolerance=tolerance, qualifier_inferred=inferred,
            needs_adjudication=True,
        )

    # --- same full key: any difference is a real disagreement ------------
    if agrees:
        return Relation(
            a.id, b.id, CORROBORATES, "within_implied_precision",
            explanation=(
                f"{_written(a)} and {_written(b)} agree to within {tolerance:,.4g}, "
                f"the precision their own notation implies."
            ),
            delta=delta, tolerance=tolerance, qualifier_inferred=inferred,
        )

    # A forecast of an unsettled future is an opinion, not a claim about a fact
    # that can be checked. Two of them differing is disagreement, not error.
    if a.modality == "projected" and b.modality == "projected":
        return Relation(
            a.id, b.id, RECONCILED, "differing_forecast",
            explanation=(
                f"Competing forecasts, not a contradiction: "
                f"{_source(a)} projects {_format(a)} and {_source(b)} projects {_format(b)} "
                f"for the same period. Forecasts differ by method and vintage."
            ),
            delta=delta, tolerance=tolerance, qualifier_inferred=inferred,
            needs_adjudication=True,
        )

    return Relation(
        a.id, b.id, CONTRADICTS, "value_mismatch",
        explanation=(
            f"{_source(a)} reports {_format(a)} but {_source(b)} reports {_format(b)} "
            f"for the same entity, metric, period and basis. They differ by "
            f"{delta:,.4g}, beyond the {tolerance:,.4g} their notation allows."
        ),
        delta=delta, tolerance=tolerance, qualifier_inferred=inferred,
        needs_adjudication=True,
    )


def _source(fact: ComparableFact) -> str:
    if fact.publisher and fact.published_date:
        return f"{fact.publisher} ({fact.published_date[:10]})"
    return fact.publisher or f"document {fact.doc_id}"


def compare_cluster(facts: list[ComparableFact]) -> list[Relation]:
    """Compare every pair within one core-key cluster."""
    relations: list[Relation] = []
    for i, a in enumerate(facts):
        for b in facts[i + 1 :]:
            relation = compare(a, b)
            if relation is not None:
                relations.append(relation)
    return relations


def prefer(a: ComparableFact, b: ComparableFact) -> ComparableFact:
    """Which fact to surface when two conflict. Neither is ever deleted.

    Ordering: a more authoritative source first, then the later vintage, on the
    principle that a restatement supersedes what it restates.
    """
    if a.source_tier != b.source_tier:
        return a if a.source_tier > b.source_tier else b

    a_date = _parse_date(a.published_date)
    b_date = _parse_date(b.published_date)
    if a_date and b_date and a_date != b_date:
        return a if a_date > b_date else b

    return a


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None
