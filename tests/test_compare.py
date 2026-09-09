"""The four required cases, written as executable specifications.

Every figure is real and taken from the starter documents.
"""

from __future__ import annotations

from dataclasses import replace

from app.normalize.numbers import normalize_quantity
from app.normalize.periods import parse_period
from app.reason.compare import (
    CONTRADICTS,
    CORROBORATES,
    RECONCILED,
    ComparableFact,
    compare,
    implied_precision,
    prefer,
)
from app.reason.keys import core_key

FY24 = parse_period("FY24")
FY26 = parse_period("FY2025/26")


def make(
    fid: int,
    value_raw: str,
    unit_raw: str,
    *,
    metric: str = "revenue from operations",
    entity: str = "delhivery",
    period=FY24,
    basis: str | None = None,
    modality: str = "actual",
    variant: str | None = None,
    doc_id: int = 1,
    publisher: str = "",
    published_date: str | None = None,
    source_tier: int = 50,
) -> ComparableFact:
    quantity = normalize_quantity(value_raw, unit_raw)
    assert quantity is not None
    scale = quantity.value / float(value_raw.replace(",", "")) if float(
        value_raw.replace(",", "")
    ) else 1.0
    return ComparableFact(
        id=fid,
        entity=entity,
        metric=metric,
        period=period,
        value_norm=quantity.value,
        unit=quantity.unit,
        unit_class=quantity.unit_class,
        value_raw=value_raw,
        scale_multiplier=scale,
        basis=basis,
        modality=modality,
        variant=variant,
        doc_id=doc_id,
        core_key=core_key(entity=entity, metric=metric, period=period),
        publisher=publisher,
        published_date=published_date,
        source_tier=source_tier,
    )


# --- case 1 -------------------------------------------------------------


def test_case_one_corroboration_across_units_and_documents() -> None:
    """Annual report "Rs 81,415.38 million" vs deck "8,142" under a "Rs Cr" header."""
    annual = make(1, "81,415.38", "₹ in Million", basis="consolidated", doc_id=1)
    deck = make(2, "8,142", "₹ Cr", basis="consolidated", doc_id=2)

    relation = compare(annual, deck)
    assert relation is not None
    assert relation.verdict == CORROBORATES
    assert relation.rule == "within_implied_precision"


def test_corroboration_is_earned_not_assumed() -> None:
    """The tolerance must come from notation, not from being generous.

    8,142 crore is rounded to the nearest crore, so it stands for a half-crore
    band. The two figures differ by 4.62 million rupees, inside that band.
    """
    assert implied_precision("8,142", 1e7) == 5e6
    assert implied_precision("81,415.38", 1e6) == 5_000.0


def test_a_genuinely_different_figure_does_not_corroborate() -> None:
    annual = make(1, "81,415.38", "₹ in Million", basis="consolidated")
    wrong = make(2, "8,500", "₹ Cr", basis="consolidated")

    relation = compare(annual, wrong)
    assert relation is not None
    assert relation.verdict == CONTRADICTS


# --- case 3 -------------------------------------------------------------


def test_case_three_standalone_vs_consolidated_is_reconciled() -> None:
    """Same company, metric and year; different numbers; not a contradiction."""
    standalone = make(1, "74,540.82", "₹ in Million", basis="standalone")
    consolidated = make(2, "81,415.38", "₹ in Million", basis="consolidated")

    relation = compare(standalone, consolidated)
    assert relation is not None
    assert relation.verdict == RECONCILED
    assert relation.rule == "different_basis"
    assert "standalone" in relation.explanation.lower()
    assert "consolidated" in relation.explanation.lower()


def test_case_three_variant_difference_is_also_reconciled() -> None:
    """Revenue from services excludes traded goods; revenue from customers does not."""
    services = make(1, "7,224", "₹ Cr", variant="services", period=parse_period("FY23"))
    customers = make(2, "7,225", "₹ Cr", variant="customers", period=parse_period("FY23"))

    relation = compare(services, customers)
    assert relation is not None
    assert relation.verdict in {RECONCILED, CORROBORATES}
    if relation.verdict == RECONCILED:
        assert relation.rule == "different_variant"


# --- case 2 -------------------------------------------------------------


def test_case_two_competing_forecasts_are_not_contradictions() -> None:
    """RBI 6.5% vs IMF 6.6% for FY2025-26.

    Neither observes anything: both are predictions, made by different methods
    at different times. The system reports disagreement and names the sources
    rather than picking a winner. This holds after the period closes too -- the
    documents still record what each institution predicted.
    """
    rbi = make(
        1, "6.5", "per cent", metric="real gdp growth", entity="india", period=FY26,
        modality="projected", publisher="Reserve Bank of India", published_date="2025-05-25",
    )
    imf = make(
        2, "6.6", "percent", metric="real gdp growth", entity="india", period=FY26,
        modality="projected", publisher="IMF", published_date="2025-11-06",
    )

    relation = compare(rbi, imf)
    assert relation is not None
    assert relation.verdict == RECONCILED
    assert relation.rule == "differing_forecast"
    assert "Reserve Bank of India" in relation.explanation
    assert "IMF" in relation.explanation
    assert relation.needs_adjudication


def test_conflicting_actuals_do_contradict() -> None:
    """Two *observations* of the same period that disagree is a real contradiction."""
    survey = make(
        1, "6.4", "per cent", metric="real gdp growth", entity="india", period=FY24,
        modality="actual", publisher="Economic Survey",
    )
    imf = make(
        2, "7.8", "percent", metric="real gdp growth", entity="india", period=FY24,
        modality="actual", publisher="IMF",
    )

    relation = compare(survey, imf)
    assert relation is not None
    assert relation.verdict == CONTRADICTS
    assert relation.needs_adjudication


def test_forecast_and_actual_are_never_compared_as_equals() -> None:
    projected = make(1, "6.5", "per cent", metric="real gdp growth", modality="projected")
    actual = make(2, "6.4", "per cent", metric="real gdp growth", modality="actual")

    relation = compare(projected, actual)
    assert relation is not None
    assert relation.verdict == RECONCILED
    assert relation.rule == "different_modality"


# --- qualifier handling -------------------------------------------------


def test_missing_qualifier_acts_as_a_wildcard_but_is_flagged() -> None:
    """An earnings deck rarely says "consolidated"; it should still corroborate."""
    stated = make(1, "81,415.38", "₹ in Million", basis="consolidated")
    unstated = make(2, "8,142", "₹ Cr", basis=None)

    relation = compare(stated, unstated)
    assert relation is not None
    assert relation.verdict == CORROBORATES
    assert relation.qualifier_inferred is True


def test_incomparable_units_yield_no_relation() -> None:
    percent = make(1, "6.5", "per cent")
    rupees = make(2, "8,142", "₹ Cr")
    assert compare(percent, rupees) is None


def test_facts_about_different_periods_are_never_compared() -> None:
    fy24 = make(1, "81,415.38", "₹ in Million", period=parse_period("FY24"))
    fy23 = make(2, "72,253.01", "₹ in Million", period=parse_period("FY23"))
    assert compare(fy24, fy23) is None


# --- trust ordering -----------------------------------------------------


def test_more_authoritative_source_is_preferred() -> None:
    audited = make(1, "81,415.38", "₹ in Million", source_tier=90)
    deck = make(2, "8,142", "₹ Cr", source_tier=40)
    assert prefer(audited, deck).id == 1
    assert prefer(deck, audited).id == 1


def test_later_vintage_wins_at_equal_authority() -> None:
    older = make(1, "6.4", "per cent", published_date="2025-01-30", source_tier=70)
    newer = make(2, "6.5", "per cent", published_date="2025-11-06", source_tier=70)
    assert prefer(older, newer).id == 2


def test_same_block_disagreement_is_a_label_collision_not_a_contradiction() -> None:
    """Found in a live run, not imagined.

    Page 23 of the FY24 annual report lists "300,000 Preference Shares of Rs 10
    each" and "4,660,337 Preference Shares of Rs 100 each". Both normalized to
    the metric "preference shares authorised", and the engine called them a
    contradiction. They are two share classes in one list; a document does not
    contradict itself inside a single passage.
    """
    small = make(1, "300,000", "", metric="preference shares authorised")
    large = make(2, "4,660,337", "", metric="preference shares authorised")
    small = replace(small, block_id=287)
    large = replace(large, block_id=287)

    relation = compare(small, large)
    assert relation is not None
    assert relation.verdict == RECONCILED
    assert relation.rule == "metric_label_collision"
    assert relation.needs_adjudication


def test_the_same_disagreement_across_blocks_is_still_a_contradiction() -> None:
    """The collision rule must not suppress genuine cross-source conflicts."""
    a = make(1, "300,000", "", metric="preference shares authorised")
    b = make(2, "4,660,337", "", metric="preference shares authorised", doc_id=2)
    a = replace(a, block_id=10)
    b = replace(b, block_id=99)

    relation = compare(a, b)
    assert relation is not None
    assert relation.verdict == CONTRADICTS


def test_unparseable_period_labels_still_separate_facts() -> None:
    """Found in the UI, on a real RBI sentence.

    "enhancement of credit limit ... from Rs 3 lakh to Rs 5 lakh" gives two
    values the extractor labels "before" and "after". Neither parses into an
    interval, and treating both as period-unknown put them in one cluster where
    they looked like a conflict. The labels differ, so the facts differ.
    """
    before = core_key(entity="rbi", metric="credit limit", period=None, period_raw="before")
    after = core_key(entity="rbi", metric="credit limit", period=None, period_raw="after")
    assert before != after

    # Two facts that genuinely state no period at all still share a key.
    blank_a = core_key(entity="rbi", metric="credit limit", period=None, period_raw="")
    blank_b = core_key(entity="rbi", metric="credit limit", period=None, period_raw="")
    assert blank_a == blank_b
