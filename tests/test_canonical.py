"""Canonicalization, including the distinctions that must NOT be collapsed."""

from __future__ import annotations

import pytest

from app import db
from app.normalize.canonical import (
    Registry,
    canonical_entity_name,
    canonical_metric_name,
    normalize_phrase,
)
from app.normalize.facts import (
    canonical_basis,
    canonical_variant,
    is_entity_like,
    split_metric_variant,
)


@pytest.fixture
def conn():
    connection = db.connect(":memory:")
    db.init_db(connection)
    yield connection
    connection.close()


# --- entities -----------------------------------------------------------


@pytest.mark.parametrize(
    ("mention", "expected"),
    [
        ("Delhivery Limited", "delhivery"),
        ("Delhivery Ltd.", "delhivery"),
        ("DELHIVERY LIMITED", "delhivery"),
        ("Spoton Logistics Private Limited", "spoton logistics"),
        ("Falcon Autotech Private Limited", "falcon autotech"),
        ("India", "india"),
    ],
)
def test_legal_suffixes_do_not_distinguish_entities(mention: str, expected: str) -> None:
    assert canonical_entity_name(mention) == expected


@pytest.mark.parametrize("anaphor", ["the Company", "your Company", "The Group", "we"])
def test_anaphora_carry_no_identity(anaphor: str) -> None:
    """These must resolve from document context, not become entities."""
    assert canonical_entity_name(anaphor) == ""


# --- metrics ------------------------------------------------------------


def test_emphasis_words_are_normalized_away() -> None:
    assert canonical_metric_name("Total Revenue") == canonical_metric_name("Revenue")


def test_case_and_punctuation_do_not_create_new_metrics() -> None:
    assert canonical_metric_name("Revenue from Operations") == canonical_metric_name(
        "revenue from operations"
    )


# --- the distinction that must survive ----------------------------------


def test_measure_variants_are_kept_apart() -> None:
    """Revenue from services excludes traded goods; revenue from customers does not.

    Collapsing them would turn a definitional difference into a contradiction.
    In this corpus the gap is real: FY23 services was 7,224 crore against 7,225
    crore from customers.
    """
    services = canonical_variant("", "revenue from services")
    customers = canonical_variant("", "revenue from customers")
    assert services and customers
    assert services != customers


def test_real_and_nominal_are_different_measures() -> None:
    assert canonical_variant("real", "GDP growth") == "real"
    assert canonical_variant("nominal", "GDP growth") == "nominal"


# --- basis --------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Consolidated", "consolidated"),
        ("Standalone", "standalone"),
        # the qualifier arrives embedded in a longer column header
        ("Consolidated – FY ended", "consolidated"),
        ("Standalone – FY ended", "standalone"),
        ("", None),
        ("Particulars", None),
    ],
)
def test_basis_is_matched_inside_column_headers(raw: str, expected: str | None) -> None:
    assert canonical_basis(raw) == expected


# --- registry -----------------------------------------------------------


def test_registry_reuses_an_existing_entry(conn) -> None:
    registry = Registry(conn, "entities")
    first = registry.resolve("Delhivery Limited", canonical="delhivery", kind="company")
    second = registry.resolve("Delhivery Ltd", canonical="delhivery", kind="company")

    assert first is not None and second is not None
    assert first.id == second.id
    assert first.created is True
    assert second.created is False


def test_registry_records_surface_forms_as_aliases(conn) -> None:
    registry = Registry(conn, "entities")
    registry.resolve("Delhivery Limited", canonical="delhivery", kind="company")
    registry.resolve("Delhivery Ltd", canonical="delhivery", kind="company")

    aliases = conn.execute(
        "SELECT aliases_json FROM entities WHERE canonical_name = 'delhivery'"
    ).fetchone()["aliases_json"]
    assert "Delhivery Limited" in aliases
    assert "Delhivery Ltd" in aliases


def test_registry_grows_for_unseen_metrics(conn) -> None:
    """New kinds of fact extend the schema instead of being dropped."""
    registry = Registry(conn, "metrics")
    for name in ("revenue", "ebitda margin", "pin codes served"):
        assert registry.resolve(name, canonical=name, unit_class="count") is not None

    count = conn.execute("SELECT COUNT(*) AS n FROM metrics").fetchone()["n"]
    assert count == 3


def test_normalize_phrase_keeps_percent_signs() -> None:
    """Percent is part of the measure, not punctuation to strip."""
    assert "%" in normalize_phrase("growth %")


# --- subject vs metric --------------------------------------------------


def test_a_subject_that_restates_the_metric_is_not_an_entity() -> None:
    """Found in a live run.

    The IMF's FY2025-26 projection came back with subject "real GDP growth"
    instead of "India". Because the entity is part of the core key, that fact
    could never meet the RBI's projection of the same figure.
    """
    assert not is_entity_like("real GDP growth", "real GDP growth")
    assert not is_entity_like("revenue from operations", "revenue")
    assert not is_entity_like("headline inflation", "headline inflation rate")


def test_a_genuine_subject_survives() -> None:
    assert is_entity_like("India", "real GDP growth")
    assert is_entity_like("Delhivery Limited", "revenue from operations")
    assert is_entity_like("Falcon Autotech", "stake")


def test_metric_shaped_subject_falls_back_to_the_document_entity(conn) -> None:
    from app.normalize.facts import resolve_entity

    registry = Registry(conn, "entities")
    _, name = resolve_entity(
        "real GDP growth", registry=registry, default_entity="India", metric="real GDP growth"
    )
    assert name == "india"


# --- metric / variant split ---------------------------------------------


@pytest.mark.parametrize(
    ("metric", "head", "tail"),
    [
        # the split that makes case 1 work across documents
        ("Revenue from Operations", "revenue", "operations"),
        ("Revenue from customers (A+B)", "revenue", "customers"),
        ("Revenue for services (A)", "revenue", "services"),
        ("provision for tax", "provision", "tax"),
        # a determiner means the tail is a period, not a qualifier
        ("Loss for the year", "loss for the year", None),
        ("Profit for the period", "profit for the period", None),
        # nothing to split
        ("Total Income", "total income", None),
        ("Other income", "other income", None),
    ],
)
def test_metric_variant_split(metric: str, head: str, tail: str | None) -> None:
    assert split_metric_variant(metric) == (head, tail)


def test_the_annual_report_and_the_deck_reach_the_same_metric() -> None:
    """Case 1 depends entirely on this.

    The annual report says "revenue from operations" and the earnings deck says
    "revenue from customers". Both report 8,142 crore for FY24. Leaving the tail
    inside the metric gave them different core keys, so they were never compared.
    """
    annual_head, annual_tail = split_metric_variant("Revenue from Operations")
    deck_head, deck_tail = split_metric_variant("Revenue from customers (A+B)")

    assert annual_head == deck_head, "the two must share a metric to be compared"
    assert annual_tail != deck_tail, "but the variant must still tell them apart"
