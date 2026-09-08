"""Grounding gate behaviour, exercised against real layout text."""

from __future__ import annotations

import pytest

from app.extract.ground import GroundingError, ground_fact, locate

# The revenue row as the layout parser emits it: columns held apart by spaces.
SOURCE = (
    "(₹ in Million)\n"
    "Particulars\n"
    "Standalone – FY ended          Consolidated – FY ended\n"
    "March 31, 2024  March 31, 2023  March 31, 2024 March 31, 2023\n"
    "Revenue from Operations                                  74,540.82"
    "       66,586.61       81,415.38       72,253.01\n"
)


def test_exact_quote_is_located() -> None:
    found = locate("81,415.38", SOURCE)
    assert SOURCE[found.char_start : found.char_end] == "81,415.38"


def test_offsets_are_shifted_into_document_coordinates() -> None:
    found = locate("81,415.38", SOURCE, source_offset=55_000)
    assert found.char_start >= 55_000
    assert found.char_end - found.char_start == len("81,415.38")


def test_collapsed_whitespace_still_matches() -> None:
    """Models collapse the column padding when quoting a table row.

    The source holds columns apart with long space runs. A quote that uses
    single spaces is the same evidence and must not be rejected.
    """
    quoted = "Revenue from Operations 74,540.82 66,586.61 81,415.38 72,253.01"
    found = locate(quoted, SOURCE)
    assert "74,540.82" in found.quote
    assert "81,415.38" in found.quote
    # The recovered quote is the source as written, not the model's rendering.
    assert found.quote == SOURCE[found.char_start : found.char_end]


def test_dash_variants_are_folded() -> None:
    """An en dash in the PDF is routinely quoted back as a hyphen."""
    found = locate("Standalone - FY ended", SOURCE)
    assert "Standalone" in found.quote


def test_value_must_appear_in_its_own_quote() -> None:
    """The check that catches a real quote paired with an invented number."""
    with pytest.raises(GroundingError, match="not present in its own evidence"):
        ground_fact(
            quote="Revenue from Operations 74,540.82",
            value_raw="99,999.99",
            source=SOURCE,
        )


def test_thousands_separators_do_not_break_the_value_check() -> None:
    grounded = ground_fact(
        quote="Revenue from Operations 74,540.82 66,586.61 81,415.38 72,253.01",
        value_raw="₹81,415.38",
        source=SOURCE,
    )
    assert "81,415.38" in grounded.quote


def test_fabricated_quote_is_rejected() -> None:
    with pytest.raises(GroundingError, match="not found"):
        ground_fact(
            quote="Revenue from Operations stood at ₹90,000.00 million",
            value_raw="90,000.00",
            source=SOURCE,
        )


def test_empty_quote_is_rejected() -> None:
    with pytest.raises(GroundingError, match="empty quote"):
        ground_fact(quote="   ", value_raw="1", source=SOURCE)


def test_non_numeric_facts_skip_the_value_check() -> None:
    """State facts such as a directorship have no figure to verify."""
    grounded = ground_fact(quote="Particulars", value_raw="active", source=SOURCE)
    assert grounded.quote == "Particulars"
