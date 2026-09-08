"""Every string here is copied verbatim out of the starter PDFs."""

from __future__ import annotations

import pytest

from app.normalize.numbers import normalize_quantity, parse_number


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("81,415.38", 81415.38),
        ("74,540.82", 74540.82),
        ("8,142", 8142.0),
        ("6.5", 6.5),
        ("18,793", 18793.0),
        ("500.40", 500.40),
        # accounting parentheses mean negative
        ("(940.69)", -940.69),
        ("(2,307.33)", -2307.33),
        ("(10,077.79)", -10077.79),
        # dashes are absent values, not zeros
        ("-", None),
        ("–", None),
        ("", None),
    ],
)
def test_parse_number(text: str, expected: float | None) -> None:
    assert parse_number(text) == expected


def test_case_one_corroboration_survives_unit_conversion() -> None:
    """The load-bearing test.

    Annual report prose/table say "Rs 81,415.38 million"; the Q4 deck says "8,142"
    under a "Rs Cr" header. Nothing downstream can call these the same fact until
    both reduce to the same magnitude.
    """
    annual_report = normalize_quantity("81,415.38", "₹ in Million")
    earnings_deck = normalize_quantity("8,142", "₹ Cr")

    assert annual_report is not None and earnings_deck is not None
    assert annual_report.currency == earnings_deck.currency == "INR"

    relative_difference = abs(annual_report.value - earnings_deck.value) / earnings_deck.value
    assert relative_difference < 0.001, "rounding to whole crore must still corroborate"


def test_standalone_and_consolidated_are_genuinely_different_numbers() -> None:
    """Case 3 is only interesting because the magnitudes really do differ."""
    standalone = normalize_quantity("74,540.82", "₹ in Million")
    consolidated = normalize_quantity("81,415.38", "₹ in Million")

    assert standalone is not None and consolidated is not None
    relative_difference = abs(standalone.value - consolidated.value) / consolidated.value
    assert relative_difference > 0.05, "these must not be mistaken for the same value"


def test_percent_never_takes_a_magnitude_scale() -> None:
    for text, unit in [("6.5", "per cent"), ("6.6", "percent"), ("39.34", "%")]:
        q = normalize_quantity(text, unit)
        assert q is not None
        assert q.unit == "PERCENT"
        assert q.scale is None
        assert q.unit_class == "percent"


def test_percent_is_not_scaled_by_a_stray_scale_word() -> None:
    """"...grew 6.4 per cent in million-tonne terms" must not multiply by 1e6."""
    q = normalize_quantity("6.4", "per cent of million tonnes")
    assert q is not None
    assert q.value == 6.4


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        ("8,142", "₹ Cr", 8142 * 1e7),
        ("81,415.38", "₹ in Million", 81415.38 * 1e6),
        ("500.40", "₹ million", 500.40 * 1e6),
        ("1.5", "lakh", 1.5 * 1e5),
        ("2.3", "billion", 2.3 * 1e9),
    ],
)
def test_scale_multipliers(value: str, unit: str, expected: float) -> None:
    q = normalize_quantity(value, unit)
    assert q is not None
    assert q.value == pytest.approx(expected)


def test_crore_beats_cr_prefix_matching() -> None:
    assert normalize_quantity("10", "crore").value == pytest.approx(10 * 1e7)  # type: ignore[union-attr]
    assert normalize_quantity("10", "Cr").value == pytest.approx(10 * 1e7)  # type: ignore[union-attr]


def test_counts_have_no_currency() -> None:
    q = normalize_quantity("18,793", "PIN codes")
    assert q is not None
    assert q.unit == "COUNT"
    assert q.currency is None
