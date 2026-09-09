"""Every label here is copied verbatim out of the starter PDFs."""

from __future__ import annotations

from datetime import date

import pytest

from app.normalize.periods import (
    KIND_CALENDAR_QUARTER,
    KIND_FISCAL_QUARTER,
    KIND_FISCAL_YEAR,
    parse_period,
)


@pytest.mark.parametrize(
    ("label", "start", "end"),
    [
        ("FY24", date(2023, 4, 1), date(2024, 3, 31)),
        ("FY23", date(2022, 4, 1), date(2023, 3, 31)),
        ("FY 2023-24", date(2023, 4, 1), date(2024, 3, 31)),
        ("FY2023-24", date(2023, 4, 1), date(2024, 3, 31)),
        # the IMF writes its fiscal years with a slash
        ("FY2025/26", date(2025, 4, 1), date(2026, 3, 31)),
        ("FY2024/25", date(2024, 4, 1), date(2025, 3, 31)),
        # RBI and the Economic Survey drop the FY prefix entirely
        ("2024-25", date(2024, 4, 1), date(2025, 3, 31)),
        # the annual report spells it out
        ("FY ended March 31, 2024", date(2023, 4, 1), date(2024, 3, 31)),
        ("financial year ended March 31, 2023", date(2022, 4, 1), date(2023, 3, 31)),
    ],
)
def test_fiscal_years(label: str, start: date, end: date) -> None:
    p = parse_period(label)
    assert p is not None, label
    assert (p.start, p.end) == (start, end)
    assert p.kind == KIND_FISCAL_YEAR


@pytest.mark.parametrize(
    ("label", "start", "end"),
    [
        ("Q4 FY24", date(2024, 1, 1), date(2024, 3, 31)),
        ("Q3 FY24", date(2023, 10, 1), date(2023, 12, 31)),
        ("Q1 FY25", date(2024, 4, 1), date(2024, 6, 30)),
        ("Q2 FY25", date(2024, 7, 1), date(2024, 9, 30)),
        ("Q4 FY23", date(2023, 1, 1), date(2023, 3, 31)),
    ],
)
def test_fiscal_quarters(label: str, start: date, end: date) -> None:
    p = parse_period(label)
    assert p is not None, label
    assert (p.start, p.end) == (start, end)
    assert p.kind == KIND_FISCAL_QUARTER


def test_half_years() -> None:
    p = parse_period("H1 FY25")
    assert p is not None
    assert (p.start, p.end) == (date(2024, 4, 1), date(2024, 9, 30))


def test_imf_calendar_quarter_is_not_a_fiscal_quarter() -> None:
    p = parse_period("2025Q2")
    assert p is not None
    assert p.kind == KIND_CALENDAR_QUARTER
    assert (p.start, p.end) == (date(2025, 4, 1), date(2025, 6, 30))


def test_case_four_the_fiscal_calendar_quarter_collision() -> None:
    """The trap this module exists to defuse.

    The IMF's "2025Q2" and an Indian source's "Q1 FY26" name the same three
    months. A system that aligns periods by matching the digit after "Q" would
    compare April-June against July-September and invent a contradiction.

    Normalizing to intervals makes the two labels share an identity key, and
    makes the naive pairing visibly wrong.
    """
    imf = parse_period("2025Q2")
    indian = parse_period("Q1 FY26")
    assert imf is not None and indian is not None

    assert imf.key == indian.key, "same three months must normalize to one identity"

    naive_string_match = parse_period("Q2 FY26")
    assert naive_string_match is not None
    assert naive_string_match.key != imf.key
    assert not naive_string_match.overlaps(imf), "the naive pairing shares no days at all"


def test_instants() -> None:
    p = parse_period("as at March 31, 2024")
    assert p is not None
    assert p.kind == "instant"
    assert p.start == p.end == date(2024, 3, 31)


def test_assumed_conventions_are_flagged_not_hidden() -> None:
    """A bare year needs a convention we cannot read off the string."""
    bare_year = parse_period("2025")
    assert bare_year is not None
    assert bare_year.inferred is True

    bare_span = parse_period("2024-25")
    assert bare_span is not None
    assert bare_span.inferred is True

    explicit = parse_period("FY24")
    assert explicit is not None
    assert explicit.inferred is False


def test_overlap_detection() -> None:
    q4fy24 = parse_period("Q4 FY24")
    fy24 = parse_period("FY24")
    fy23 = parse_period("FY23")
    assert q4fy24 is not None and fy24 is not None and fy23 is not None

    assert q4fy24.overlaps(fy24), "a quarter falls inside its own fiscal year"
    assert not q4fy24.overlaps(fy23)


def test_unparseable_returns_none_rather_than_guessing() -> None:
    assert parse_period("the current fiscal") is None
    assert parse_period("") is None
    assert parse_period(None) is None


def test_non_consecutive_spans_are_rejected() -> None:
    """Found by scanning the corpus, not by imagination.

    The FY24 annual report contains 21 URL path segments of the form
    ``.../uploads/2023/04/...``. Read as a fiscal span that becomes a year
    ending in 2104. A fiscal span covers two consecutive years; enforcing that
    rejects the URL fragment without needing to know it is a URL.
    """
    assert parse_period("2023/04") is None
    assert parse_period("2019/07") is None

    # genuine consecutive spans still parse, including the century rollover
    assert parse_period("2024-25") is not None
    assert parse_period("FY2025/26") is not None
    rollover = parse_period("1999-00")
    assert rollover is not None
    assert rollover.end == date(2000, 3, 31)


@pytest.mark.parametrize(
    ("label", "start", "end"),
    [
        ("September 2024", date(2024, 9, 1), date(2024, 9, 30)),
        ("end of January 2024", date(2024, 1, 1), date(2024, 1, 31)),
        ("March 2025", date(2025, 3, 1), date(2025, 3, 31)),
    ],
)
def test_a_month_is_a_month_not_a_year(label: str, start: date, end: date) -> None:
    """Found by the engine reporting a false contradiction.

    The IMF reports FX reserves for "September 2024" and the Economic Survey for
    "end of January 2024". With no month pattern both collapsed onto calendar
    year 2024, so two figures about different months were compared as if they
    described the same period.
    """
    p = parse_period(label)
    assert p is not None, label
    assert (p.start, p.end) == (start, end)


def test_a_day_first_date_is_still_an_instant() -> None:
    """"as on 3 January 2025" -- day before month, as Indian sources write it."""
    p = parse_period("as on 3 January 2025")
    assert p is not None
    assert p.start == p.end == date(2025, 1, 3)


def test_different_months_do_not_share_a_period() -> None:
    september = parse_period("September 2024")
    january = parse_period("end of January 2024")
    assert september is not None and january is not None
    assert september.key != january.key
    assert not september.overlaps(january)
