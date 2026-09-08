"""Turn period labels as written into explicit date intervals.

Six documents from four institutions use at least eight conventions for naming
time: ``FY24``, ``FY 2023-24``, ``FY2025/26``, ``2024-25``, ``Q4 FY24``,
``H1 FY25``, ``FY ended March 31, 2024`` and ``2025Q2``.

The last one is the trap. The IMF's ``2025Q2`` is a *calendar* quarter --
April-June 2025 -- which Indian sources call ``Q1 FY26``. A system that aligns
periods by matching the string "Q2" to "Q2" compares April-June against
July-September and invents a contradiction that does not exist.

Reducing every label to a concrete ``(start, end)`` interval is what makes that
class of error impossible. Where a convention has to be assumed rather than read,
``Period.inferred`` is set so the uncertainty stays visible downstream.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date

# Indian fiscal years end 31 March: FY24 runs 2023-04-01 .. 2024-03-31.
FISCAL_YEAR_END_MONTH = 3

MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10,
    "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}

KIND_FISCAL_YEAR = "fiscal_year"
KIND_FISCAL_QUARTER = "fiscal_quarter"
KIND_FISCAL_HALF = "fiscal_half"
KIND_CALENDAR_YEAR = "calendar_year"
KIND_CALENDAR_QUARTER = "calendar_quarter"
KIND_INSTANT = "instant"


@dataclass(frozen=True)
class Period:
    """A period reduced to the interval it actually covers."""

    start: date
    end: date
    kind: str
    raw: str
    inferred: bool = False

    @property
    def key(self) -> str:
        """Interval identity. Two labels naming the same span share this key.

        ``2025Q2`` and ``Q1 FY26`` both produce ``2025-04-01/2025-06-30``, which
        is precisely the collision this module exists to resolve.
        """
        return f"{self.start.isoformat()}/{self.end.isoformat()}"

    def overlaps(self, other: Period) -> bool:
        return self.start <= other.end and other.start <= self.end


def _end_of_month(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def _four_digit(year: int) -> int:
    """FY24 -> 2024. Adequate for a corpus that starts in 2011."""
    if year >= 100:
        return year
    return 2000 + year


def fiscal_year(end_year: int) -> tuple[date, date]:
    """FY24 -> (2023-04-01, 2024-03-31)."""
    end_year = _four_digit(end_year)
    start = date(end_year - 1, FISCAL_YEAR_END_MONTH + 1, 1)
    end = _end_of_month(end_year, FISCAL_YEAR_END_MONTH)
    return start, end


def fiscal_quarter(end_year: int, quarter: int) -> tuple[date, date]:
    """Q4 FY24 -> (2024-01-01, 2024-03-31)."""
    fy_start, _ = fiscal_year(end_year)
    start_month_index = (quarter - 1) * 3
    year = fy_start.year + (fy_start.month - 1 + start_month_index) // 12
    month = (fy_start.month - 1 + start_month_index) % 12 + 1
    start = date(year, month, 1)
    end_offset = month + 2
    end_year_ = year + (end_offset - 1) // 12
    end_month = (end_offset - 1) % 12 + 1
    return start, _end_of_month(end_year_, end_month)


def fiscal_half(end_year: int, half: int) -> tuple[date, date]:
    """H1 FY25 -> (2024-04-01, 2024-09-30)."""
    start, _ = fiscal_quarter(end_year, 1 if half == 1 else 3)
    _, end = fiscal_quarter(end_year, 2 if half == 1 else 4)
    return start, end


def calendar_quarter(year: int, quarter: int) -> tuple[date, date]:
    """2025Q2 -> (2025-04-01, 2025-06-30). Calendar, not fiscal."""
    year = _four_digit(year)
    start_month = (quarter - 1) * 3 + 1
    return date(year, start_month, 1), _end_of_month(year, start_month + 2)


# Patterns are tried in order; the first match wins, so the most specific
# conventions must come first.
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Q4 FY24 / Q4FY2024 / Q4 FY 24
    (re.compile(r"\bQ([1-4])\s*[-,]?\s*FY\s*'?(\d{2,4})\b", re.I), "fiscal_quarter"),
    # H1 FY25
    (re.compile(r"\bH([12])\s*[-,]?\s*FY\s*'?(\d{2,4})\b", re.I), "fiscal_half"),
    # FY2023-24 / FY 2023-24 / FY2025/26 / FY23-24
    (re.compile(r"\bFY\s*'?(\d{4})\s*[-/]\s*(\d{2,4})\b", re.I), "fiscal_span"),
    (re.compile(r"\bFY\s*'?(\d{2})\s*[-/]\s*(\d{2})\b", re.I), "fiscal_span_short"),
    # FY24 / FY 2024 / FY'24
    (re.compile(r"\bFY\s*'?(\d{2,4})\b", re.I), "fiscal_year"),
    # "FY ended March 31, 2024" -- the annual report's column header -- plus the
    # spelled-out variants. FY must be listed here or it falls through to the
    # bare-date matcher and collapses a whole year into a single instant.
    (
        re.compile(
            r"\b(?:FY|financial\s+year|fiscal\s+year|year|period)\s+end(?:ed|ing)\s+"
            r"(\w+)\s+(\d{1,2}),?\s*(\d{4})",
            re.I,
        ),
        "year_ended",
    ),
    # as at / as of March 31, 2024
    (re.compile(r"\bas\s+(?:at|of)\s+(\w+)\s+(\d{1,2}),?\s*(\d{4})", re.I), "instant"),
    # 2025Q2 / 2025:Q2 / 2025 Q2  (calendar quarter -- the IMF convention)
    (re.compile(r"\b(\d{4})\s*[:\-]?\s*Q([1-4])\b"), "calendar_quarter"),
    # Q2 2025 (calendar quarter, written the other way round)
    (re.compile(r"\bQ([1-4])\s+(\d{4})\b"), "calendar_quarter_reversed"),
    # 2024-25 / 2024/25  (RBI and Economic Survey style: an Indian fiscal year)
    (re.compile(r"\b(\d{4})\s*[-/]\s*(\d{2})\b"), "bare_fiscal_span"),
    # March 31, 2024
    (re.compile(r"\b(\w+)\s+(\d{1,2}),?\s+(\d{4})\b"), "instant"),
    # bare 2025
    (re.compile(r"\b(\d{4})\b"), "calendar_year"),
]


# Kinds whose validity check is authoritative: if they match and then fail, the
# text is not a period at all and no looser pattern may reinterpret it.
_HARD_REJECT_KINDS = frozenset({"fiscal_span", "bare_fiscal_span"})


def parse_period(text: str | None) -> Period | None:
    """Parse the first period label found in ``text``.

    Returns ``None`` when nothing period-shaped is present, which the caller
    should treat as an extraction failure rather than a default.
    """
    if not text:
        return None
    raw = text.strip()

    for pattern, kind in _PATTERNS:
        m = pattern.search(raw)
        if not m:
            continue
        period = _build(kind, m, raw)
        if period is not None:
            return period
        if kind in _HARD_REJECT_KINDS:
            # The text matched a specific convention and failed its validity
            # check. Falling through would let a looser pattern reinterpret the
            # same characters -- "2023/04" would come back as calendar year
            # 2023 having just been rejected as a fiscal span.
            return None
    return None


def _build(kind: str, m: re.Match[str], raw: str) -> Period | None:
    if kind == "fiscal_quarter":
        quarter, year = int(m.group(1)), int(m.group(2))
        start, end = fiscal_quarter(year, quarter)
        return Period(start, end, KIND_FISCAL_QUARTER, raw)

    if kind == "fiscal_half":
        half, year = int(m.group(1)), int(m.group(2))
        start, end = fiscal_half(year, half)
        return Period(start, end, KIND_FISCAL_HALF, raw)

    if kind == "fiscal_span":
        # FY2023-24 and FY2025/26 are both named by the year they end in.
        end_year = _span_end_year(int(m.group(1)), m.group(2))
        if end_year is None:
            return None
        start, end = fiscal_year(end_year)
        return Period(start, end, KIND_FISCAL_YEAR, raw)

    if kind == "fiscal_span_short":
        end_year = _four_digit(int(m.group(2)))
        start, end = fiscal_year(end_year)
        return Period(start, end, KIND_FISCAL_YEAR, raw)

    if kind == "fiscal_year":
        start, end = fiscal_year(int(m.group(1)))
        return Period(start, end, KIND_FISCAL_YEAR, raw)

    if kind == "year_ended":
        month = MONTHS.get(m.group(1).lower())
        if month is None:
            return None
        year = int(m.group(3))
        if month == FISCAL_YEAR_END_MONTH:
            start, end = fiscal_year(year)
            return Period(start, end, KIND_FISCAL_YEAR, raw)
        start = date(year - 1, month, 1)
        return Period(start, _end_of_month(year, month), "interval", raw, inferred=True)

    if kind == "calendar_quarter":
        year, quarter = int(m.group(1)), int(m.group(2))
        start, end = calendar_quarter(year, quarter)
        return Period(start, end, KIND_CALENDAR_QUARTER, raw)

    if kind == "calendar_quarter_reversed":
        quarter, year = int(m.group(1)), int(m.group(2))
        start, end = calendar_quarter(year, quarter)
        return Period(start, end, KIND_CALENDAR_QUARTER, raw)

    if kind == "bare_fiscal_span":
        # "2024-25" in an Indian institutional report means the fiscal year
        # ending March 2025. Marked inferred: the convention is read from
        # context, not from the string.
        end_year = _span_end_year(int(m.group(1)), m.group(2))
        if end_year is None:
            return None
        start, end = fiscal_year(end_year)
        return Period(start, end, KIND_FISCAL_YEAR, raw, inferred=True)

    if kind == "instant":
        month = MONTHS.get(m.group(1).lower())
        if month is None:
            return None
        day, year = int(m.group(2)), int(m.group(3))
        try:
            point = date(year, month, day)
        except ValueError:
            return None
        return Period(point, point, KIND_INSTANT, raw)

    if kind == "calendar_year":
        year = int(m.group(1))
        return Period(
            date(year, 1, 1), date(year, 12, 31), KIND_CALENDAR_YEAR, raw, inferred=True
        )

    return None


def _span_end_year(first: int, second_raw: str) -> int | None:
    """Resolve the closing year of a span like 2023-24, 2025/26 or 2024-2025.

    A fiscal span always covers two *consecutive* years, and enforcing that is
    what rejects things that merely look like spans. The corpus contains 21
    occurrences of ``2023/04`` -- a URL path segment, ``uploads/2023/04/`` --
    which without this check parsed as a fiscal year ending in 2104.

    Returns ``None`` when the two years are not consecutive.
    """
    second = int(second_raw)

    if len(second_raw) == 4:
        end_year = second
    else:
        century = (first // 100) * 100
        end_year = century + second
        if end_year < first:  # 1999-00 rolls over into the next century
            end_year += 100

    return end_year if end_year == first + 1 else None
