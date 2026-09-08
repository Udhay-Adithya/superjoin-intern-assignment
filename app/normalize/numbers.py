"""Turn numbers as written into comparable canonical magnitudes.

This is what makes Case 1 work: the annual report says "Rs 81,415.38 million" and
the earnings deck says "8,142" under a "Rs Cr" header. Those are the same quantity
and nothing downstream can see that until this module has run.

Indian scales (lakh, crore) sit alongside western ones (million, billion) in the
same corpus, and sometimes in the same document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Ordered longest-first so "crore" is matched before "cr", "million" before "mn".
SCALES: list[tuple[str, float]] = [
    ("trillion", 1e12),
    ("billion", 1e9),
    ("million", 1e6),
    ("thousand", 1e3),
    ("crore", 1e7),
    ("lakhs", 1e5),
    ("lakh", 1e5),
    ("lacs", 1e5),
    ("lac", 1e5),
    ("mln", 1e6),
    ("mn", 1e6),
    ("bn", 1e9),
    ("tn", 1e12),
    ("cr", 1e7),
]

_SCALE_RE = re.compile(
    r"\b(" + "|".join(re.escape(name) for name, _ in SCALES) + r")\b",
    re.IGNORECASE,
)

_PERCENT_RE = re.compile(r"(%|per\s*cent|percent|percentage\s+point|pp\b)", re.IGNORECASE)

_CURRENCY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("INR", re.compile(r"(₹|\bINR\b|\bRs\.?\b|\brupee)", re.IGNORECASE)),
    ("USD", re.compile(r"(\$|\bUSD\b|\bUS\$)", re.IGNORECASE)),
    ("EUR", re.compile(r"(€|\bEUR\b)", re.IGNORECASE)),
]

# Accounting negatives: (940.69) means -940.69
_PARENS_RE = re.compile(r"^\(\s*(.+?)\s*\)$")

# A bare dash in a table cell means "no value", not zero.
_NULL_TOKENS = {"-", "--", "–", "—", "n/a", "na", "nil", ""}

_NUM_RE = re.compile(r"[-+]?\d[\d,\s]*(?:\.\d+)?")


@dataclass(frozen=True)
class Quantity:
    """A number reduced to a comparable form.

    ``value`` is in base units: rupees for INR, percent for PERCENT, bare count
    otherwise. ``scale`` records what was written so the UI can show provenance.
    """

    value: float
    unit: str  # INR | USD | EUR | PERCENT | COUNT
    scale: str | None = None
    currency: str | None = None
    raw: str = ""

    @property
    def unit_class(self) -> str:
        if self.unit == "PERCENT":
            return "percent"
        if self.unit in {"INR", "USD", "EUR"}:
            return "currency"
        return "count"


def parse_number(text: str) -> float | None:
    """Parse a written number, honouring accounting parentheses for negatives."""
    if text is None:
        return None
    s = text.strip()
    if s.lower() in _NULL_TOKENS:
        return None

    negative = False
    m = _PARENS_RE.match(s)
    if m:
        negative = True
        s = m.group(1)

    s = s.replace("−", "-")  # unicode minus
    match = _NUM_RE.search(s)
    if not match:
        return None

    cleaned = re.sub(r"[,\s]", "", match.group(0))
    try:
        value = float(cleaned)
    except ValueError:
        return None

    return -abs(value) if negative else value


def detect_scale(*texts: str | None) -> tuple[float, str | None]:
    """Find a scale word in any of the given strings (value, unit, caption)."""
    for text in texts:
        if not text:
            continue
        m = _SCALE_RE.search(text)
        if m:
            found = m.group(1).lower()
            for name, mult in SCALES:
                if name == found:
                    return mult, name
    return 1.0, None


def detect_currency(*texts: str | None) -> str | None:
    for text in texts:
        if not text:
            continue
        for code, pattern in _CURRENCY_PATTERNS:
            if pattern.search(text):
                return code
    return None


def is_percent(*texts: str | None) -> bool:
    return any(_PERCENT_RE.search(t) for t in texts if t)


def normalize_quantity(value_raw: str, unit_raw: str | None = None) -> Quantity | None:
    """Reduce a raw value plus its unit context to a canonical Quantity.

    ``unit_raw`` carries table-caption context such as "(Rs in Million)" or a
    column header like "Rs Cr", which is why the extractor is told to capture it.
    """
    number = parse_number(value_raw)
    if number is None:
        return None

    raw = f"{value_raw} {unit_raw or ''}".strip()

    # Percentages never take a magnitude scale.
    if is_percent(value_raw, unit_raw):
        return Quantity(value=number, unit="PERCENT", scale=None, currency=None, raw=raw)

    multiplier, scale_name = detect_scale(value_raw, unit_raw)
    currency = detect_currency(value_raw, unit_raw)

    return Quantity(
        value=number * multiplier,
        unit=currency or "COUNT",
        scale=scale_name,
        currency=currency,
        raw=raw,
    )
