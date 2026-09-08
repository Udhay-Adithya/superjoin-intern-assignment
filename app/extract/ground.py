"""The grounding gate: prove every fact is really in the document.

A model asked for a verbatim quote will usually give one, but not always, and
"usually" is not a property you can build a knowledge layer on. This module
verifies each claimed quote against the source text, recovers its exact
character offsets, and rejects anything it cannot find.

Matching is whitespace- and punctuation-tolerant by design. The layout-preserving
text contains long runs of spaces holding table columns apart, and models
reliably collapse those when quoting. Rejecting a fact over invisible whitespace
would measure the model's formatting habits rather than its honesty. Everything
that matters -- the digits, the words, the order -- must still match exactly.

What this catches is worth more than what it costs: a value that never appears
in the source, a quote assembled from two different rows, a number quietly
"corrected" during extraction.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Characters that differ between a PDF and a model's rendering of it without
# changing meaning. Mapped one-to-one so offsets stay aligned.
_CHAR_FOLD = {
    "–": "-", "—": "-", "‒": "-", "−": "-",  # dashes
    "‘": "'", "’": "'", "‛": "'",                  # single quotes
    "“": '"', "”": '"',                                 # double quotes
    " ": " ", " ": " ", " ": " ", " ": " ",   # spaces
}

_WS = re.compile(r"\s+")


class GroundingError(Exception):
    """A candidate fact could not be tied to the source text."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class Grounded:
    """A quote located in the source, with offsets that resolve."""

    quote: str  # the source text as written, not as the model rendered it
    char_start: int  # absolute offset into the document stream
    char_end: int
    repaired: bool = False  # evidence was recovered, not quoted correctly


def _fold(text: str) -> str:
    """Fold characters that vary harmlessly, preserving length."""
    normalized = unicodedata.normalize("NFC", text)
    return "".join(_CHAR_FOLD.get(ch, ch) for ch in normalized)


def _collapse(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace runs, returning the result and an offset map.

    ``offsets[i]`` is the index in ``text`` that produced ``collapsed[i]``, which
    is what lets a match in collapsed space be reported in source coordinates.
    """
    collapsed: list[str] = []
    offsets: list[int] = []

    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char.isspace():
            run = _WS.match(text, index)
            end = run.end() if run else index + 1
            # A whitespace run is meaningful as a separator but not as content.
            if collapsed and not collapsed[-1].isspace():
                collapsed.append(" ")
                offsets.append(index)
            index = end
        else:
            collapsed.append(char)
            offsets.append(index)
            index += 1

    return "".join(collapsed), offsets


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def locate(quote: str, source: str, *, source_offset: int = 0) -> Grounded:
    """Find ``quote`` in ``source`` and return its offsets.

    ``source_offset`` shifts the result into document-wide coordinates.
    Raises :class:`GroundingError` when the quote is not present.
    """
    if not quote or not quote.strip():
        raise GroundingError("empty quote")

    folded_source = _fold(source)
    folded_quote = _fold(quote)

    # Exact match first: cheapest, and the common case.
    position = folded_source.find(folded_quote)
    if position != -1:
        return Grounded(
            quote=source[position : position + len(folded_quote)],
            char_start=source_offset + position,
            char_end=source_offset + position + len(folded_quote),
        )

    # Fall back to whitespace-insensitive matching.
    collapsed_source, offsets = _collapse(folded_source)
    collapsed_quote = _WS.sub(" ", folded_quote).strip()
    if not collapsed_quote:
        raise GroundingError("empty quote")

    position = collapsed_source.find(collapsed_quote)
    if position == -1:
        raise GroundingError("quote not found in source")

    start = offsets[position]
    end = offsets[position + len(collapsed_quote) - 1] + 1
    return Grounded(
        quote=source[start:end],
        char_start=source_offset + start,
        char_end=source_offset + end,
    )


def _repair_from_value(value_raw: str, source: str, source_offset: int) -> Grounded:
    """Recover evidence for a real value whose quote was badly formed.

    Models routinely build a quote for a table cell by pairing the row label
    with just that one value -- "Revenue from Operations 81,415.38" -- when the
    row actually holds four figures in sequence. The value is genuinely in the
    document; only the quote is synthetic.

    Rather than discard a real fact, the containing line is used as the
    evidence. The guarantee is unchanged: the evidence is still verbatim source
    text that demonstrably contains the value. The repair is refused when the
    value appears on more than one line, because then the evidence is ambiguous
    and a guess would be worse than a rejection.
    """
    needle = _digits(value_raw)
    if not needle:
        raise GroundingError("quote not found in source")

    matches = []
    cursor = 0
    for line in source.split("\n"):
        if needle in _digits(line):
            matches.append((cursor, cursor + len(line), line))
        cursor += len(line) + 1

    if not matches:
        raise GroundingError("quote not found in source")
    if len(matches) > 1:
        raise GroundingError("evidence ambiguous: value appears on several lines")

    start, end, line = matches[0]
    return Grounded(
        quote=line,
        char_start=source_offset + start,
        char_end=source_offset + end,
        repaired=True,
    )


def ground_fact(
    *, quote: str, value_raw: str, source: str, source_offset: int = 0
) -> Grounded:
    """Verify a quote is in the source and that it actually contains the value.

    The second check is the one that matters. A model can return a real quote
    from the document alongside a value that appears nowhere in it -- pairing a
    genuine sentence with a hallucinated or silently converted number. Comparing
    digits ignores thousands separators and currency symbols while still
    requiring the figure itself to be present.
    """
    # An absent quote is a different failure from a wrong one: the model
    # ignored the contract rather than misapplied it, so there is nothing to
    # repair from.
    if not quote or not quote.strip():
        raise GroundingError("empty quote")

    value_digits = _digits(value_raw)

    try:
        located = locate(quote, source, source_offset=source_offset)
    except GroundingError:
        if not value_digits:
            raise
        # The quote is wrong, but the value may still be real. Try to recover
        # its evidence from the source rather than lose a genuine fact.
        return _repair_from_value(value_raw, source, source_offset)

    if not value_digits:
        # Non-numeric facts (a director's status, say) have nothing to check.
        return located

    if value_digits not in _digits(located.quote):
        # A real quote paired with a value that is not in it. The quote may
        # simply be the wrong row; fall back to locating the value itself.
        return _repair_from_value(value_raw, source, source_offset)

    return located
