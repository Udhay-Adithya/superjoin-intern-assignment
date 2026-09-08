"""Read a PDF into layout-preserving text with per-word coordinates.

Why reconstruct layout rather than take PyMuPDF's reading-order text: the
financial tables in this corpus are borderless, so ``page.find_tables()`` finds
only the ruled header and drops the body, and ``get_text("text")`` flattens a
table into a column of loose values where the mapping from header to number is
positional and implicit.

Rebuilding the visual layout from word coordinates keeps the table looking like a
table, which is the form a language model reads most reliably. It also yields a
character-offset to bounding-box map for free, which is what lets the UI
highlight the exact evidence span on the rendered page.
"""

from __future__ import annotations

import hashlib
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

# A word belongs to the current line while its vertical centre stays within this
# fraction of the median glyph height.
LINE_TOLERANCE = 0.6

# Guardrails for the estimated character width used to place words into columns.
MIN_CHAR_WIDTH = 3.0
MAX_CHAR_WIDTH = 12.0
DEFAULT_CHAR_WIDTH = 5.5

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class Word:
    """One word, with its position both on the page and in the text stream."""

    text: str
    bbox: BBox
    char_start: int
    char_end: int


@dataclass
class Block:
    """A layout block as PyMuPDF's page analysis sees it.

    Blocks are the unit that respects column boundaries. The pages in this
    corpus are two-page spreads with four text columns, so rendering a whole
    page as one grid interleaves unrelated columns into the same line.
    """

    index: int
    bbox: BBox
    text: str
    char_start: int
    char_end: int


@dataclass
class ParsedPage:
    page_no: int  # 1-indexed, matching what a reader sees
    text: str
    words: list[Word] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0
    char_start: int = 0  # offset into the document-wide stream
    char_end: int = 0


@dataclass
class ParsedDoc:
    path: Path
    sha256: str
    n_pages: int
    pages: list[ParsedPage]

    @property
    def text(self) -> str:
        return "\n".join(p.text for p in self.pages)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _estimate_char_width(raw_words: list[tuple]) -> float:
    """Estimate the width of one character, for placing words into columns."""
    widths = [
        (w[2] - w[0]) / len(w[4])
        for w in raw_words
        if len(w[4]) >= 2 and (w[2] - w[0]) > 0
    ]
    if not widths:
        return DEFAULT_CHAR_WIDTH
    return min(max(statistics.median(widths), MIN_CHAR_WIDTH), MAX_CHAR_WIDTH)


def _group_into_lines(raw_words: list[tuple]) -> list[list[tuple]]:
    """Group words into visual lines by vertical position."""
    if not raw_words:
        return []

    heights = [w[3] - w[1] for w in raw_words if (w[3] - w[1]) > 0]
    tolerance = (statistics.median(heights) if heights else 10.0) * LINE_TOLERANCE

    ordered = sorted(raw_words, key=lambda w: ((w[1] + w[3]) / 2, w[0]))

    lines: list[list[tuple]] = []
    current: list[tuple] = []
    current_centre = None

    for word in ordered:
        centre = (word[1] + word[3]) / 2
        if current_centre is None or abs(centre - current_centre) <= tolerance:
            current.append(word)
            # Track a running centre so gently sloping lines still group.
            current_centre = centre if current_centre is None else current_centre
        else:
            lines.append(sorted(current, key=lambda w: w[0]))
            current = [word]
            current_centre = centre

    if current:
        lines.append(sorted(current, key=lambda w: w[0]))
    return lines


def _layout_words(
    raw_words: list[tuple], char_width: float, offset: int
) -> tuple[str, list[Word]]:
    """Render a set of words as aligned text starting at ``offset``.

    Column positions are relative to the leftmost word in this set, so a block
    is aligned against its own margin rather than the page's.
    """
    if not raw_words:
        return "", []

    left_margin = min(w[0] for w in raw_words)
    lines = _group_into_lines(raw_words)

    out: list[str] = []
    words: list[Word] = []
    cursor = offset

    for line in lines:
        buffer = ""
        for w in line:
            x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
            column = int(round((x0 - left_margin) / char_width))
            if column > len(buffer):
                buffer += " " * (column - len(buffer))
            elif buffer:
                buffer += " "  # never let two words run together

            start = cursor + len(buffer)
            buffer += text
            words.append(
                Word(text=text, bbox=(x0, y0, x1, y1), char_start=start, char_end=start + len(text))
            )
        out.append(buffer)
        cursor += len(buffer) + 1  # +1 for the newline joining lines

    return "\n".join(out), words


def layout_page(page: pymupdf.Page) -> tuple[str, list[Word], list[Block]]:
    """Rebuild a page block by block, preserving column structure.

    Returns the page text, the word offset/bbox map, and the block spans.
    """
    raw_words = page.get_text("words")
    if not raw_words:
        return "", [], []

    char_width = _estimate_char_width(raw_words)

    # Words carry the block number PyMuPDF assigned them (index 5), so grouping
    # by it reuses PyMuPDF's own column-aware reading order.
    by_block: dict[int, list[tuple]] = {}
    for w in raw_words:
        by_block.setdefault(int(w[5]), []).append(w)

    chunks: list[str] = []
    words: list[Word] = []
    blocks: list[Block] = []
    cursor = 0

    for block_no in sorted(by_block):
        members = by_block[block_no]
        text, block_words = _layout_words(members, char_width, cursor)
        if not text.strip():
            continue

        blocks.append(
            Block(
                index=block_no,
                bbox=(
                    min(w[0] for w in members),
                    min(w[1] for w in members),
                    max(w[2] for w in members),
                    max(w[3] for w in members),
                ),
                text=text,
                char_start=cursor,
                char_end=cursor + len(text),
            )
        )
        chunks.append(text)
        words.extend(block_words)
        cursor += len(text) + 1  # +1 for the newline joining blocks

    return "\n".join(chunks), words, blocks


def parse_pdf(path: str | Path) -> ParsedDoc:
    """Parse a PDF into pages of layout-preserving text."""
    path = Path(path)
    doc = pymupdf.open(path)

    pages: list[ParsedPage] = []
    offset = 0
    try:
        for index, page in enumerate(doc):
            text, words, blocks = layout_page(page)
            # Shift page-local offsets into the document-wide stream.
            shifted_words = [
                Word(w.text, w.bbox, w.char_start + offset, w.char_end + offset) for w in words
            ]
            shifted_blocks = [
                Block(b.index, b.bbox, b.text, b.char_start + offset, b.char_end + offset)
                for b in blocks
            ]
            rect = page.rect
            pages.append(
                ParsedPage(
                    page_no=index + 1,
                    text=text,
                    words=shifted_words,
                    blocks=shifted_blocks,
                    width=rect.width,
                    height=rect.height,
                    char_start=offset,
                    char_end=offset + len(text),
                )
            )
            offset += len(text) + 1  # +1 for the newline joining pages
        n_pages = doc.page_count
    finally:
        doc.close()

    return ParsedDoc(path=path, sha256=sha256_of(path), n_pages=n_pages, pages=pages)


def bboxes_for_span(page: ParsedPage, char_start: int, char_end: int) -> list[BBox]:
    """Bounding boxes covering a character span, merged per visual line.

    This is what turns a verified evidence quote into a highlight on the page.
    """
    hits = [w for w in page.words if w.char_start < char_end and w.char_end > char_start]
    if not hits:
        return []

    by_line: dict[int, list[Word]] = {}
    for w in hits:
        # Round the vertical centre so words on one line share a bucket.
        key = int((w.bbox[1] + w.bbox[3]) / 2)
        by_line.setdefault(key, []).append(w)

    merged: list[BBox] = []
    for line_words in by_line.values():
        merged.append(
            (
                min(w.bbox[0] for w in line_words),
                min(w.bbox[1] for w in line_words),
                max(w.bbox[2] for w in line_words),
                max(w.bbox[3] for w in line_words),
            )
        )
    return sorted(merged, key=lambda b: (b[1], b[0]))
