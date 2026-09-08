"""Group layout blocks into the regions handed to the extractor.

PyMuPDF splits a financial table into several blocks: the unit caption
``(Rs in Million)`` is one, the ``Standalone / Consolidated`` headers another,
the period headers a third, and the data rows a fourth. Read alone, a data row
says ``Revenue from Operations 74,540.82 ... 81,415.38`` with no unit, no basis
and no period -- every qualifier the comparison engine depends on is missing.

So blocks are merged into regions before extraction. Merging is driven by
reading-order adjacency rather than horizontal overlap, because a right-aligned
unit caption shares no x-range with the left-aligned row labels beneath it.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

from app.ingest.parse import BBox, Block, ParsedPage

# Vertical gap between blocks, as a multiple of line height, that still reads as
# continuous content.
GAP_TOLERANCE = 2.0

# Blocks may sit slightly above the previous one (side-by-side header cells)
# without being treated as a jump back up the page.
BACKTRACK_TOLERANCE = 1.2

# A region wider than this fraction of the page is spanning columns that are not
# actually related -- these pages are two-page spreads.
MAX_WIDTH_RATIO = 0.60

# Extraction output scales with how many values a region contains, and reasoning
# models spend their whole token budget before answering if a region is too
# dense. 2500 keeps a financial table together with its caption and headers --
# the invariant that matters -- without also absorbing the narrative that
# follows it.
MAX_REGION_CHARS = 2500

KIND_TABLE = "table"
KIND_HEADING = "heading"
KIND_PARAGRAPH = "paragraph"

# 1,234.56 | (940.69) | 81,415.38 -- at least two digits so years and list
# markers do not count as tabular data.
_NUMERIC = re.compile(r"\(?\d[\d,]*\.?\d*\)?")
_MIN_NUMERIC_TOKENS = 4


@dataclass
class Region:
    """A contiguous run of blocks handed to the extractor as one unit."""

    page_no: int
    kind: str
    text: str
    bbox: BBox
    char_start: int
    char_end: int
    block_indices: list[int]


def _numeric_token_count(text: str) -> int:
    return sum(1 for m in _NUMERIC.finditer(text) if len(re.sub(r"\D", "", m.group(0))) >= 2)


def _line_height(page: ParsedPage) -> float:
    heights = [w.bbox[3] - w.bbox[1] for w in page.words if w.bbox[3] > w.bbox[1]]
    return statistics.median(heights) if heights else 11.0


def _classify(text: str) -> str:
    stripped = text.strip()
    if _numeric_token_count(stripped) >= _MIN_NUMERIC_TOKENS:
        return KIND_TABLE
    if len(stripped) < 80 and "\n" not in stripped and not re.search(r"\d", stripped):
        return KIND_HEADING
    return KIND_PARAGRAPH


def _union(boxes: list[BBox]) -> BBox:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _continues(prev: Block, current: Block, line_height: float, page_width: float,
               current_boxes: list[BBox]) -> bool:
    """Does ``current`` continue the region that ``prev`` ends?"""
    # Moving back up the page means a new column or a new section. A small
    # backward step is tolerated so side-by-side header cells stay together.
    if current.bbox[1] < prev.bbox[1] - line_height * BACKTRACK_TOLERANCE:
        return False

    # Negative gaps happen when blocks overlap vertically; those belong together.
    gap = current.bbox[1] - prev.bbox[3]
    if gap > line_height * GAP_TOLERANCE:
        return False

    merged = _union([*current_boxes, current.bbox])
    return (merged[2] - merged[0]) <= page_width * MAX_WIDTH_RATIO


def segment_page(page: ParsedPage) -> list[Region]:
    """Merge a page's blocks into extraction regions, in reading order."""
    if not page.blocks:
        return []

    line_height = _line_height(page)
    regions: list[Region] = []

    current: list[Block] = []
    boxes: list[BBox] = []

    def flush() -> None:
        if not current:
            return
        text = "\n".join(b.text for b in current)
        regions.append(
            Region(
                page_no=page.page_no,
                kind=_classify(text),
                text=text,
                bbox=_union(boxes),
                char_start=current[0].char_start,
                char_end=current[-1].char_end,
                block_indices=[b.index for b in current],
            )
        )

    for block in page.blocks:
        if not current:
            current, boxes = [block], [block.bbox]
            continue

        size = sum(len(b.text) for b in current)
        if size < MAX_REGION_CHARS and _continues(
            current[-1], block, line_height, page.width, boxes
        ):
            current.append(block)
            boxes.append(block.bbox)
        else:
            flush()
            current, boxes = [block], [block.bbox]

    flush()
    return regions


def segment_document(pages: list[ParsedPage]) -> list[Region]:
    return [region for page in pages for region in segment_page(page)]
