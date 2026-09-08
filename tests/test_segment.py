"""Segmentation invariants, checked against the real annual report."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingest.parse import ParsedDoc, parse_pdf
from app.ingest.segment import KIND_TABLE, segment_page

ANNUAL_REPORT = Path("starter-datasets/delhivery/02-delhivery-annual-report-fy24-excerpt.pdf")
REVENUE_PAGE_INDEX = 21  # 0-indexed; the Directors' Report financial summary


@pytest.fixture(scope="module")
def annual_report() -> ParsedDoc:
    if not ANNUAL_REPORT.exists():
        pytest.skip("starter dataset not present")
    return parse_pdf(ANNUAL_REPORT)


def test_layout_preserves_table_columns(annual_report: ParsedDoc) -> None:
    """All four figures of the revenue row must survive on one line.

    Reading-order extraction flattens them into a vertical list, which destroys
    the mapping from column header to value.
    """
    page = annual_report.pages[REVENUE_PAGE_INDEX]
    row = next(ln for ln in page.text.splitlines() if "Revenue from Operations" in ln)
    for figure in ("74,540.82", "66,586.61", "81,415.38", "72,253.01"):
        assert figure in row, f"{figure} left the revenue row"


def test_every_region_carries_the_qualifiers_for_its_figures(
    annual_report: ParsedDoc,
) -> None:
    """The invariant the whole comparison engine rests on.

    A value is meaningless without the unit, basis and period that qualify it.
    The consolidated revenue figure appears twice on this page -- once in the
    table and once in the prose that restates it -- and *each* occurrence must
    be extractable on its own, because the extractor sees one region at a time.
    """
    page = annual_report.pages[REVENUE_PAGE_INDEX]
    regions = [r for r in segment_page(page) if "81,415.38" in r.text]
    assert regions, "the figure disappeared from the page entirely"

    for region in regions:
        text = region.text.lower()
        assert "consolidated" in text, "basis qualifier separated from the value"
        assert "million" in text, "unit separated from the value"
        assert "fy24" in text or "march 31, 2024" in text, "period separated from the value"


def test_the_table_region_holds_both_bases_together(annual_report: ParsedDoc) -> None:
    """Case 3 needs both figures visible together to be explained by basis."""
    page = annual_report.pages[REVENUE_PAGE_INDEX]
    table = next(
        r
        for r in segment_page(page)
        if r.kind == KIND_TABLE and "81,415.38" in r.text and "Particulars" in r.text
    )
    assert "74,540.82" in table.text
    assert "Standalone" in table.text and "Consolidated" in table.text
    assert "₹ in Million" in table.text


def test_regions_stay_within_the_prompt_budget(annual_report: ParsedDoc) -> None:
    for page in annual_report.pages[:30]:
        for region in segment_page(page):
            assert len(region.text) <= 8000, "region too large to prompt with reliably"


def test_offsets_address_the_document_stream(annual_report: ParsedDoc) -> None:
    """Evidence offsets must resolve, or the grounding gate cannot verify a quote."""
    page = annual_report.pages[REVENUE_PAGE_INDEX]
    region = next(r for r in segment_page(page) if "81,415.38" in r.text)
    assert annual_report.text[region.char_start : region.char_end] == region.text
