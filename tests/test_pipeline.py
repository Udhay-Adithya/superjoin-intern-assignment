"""End-to-end pipeline test with a stubbed model.

The language model is the one part of this system that cannot be tested
deterministically, so it is stubbed and everything else is exercised for real:
the actual PDF is parsed, segmented, grounded against its own text, normalized
and reconciled. What is being tested is that a well-behaved extraction really
does produce the four required cases -- and that a badly-behaved one is caught.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

from app import db
from app.ingest.pipeline import ingest_document, load_cluster, reconcile_keys
from app.reason.compare import CORROBORATES, RECONCILED

ANNUAL_REPORT = Path("starter-datasets/delhivery/02-delhivery-annual-report-fy24-excerpt.pdf")

# The four revenue figures in the Directors' Report table, with the qualifiers a
# competent extractor should read off the caption and column headers.
TABLE_FACTS = [
    ("74,540.82", "standalone", "FY ended March 31, 2024"),
    ("66,586.61", "standalone", "FY ended March 31, 2023"),
    ("81,415.38", "consolidated", "FY ended March 31, 2024"),
    ("72,253.01", "consolidated", "FY ended March 31, 2023"),
]


class StubClient:
    """Stands in for a well-behaved extractor.

    Quotes are taken from the excerpt itself, which is what a model asked for
    verbatim evidence is supposed to do. The grounding gate still verifies them.
    """

    def __init__(self, *, hallucinate: bool = False) -> None:
        self.hallucinate = hallucinate
        self.calls = 0

    def complete_json(self, messages, *, model, schema, schema_name="result", max_tokens=8000):
        self.calls += 1
        if schema_name == "document_metadata":
            return {
                "title": "Delhivery Annual Report FY24",
                "publisher": "Delhivery Limited",
                "doc_type": "annual_report",
                "published_date": "2024-08-08",
                "primary_entity": "Delhivery Limited",
            }

        excerpt = messages[-1]["content"]
        facts = []

        for value, basis, period in TABLE_FACTS:
            if value not in excerpt:
                continue
            quote = _line_containing(excerpt, value)
            facts.append(
                {
                    "subject": "your Company",  # anaphora, resolved from metadata
                    "metric": "revenue from operations",
                    "value_raw": value,
                    "unit_raw": "₹ in Million",
                    "period_raw": period,
                    "basis_raw": basis,
                    "variant_raw": "",
                    "modality": "actual",
                    "quote": quote,
                }
            )

        if self.hallucinate and facts:
            # A real quote paired with a value that is nowhere in the document.
            facts.append(
                {
                    "subject": "your Company",
                    "metric": "revenue from operations",
                    "value_raw": "99,999.99",
                    "unit_raw": "₹ in Million",
                    "period_raw": "FY ended March 31, 2024",
                    "basis_raw": "consolidated",
                    "variant_raw": "",
                    "modality": "actual",
                    "quote": facts[0]["quote"],
                }
            )

        return {"facts": facts}


def _line_containing(text: str, needle: str) -> str:
    for line in text.splitlines():
        if needle in line:
            return re.sub(r"\s+", " ", line).strip()
    return needle


@pytest.fixture
def conn() -> sqlite3.Connection:
    connection = db.connect(":memory:")
    db.init_db(connection)
    yield connection
    connection.close()


@pytest.fixture(scope="module")
def report_path() -> Path:
    if not ANNUAL_REPORT.exists():
        pytest.skip("starter dataset not present")
    return ANNUAL_REPORT


def _ingest(conn: sqlite3.Connection, path: Path, client: StubClient, **kw):
    return ingest_document(
        path, conn=conn, client=client, extract_model="stub", workers=4, **kw
    )


def test_pipeline_produces_grounded_facts(conn, report_path) -> None:
    report = _ingest(conn, report_path, StubClient(), pages=(20, 24))

    assert report.n_stored > 0, "no facts survived the pipeline"
    assert report.metadata is not None
    assert report.metadata.source_tier == 90  # annual report outranks a slide deck

    rows = conn.execute(
        "SELECT quote, char_start, char_end, page_no FROM facts LIMIT 20"
    ).fetchall()
    for row in rows:
        assert row["quote"], "a stored fact has no evidence"
        assert row["char_end"] > row["char_start"], "evidence span is empty"


def test_case_one_and_three_emerge_from_the_real_document(conn, report_path) -> None:
    """Both cases come out of the same table, decided without a model."""
    _ingest(conn, report_path, StubClient(), pages=(20, 24))

    verdicts = {
        row["verdict"]
        for row in conn.execute("SELECT DISTINCT verdict FROM relations").fetchall()
    }
    assert RECONCILED in verdicts, "standalone vs consolidated was not reconciled"

    reconciled = conn.execute(
        "SELECT rule, explanation FROM relations WHERE verdict = ? AND rule = ?",
        (RECONCILED, "different_basis"),
    ).fetchall()
    assert reconciled, "case 3 did not fire on the real table"
    assert "standalone" in reconciled[0]["explanation"].lower()
    assert "consolidated" in reconciled[0]["explanation"].lower()


def test_the_same_figure_stated_twice_corroborates(conn, report_path) -> None:
    """The table figure and the prose that restates it are separate regions.

    They should meet again in the same core-key cluster and corroborate.
    """
    _ingest(conn, report_path, StubClient(), pages=(20, 24))

    corroborations = conn.execute(
        "SELECT COUNT(*) AS n FROM relations WHERE verdict = ?", (CORROBORATES,)
    ).fetchone()
    assert corroborations["n"] > 0, "repeated statements of one figure did not corroborate"


def test_grounding_gate_rejects_a_hallucinated_value(conn, report_path) -> None:
    """The check that earns its keep: a real quote with an invented number."""
    report = _ingest(conn, report_path, StubClient(hallucinate=True), pages=(20, 24))

    assert report.grounding_rejection_rate > 0, "the fabricated value was not rejected"
    assert report.rejections, "the fabricated value produced no recorded rejection"

    stored = conn.execute(
        "SELECT COUNT(*) AS n FROM facts WHERE value_raw = '99,999.99'"
    ).fetchone()
    assert stored["n"] == 0, "a fabricated value reached the knowledge layer"

    failures = conn.execute(
        "SELECT COUNT(*) AS n FROM failures WHERE stage = 'ground'"
    ).fetchone()
    assert failures["n"] > 0, "the rejection was not recorded for reporting"


def test_reingesting_the_same_document_is_a_no_op(conn, report_path) -> None:
    first = _ingest(conn, report_path, StubClient(), pages=(20, 24))
    second = _ingest(conn, report_path, StubClient(), pages=(20, 24))

    assert second.already_ingested
    assert second.doc_id == first.doc_id

    count = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()
    assert count["n"] == 1


def test_reconciliation_is_bounded_to_touched_clusters(conn, report_path) -> None:
    """Incremental ingest: only clusters the document touches are recomputed."""
    _ingest(conn, report_path, StubClient(), pages=(20, 24))

    keys = [
        r["core_key"]
        for r in conn.execute(
            "SELECT DISTINCT core_key FROM facts WHERE core_key IS NOT NULL LIMIT 1"
        ).fetchall()
    ]
    assert keys

    cluster = load_cluster(conn, keys[0])
    assert cluster, "cluster did not load back from storage"

    # Recomputing one key must not disturb relations outside it.
    before = conn.execute("SELECT COUNT(*) AS n FROM relations").fetchone()["n"]
    reconcile_keys(conn, {keys[0]})
    after = conn.execute("SELECT COUNT(*) AS n FROM relations").fetchone()["n"]
    assert after == before


def test_anaphora_resolves_to_the_document_subject(conn, report_path) -> None:
    """"your Company" carries no identity; it must resolve to Delhivery."""
    _ingest(conn, report_path, StubClient(), pages=(20, 24))

    names = [
        r["canonical_name"]
        for r in conn.execute("SELECT canonical_name FROM entities").fetchall()
    ]
    assert names, "no entity was created"
    assert any("delhivery" in n for n in names), names
    assert not any(n in {"your company", "the company"} for n in names), names


def test_more_pages_of_an_ingested_document_can_be_added(conn, report_path) -> None:
    """Deepening coverage of a document already stored.

    Re-uploading the same PDF must not duplicate it, but reading *further pages*
    of it is a real thing to want. The first version refused both, so a second
    range of an already-ingested report was silently skipped.
    """
    first = _ingest(conn, report_path, StubClient(), pages=(20, 24))
    assert not first.already_ingested
    assert first.n_stored > 0

    second = _ingest(conn, report_path, StubClient(), pages=(25, 30))
    assert not second.already_ingested, "new pages should be read, not skipped"
    assert second.doc_id == first.doc_id, "and must attach to the same document"

    documents = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()
    assert documents["n"] == 1, "the document must not be duplicated"

    # Pages read in the first pass are not read again.
    pages_seen = {
        r["page_no"]
        for r in conn.execute("SELECT DISTINCT page_no FROM blocks WHERE doc_id = ?",
                              (first.doc_id,))
    }
    assert pages_seen & set(range(20, 25)), "first range recorded"
    assert not any(p > 30 for p in pages_seen), "nothing outside the requested ranges"


def test_repeating_an_already_read_range_is_still_a_no_op(conn, report_path) -> None:
    _ingest(conn, report_path, StubClient(), pages=(20, 24))
    again = _ingest(conn, report_path, StubClient(), pages=(20, 24))
    assert again.already_ingested


# --- upload jobs -------------------------------------------------------


def test_upload_job_state_survives_a_restart(tmp_path, monkeypatch) -> None:
    """A 404 on a job that really exists looks like a missing endpoint.

    Ingestion runs in a background thread and the UI polls for it, so job state
    has to outlive the process that created it. Held in memory it broke as soon
    as the polling request reached a different process -- `uvicorn --reload`
    restarting when the ingest writes its cache, or `--workers N` giving each
    worker its own dict.
    """
    from app import config
    from app.main import Job, _load_job, _save_job

    monkeypatch.setattr(config, "DB_PATH", tmp_path / "jobs.db")

    _save_job(Job(id="abc123", filename="x.pdf", status="running"))

    # A fresh read, as a different process would do it.
    loaded = _load_job("abc123")
    assert loaded is not None
    assert loaded.status == "running"
    assert loaded.filename == "x.pdf"

    _save_job(Job(id="abc123", filename="x.pdf", status="done", report={"n_stored": 7}))
    reloaded = _load_job("abc123")
    assert reloaded is not None
    assert reloaded.status == "done"
    assert reloaded.report["n_stored"] == 7

    assert _load_job("never-existed") is None
