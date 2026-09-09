"""HTTP API and UI for the fact knowledge layer."""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app import config, db
from app.ingest.pipeline import ingest_document, load_cluster
from app.llm.client import LLMClient

WEB_DIR = Path(__file__).resolve().parent / "web"

def _abandon_orphaned_jobs() -> None:
    """Fail jobs whose worker died with the previous process.

    Job state outlives the process but the thread doing the work does not, so a
    job left mid-flight by a restart would otherwise sit at "running" forever
    and the UI would poll it indefinitely.
    """
    conn = get_conn()
    try:
        conn.execute(
            "UPDATE jobs SET status = 'failed', "
            "detail = 'interrupted by a server restart; upload again', "
            "updated_at = datetime('now') "
            "WHERE status IN ('queued', 'running')"
        )
        conn.commit()
    finally:
        conn.close()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    _abandon_orphaned_jobs()
    yield


app = FastAPI(title="Fact Knowledge Layer", lifespan=lifespan)


# --- job tracking ----------------------------------------------------------
# Ingestion takes minutes on a rate-limited endpoint, far longer than a request
# should hold a connection open, so an upload returns a job id that the UI
# polls.
#
# Job state lives in the database rather than in a module-level dict. In memory
# it broke as soon as the process that created a job was not the process
# answering the poll: `uvicorn --reload` restarts when the ingest writes its
# cache files, and `--workers N` gives every worker its own dict. Both return a
# 404 for a job that really exists, which looks like a missing endpoint and is
# not.


@dataclass
class Job:
    id: str
    filename: str
    status: str = "queued"  # queued | running | done | failed
    detail: str = ""
    report: dict[str, Any] = field(default_factory=dict)


def _save_job(job: Job) -> None:
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO jobs (id, filename, status, detail, report_json) "
            "VALUES (?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET status=excluded.status, "
            "detail=excluded.detail, report_json=excluded.report_json, "
            "updated_at=datetime('now')",
            (job.id, job.filename, job.status, job.detail, json.dumps(job.report)),
        )
        conn.commit()
    finally:
        conn.close()


def _load_job(job_id: str) -> Job | None:
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    try:
        report = json.loads(row["report_json"])
    except json.JSONDecodeError:
        report = {}
    return Job(
        id=row["id"],
        filename=row["filename"],
        status=row["status"],
        detail=row["detail"],
        report=report,
    )


# --- helpers ---------------------------------------------------------------


def get_conn() -> sqlite3.Connection:
    conn = db.connect()
    db.init_db(conn)
    return conn


def _rows(cursor: sqlite3.Cursor) -> list[dict]:
    return [dict(row) for row in cursor.fetchall()]


def _decode_bboxes(value: str | None) -> list:
    try:
        return json.loads(value or "[]")
    except json.JSONDecodeError:
        return []


FACT_SELECT = """
SELECT f.id, f.value_raw, f.unit_raw, f.period_raw, f.basis, f.modality, f.variant,
       f.value_norm, f.unit, f.scale, f.currency,
       f.period_start, f.period_end, f.period_kind,
       f.quote, f.page_no, f.char_start, f.char_end, f.bbox_json,
       f.core_key, f.full_key, f.status, f.doc_id,
       e.canonical_name AS entity, m.canonical_name AS metric,
       d.title AS doc_title, d.publisher, d.published_date, d.source_tier, d.filename
FROM facts f
LEFT JOIN entities e ON e.id = f.entity_id
LEFT JOIN metrics  m ON m.id = f.metric_id
JOIN documents d ON d.id = f.doc_id
"""


# --- documents -------------------------------------------------------------


@app.get("/api/documents")
def list_documents() -> list[dict]:
    conn = get_conn()
    try:
        return _rows(
            conn.execute(
                """
                SELECT d.*, (SELECT COUNT(*) FROM facts f WHERE f.doc_id = d.id) AS n_facts
                FROM documents d ORDER BY d.ingested_at DESC
                """
            )
        )
    finally:
        conn.close()


@app.post("/api/documents")
async def upload_document(file: UploadFile) -> dict:
    """Accept a PDF and ingest it in the background."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="only pdf files are accepted")

    config.ensure_dirs()
    destination = config.UPLOAD_DIR / file.filename
    with destination.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    job = Job(id=uuid.uuid4().hex[:12], filename=file.filename)
    _save_job(job)
    threading.Thread(target=_run_ingest, args=(job, destination), daemon=True).start()
    # The same shape as GET /api/jobs/{id}. Returning a differently-named field
    # here (`job_id`) than the one the client reads back (`id`) meant the client
    # polled /api/jobs/undefined and got a 404 that looked like a missing route.
    return asdict(job)


def _run_ingest(job: Job, path: Path) -> None:
    job.status = "running"
    _save_job(job)
    conn = get_conn()
    try:
        report = ingest_document(
            path, conn=conn, client=LLMClient(), extract_model=config.EXTRACT_MODEL
        )
        payload = asdict(report)
        payload.pop("metadata", None)
        payload["grounding_rejection_rate"] = report.grounding_rejection_rate
        payload["repair_rate"] = report.repair_rate
        if report.metadata:
            payload["publisher"] = report.metadata.publisher
            payload["doc_type"] = report.metadata.doc_type
        job.report = payload
        job.status = "done"
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller as job state
        job.status = "failed"
        job.detail = f"{type(exc).__name__}: {exc}"
    finally:
        conn.close()
        _save_job(job)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = _load_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return asdict(job)


@app.get("/api/documents/{doc_id}/pdf")
def get_pdf(doc_id: int) -> FileResponse:
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT stored_path, filename FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
    finally:
        conn.close()
    if row is None or not Path(row["stored_path"]).exists():
        raise HTTPException(status_code=404, detail="pdf not available")
    return FileResponse(row["stored_path"], media_type="application/pdf")


# --- facts -----------------------------------------------------------------


@app.get("/api/facts")
def list_facts(
    entity: str | None = None,
    metric: str | None = None,
    doc_id: int | None = None,
    q: str | None = None,
    limit: int = Query(200, le=1000),
    offset: int = 0,
) -> dict:
    clauses, params = ["f.status = 'active'"], []
    if entity:
        clauses.append("e.canonical_name LIKE ?")
        params.append(f"%{entity.lower()}%")
    if metric:
        clauses.append("m.canonical_name LIKE ?")
        params.append(f"%{metric.lower()}%")
    if doc_id:
        clauses.append("f.doc_id = ?")
        params.append(doc_id)
    if q:
        clauses.append("(m.canonical_name LIKE ? OR f.quote LIKE ? OR f.value_raw LIKE ?)")
        params.extend([f"%{q.lower()}%", f"%{q}%", f"%{q}%"])

    where = " WHERE " + " AND ".join(clauses)
    conn = get_conn()
    try:
        total = conn.execute(
            "SELECT COUNT(*) AS n FROM facts f "
            "LEFT JOIN entities e ON e.id=f.entity_id "
            "LEFT JOIN metrics m ON m.id=f.metric_id " + where,
            params,
        ).fetchone()["n"]
        rows = _rows(
            conn.execute(
                FACT_SELECT + where + " ORDER BY f.id DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            )
        )
    finally:
        conn.close()
    for row in rows:
        row["bboxes"] = _decode_bboxes(row.pop("bbox_json", None))
    return {"total": total, "facts": rows}


@app.get("/api/facts/{fact_id}")
def get_fact(fact_id: int) -> dict:
    conn = get_conn()
    try:
        row = conn.execute(FACT_SELECT + " WHERE f.id = ?", (fact_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="unknown fact")
        fact = dict(row)
        fact["bboxes"] = _decode_bboxes(fact.pop("bbox_json", None))

        related = _rows(
            conn.execute(
                """
                SELECT r.*,
                       CASE WHEN r.fact_a = ? THEN r.fact_b ELSE r.fact_a END AS other_id
                FROM relations r WHERE r.fact_a = ? OR r.fact_b = ?
                """,
                (fact_id, fact_id, fact_id),
            )
        )
        for relation in related:
            other = conn.execute(
                FACT_SELECT + " WHERE f.id = ?", (relation["other_id"],)
            ).fetchone()
            relation["other"] = dict(other) if other else None
            if relation["other"]:
                relation["other"]["bboxes"] = _decode_bboxes(
                    relation["other"].pop("bbox_json", None)
                )
        fact["relations"] = related
        return fact
    finally:
        conn.close()


# --- reconciliation --------------------------------------------------------


@app.get("/api/conflicts")
def list_conflicts(
    verdict: str | None = None, limit: int = Query(200, le=1000)
) -> list[dict]:
    """Relation pairs, newest first, with both sides' evidence attached."""
    clauses, params = [], []
    if verdict:
        clauses.append("r.verdict = ?")
        params.append(verdict)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

    conn = get_conn()
    try:
        relations = _rows(
            conn.execute(
                f"SELECT r.* FROM relations r{where} "
                "ORDER BY CASE r.verdict WHEN 'contradicts' THEN 0 "
                "WHEN 'reconciled' THEN 1 ELSE 2 END, r.id DESC LIMIT ?",
                [*params, limit],
            )
        )
        for relation in relations:
            for side in ("a", "b"):
                row = conn.execute(
                    FACT_SELECT + " WHERE f.id = ?", (relation[f"fact_{side}"],)
                ).fetchone()
                fact = dict(row) if row else None
                if fact:
                    fact["bboxes"] = _decode_bboxes(fact.pop("bbox_json", None))
                relation[side] = fact
        return relations
    finally:
        conn.close()


@app.get("/api/clusters")
def list_clusters(limit: int = Query(100, le=500)) -> list[dict]:
    """Core-key clusters holding more than one fact -- where reconciliation happens."""
    conn = get_conn()
    try:
        return _rows(
            conn.execute(
                """
                SELECT f.core_key,
                       e.canonical_name AS entity, m.canonical_name AS metric,
                       f.period_start, f.period_end,
                       COUNT(*) AS n_facts,
                       COUNT(DISTINCT f.doc_id) AS n_docs
                FROM facts f
                LEFT JOIN entities e ON e.id = f.entity_id
                LEFT JOIN metrics  m ON m.id = f.metric_id
                WHERE f.status = 'active'
                GROUP BY f.core_key
                HAVING n_facts > 1
                ORDER BY n_docs DESC, n_facts DESC
                LIMIT ?
                """,
                (limit,),
            )
        )
    finally:
        conn.close()


@app.get("/api/clusters/{core_key:path}")
def get_cluster(core_key: str) -> dict:
    conn = get_conn()
    try:
        facts = load_cluster(conn, core_key)
        rows = _rows(conn.execute(FACT_SELECT + " WHERE f.core_key = ?", (core_key,)))
        for row in rows:
            row["bboxes"] = _decode_bboxes(row.pop("bbox_json", None))
        return {"core_key": core_key, "n_facts": len(facts), "facts": rows}
    finally:
        conn.close()


# --- stats -----------------------------------------------------------------


@app.get("/api/stats")
def stats() -> dict:
    conn = get_conn()
    try:
        one = lambda sql: conn.execute(sql).fetchone()[0]
        verdicts = {
            row["verdict"]: row["n"]
            for row in conn.execute(
                "SELECT verdict, COUNT(*) n FROM relations GROUP BY verdict"
            )
        }
        failures = _rows(
            conn.execute(
                "SELECT stage, reason, COUNT(*) n FROM failures "
                "GROUP BY stage, reason ORDER BY n DESC LIMIT 15"
            )
        )
        grounded = one("SELECT COUNT(*) FROM facts")
        ungrounded = one("SELECT COUNT(*) FROM failures WHERE stage = 'ground'")
        checked = grounded + ungrounded
        return {
            "documents": one("SELECT COUNT(*) FROM documents"),
            "pages": one("SELECT COALESCE(SUM(n_pages),0) FROM documents"),
            "facts": grounded,
            "entities": one("SELECT COUNT(*) FROM entities"),
            "metrics": one("SELECT COUNT(*) FROM metrics"),
            "relations": one("SELECT COUNT(*) FROM relations"),
            "verdicts": verdicts,
            "grounding_rejection_rate": (ungrounded / checked) if checked else 0.0,
            "failures": failures,
        }
    finally:
        conn.close()


# --- ui --------------------------------------------------------------------

if (WEB_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    page = WEB_DIR / "templates" / "index.html"
    if not page.exists():
        return "<h1>Fact Knowledge Layer</h1><p>UI not built yet.</p>"
    return page.read_text()
