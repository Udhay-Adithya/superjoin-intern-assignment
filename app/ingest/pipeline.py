"""Ingest one document end to end, then reconcile it against everything known.

The order matters. Facts are extracted, grounded and normalized before anything
is compared, and comparison happens only within core-key clusters the new
document actually touched. That is what makes ingestion incremental: adding a
sixth document re-examines the handful of clusters it has something to say
about, not the whole knowledge layer.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from app import config, db
from app.extract.extractor import UNUSABLE_REASONS, extract_region
from app.ingest.metadata import DocumentMetadata, extract_metadata
from app.ingest.parse import ParsedDoc, bboxes_for_span, parse_pdf
from app.ingest.segment import Region, _numeric_token_count, segment_document
from app.llm.client import LLMClient, map_concurrent
from app.normalize.facts import NormalizationError, normalize_candidate, scale_multiplier
from app.normalize.periods import Period
from app.reason.compare import ComparableFact, compare_cluster

# A region with no numeral and barely any text cannot carry a fact worth
# storing. Skipping these cuts LLM calls by roughly two thirds.
MIN_REGION_CHARS = 60


@dataclass
class IngestReport:
    """What happened, including everything that did not work."""

    doc_id: int
    filename: str
    metadata: DocumentMetadata | None = None
    n_pages: int = 0
    n_regions: int = 0
    n_regions_extracted: int = 0
    n_candidates: int = 0
    n_grounded: int = 0
    n_ungrounded: int = 0
    n_unusable: int = 0
    n_repaired: int = 0
    n_stored: int = 0
    n_relations: int = 0
    rejections: dict[str, int] = field(default_factory=dict)
    already_ingested: bool = False

    @property
    def grounding_rejection_rate(self) -> float:
        """Share of well-formed claims whose evidence did not verify.

        Rows the model never produced properly are excluded: they measure its
        formatting, not whether it invents evidence.
        """
        checked = self.n_grounded + self.n_ungrounded
        return self.n_ungrounded / checked if checked else 0.0

    @property
    def repair_rate(self) -> float:
        return self.n_repaired / self.n_grounded if self.n_grounded else 0.0

    def note(self, reason: str) -> None:
        self.rejections[reason] = self.rejections.get(reason, 0) + 1


def _fact_bearing(region: Region) -> bool:
    return (
        len(region.text.strip()) >= MIN_REGION_CHARS
        and _numeric_token_count(region.text) >= 1
    )


def _page_for(doc: ParsedDoc, page_no: int):
    index = page_no - 1
    if 0 <= index < len(doc.pages):
        return doc.pages[index]
    return None


def ingest_document(
    path: str | Path,
    *,
    conn: sqlite3.Connection,
    client: LLMClient,
    extract_model: str,
    metadata_model: str | None = None,
    workers: int | None = None,
    max_regions: int | None = None,
    pages: tuple[int, int] | None = None,
) -> IngestReport:
    """Parse, extract, ground, normalize and store one PDF.

    ``pages`` restricts extraction to an inclusive 1-indexed page range. The
    whole document is still parsed and stored, so evidence offsets stay valid;
    only the LLM calls are limited. Useful for iterating on a section without
    paying to re-extract a hundred pages.
    """
    path = Path(path)
    workers = workers if workers is not None else config.LLM_WORKERS
    parsed = parse_pdf(path)

    existing = conn.execute(
        "SELECT id FROM documents WHERE sha256 = ?", (parsed.sha256,)
    ).fetchone()
    if existing is not None:
        return IngestReport(
            doc_id=int(existing["id"]),
            filename=path.name,
            already_ingested=True,
            n_pages=parsed.n_pages,
        )

    metadata = extract_metadata(
        parsed, client=client, model=metadata_model or extract_model
    )

    doc_id = db.insert(
        conn,
        "documents",
        filename=path.name,
        sha256=parsed.sha256,
        title=metadata.title,
        publisher=metadata.publisher,
        doc_type=metadata.doc_type,
        published_date=metadata.published_date,
        source_tier=metadata.source_tier,
        n_pages=parsed.n_pages,
        stored_path=str(path),
    )

    report = IngestReport(
        doc_id=doc_id, filename=path.name, metadata=metadata, n_pages=parsed.n_pages
    )

    for page in parsed.pages:
        db.insert(
            conn,
            "pages",
            doc_id=doc_id,
            page_no=page.page_no,
            text=page.text,
            char_start=page.char_start,
            char_end=page.char_end,
        )

    regions = segment_document(parsed.pages)
    report.n_regions = len(regions)

    block_ids: dict[int, int] = {}
    for ordinal, region in enumerate(regions):
        block_ids[ordinal] = db.insert(
            conn,
            "blocks",
            doc_id=doc_id,
            page_no=region.page_no,
            kind=region.kind,
            text=region.text,
            char_start=region.char_start,
            char_end=region.char_end,
            bbox_json=json.dumps(region.bbox),
            ordinal=ordinal,
        )

    targets = [(i, r) for i, r in enumerate(regions) if _fact_bearing(r)]
    if pages is not None:
        first, last = pages
        targets = [(i, r) for i, r in targets if first <= r.page_no <= last]
    if max_regions is not None:
        targets = targets[:max_regions]
    report.n_regions_extracted = len(targets)

    results = map_concurrent(
        lambda pair: extract_region(
            pair[1], client=client, model=extract_model, region_index=pair[0]
        ),
        targets,
        workers=workers,
    )

    touched_core_keys: set[str] = set()

    for (ordinal, region), result in zip(targets, results, strict=True):
        block_id = block_ids[ordinal]
        report.n_candidates += len(result.facts) + len(result.rejected)
        report.n_grounded += len(result.facts)
        report.n_ungrounded += len(result.ungrounded)
        report.n_unusable += len(result.unusable)
        report.n_repaired += result.n_repaired

        for reason, detail in result.rejected:
            report.note(reason)
            db.record_failure(
                conn,
                stage="extract" if reason in UNUSABLE_REASONS else "ground",
                reason=reason,
                doc_id=doc_id,
                block_id=block_id,
                payload=detail,
            )

        for candidate in result.facts:
            try:
                normalized = normalize_candidate(
                    candidate, conn=conn, default_entity=metadata.primary_entity
                )
            except NormalizationError as exc:
                report.note(exc.reason)
                db.record_failure(
                    conn,
                    stage="normalize",
                    reason=exc.reason,
                    doc_id=doc_id,
                    block_id=block_id,
                    payload=f"{candidate.metric}={candidate.value_raw}",
                )
                continue

            page = _page_for(parsed, candidate.page_no)
            bboxes = (
                bboxes_for_span(page, candidate.char_start or 0, candidate.char_end or 0)
                if page
                else []
            )

            period: Period | None = normalized.period
            db.insert(
                conn,
                "facts",
                doc_id=doc_id,
                block_id=block_id,
                entity_id=normalized.entity_id,
                metric_id=normalized.metric_id,
                value_raw=candidate.value_raw,
                unit_raw=candidate.unit_raw,
                period_raw=candidate.period_raw,
                basis_raw=candidate.basis_raw,
                value_norm=normalized.value_norm,
                unit=normalized.quantity.unit,
                scale=normalized.quantity.scale,
                currency=normalized.quantity.currency,
                period_start=period.start.isoformat() if period else None,
                period_end=period.end.isoformat() if period else None,
                period_kind=period.kind if period else None,
                basis=normalized.basis,
                modality=normalized.modality,
                variant=normalized.variant,
                quote=candidate.grounded_quote or candidate.quote,
                page_no=candidate.page_no,
                char_start=candidate.char_start or 0,
                char_end=candidate.char_end or 0,
                bbox_json=json.dumps(bboxes),
                confidence=1.0,
                core_key=normalized.core_key,
                full_key=normalized.full_key,
            )
            report.n_stored += 1
            touched_core_keys.add(normalized.core_key)

    conn.commit()
    report.n_relations = reconcile_keys(conn, touched_core_keys)
    conn.commit()
    return report


# --- reconciliation --------------------------------------------------------


def _row_to_comparable(row: sqlite3.Row) -> ComparableFact:
    period = None
    if row["period_start"] and row["period_end"]:
        from datetime import date

        period = Period(
            start=date.fromisoformat(row["period_start"]),
            end=date.fromisoformat(row["period_end"]),
            kind=row["period_kind"] or "interval",
            raw=row["period_raw"] or "",
        )

    unit = row["unit"] or "COUNT"
    unit_class = (
        "percent" if unit == "PERCENT" else "currency" if unit in {"INR", "USD", "EUR"} else "count"
    )

    from app.normalize.numbers import Quantity

    quantity = Quantity(
        value=row["value_norm"] or 0.0,
        unit=unit,
        scale=row["scale"],
        currency=row["currency"],
    )

    return ComparableFact(
        id=int(row["id"]),
        entity=row["entity_name"] or "",
        metric=row["metric_name"] or "",
        period=period,
        value_norm=row["value_norm"] or 0.0,
        unit=unit,
        unit_class=unit_class,
        value_raw=row["value_raw"],
        scale_multiplier=scale_multiplier(quantity, row["value_raw"]),
        basis=row["basis"],
        modality=row["modality"] or "actual",
        variant=row["variant"],
        doc_id=int(row["doc_id"]),
        core_key=row["core_key"] or "",
        block_id=int(row["block_id"] or 0),
        unit_raw=row["unit_raw"] or "",
        publisher=row["publisher"] or "",
        published_date=row["published_date"],
        source_tier=int(row["source_tier"] or 50),
        page_no=int(row["page_no"] or 0),
        quote=row["quote"] or "",
    )


_CLUSTER_SQL = """
SELECT f.*, e.canonical_name AS entity_name, m.canonical_name AS metric_name,
       d.publisher, d.published_date, d.source_tier
FROM facts f
LEFT JOIN entities  e ON e.id = f.entity_id
LEFT JOIN metrics   m ON m.id = f.metric_id
JOIN documents d ON d.id = f.doc_id
WHERE f.core_key = ? AND f.status = 'active'
"""


def load_cluster(conn: sqlite3.Connection, core_key: str) -> list[ComparableFact]:
    rows = conn.execute(_CLUSTER_SQL, (core_key,)).fetchall()
    return [_row_to_comparable(r) for r in rows]


def _collapse_collisions(relations: list, facts: list[ComparableFact]) -> list:
    """Keep one representative per colliding block, not every pair.

    A dense table whose rows all normalize to the same metric produces a
    quadratic number of identical findings -- the earnings deck alone generated
    167 of them. They all say the same thing: the labels in this block collapsed.
    One finding per block is the signal; the rest buries the genuine
    cross-document relations underneath it.
    """
    block_of = {fact.id: fact.block_id for fact in facts}
    kept, seen_blocks = [], set()

    for relation in relations:
        if relation.rule != "metric_label_collision":
            kept.append(relation)
            continue
        block = block_of.get(relation.fact_a, 0)
        if block in seen_blocks:
            continue
        seen_blocks.add(block)
        kept.append(relation)
    return kept


def reconcile_keys(conn: sqlite3.Connection, core_keys: set[str]) -> int:
    """Recompute relations for the given clusters only.

    Bounded work: a new document touches a limited set of core keys, so the
    knowledge layer never has to be rebuilt from scratch.
    """
    written = 0
    for core_key in core_keys:
        facts = load_cluster(conn, core_key)
        if len(facts) < 2:
            continue

        ids = [f.id for f in facts]
        placeholders = ",".join("?" for _ in ids)
        conn.execute(
            f"DELETE FROM relations WHERE fact_a IN ({placeholders}) "
            f"AND fact_b IN ({placeholders})",
            (*ids, *ids),
        )

        for relation in _collapse_collisions(compare_cluster(facts), facts):
            conn.execute(
                "INSERT OR IGNORE INTO relations "
                "(fact_a, fact_b, verdict, rule, delta, explanation, confidence, "
                " qualifier_inferred) VALUES (?,?,?,?,?,?,?,?)",
                (
                    relation.fact_a,
                    relation.fact_b,
                    relation.verdict,
                    relation.rule,
                    relation.delta,
                    relation.explanation,
                    1.0,
                    int(relation.qualifier_inferred),
                ),
            )
            written += 1
    return written


def reconcile_all(conn: sqlite3.Connection) -> int:
    """Recompute every cluster. Used after a change to the comparison rules."""
    rows = conn.execute(
        "SELECT DISTINCT core_key FROM facts WHERE status = 'active' AND core_key IS NOT NULL"
    ).fetchall()
    return reconcile_keys(conn, {r["core_key"] for r in rows})
