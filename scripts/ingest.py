"""Ingest PDFs into the knowledge layer from the command line.

Examples
--------
    python scripts/ingest.py starter-datasets/delhivery/*.pdf
    python scripts/ingest.py --pages 20 30 report.pdf
    python scripts/ingest.py --reconcile-only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, db
from app.ingest.pipeline import ingest_document, reconcile_all
from app.llm.client import LLMClient


def main() -> int:
    parser = argparse.ArgumentParser(description="ingest pdfs into the fact knowledge layer")
    parser.add_argument("paths", nargs="*", type=Path, help="pdf files to ingest")
    parser.add_argument(
        "--pages",
        nargs=2,
        type=int,
        metavar=("FIRST", "LAST"),
        help="only extract from this inclusive 1-indexed page range",
    )
    parser.add_argument("--workers", type=int, default=config.LLM_WORKERS)
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    parser.add_argument(
        "--reconcile-only",
        action="store_true",
        help="recompute every relation without extracting anything",
    )
    args = parser.parse_args()

    if not args.paths and not args.reconcile_only:
        parser.error("give at least one pdf, or --reconcile-only")

    if not config.LLM_API_KEY and not args.reconcile_only:
        print("LLM_API_KEY is not set; copy .env.example to .env first", file=sys.stderr)
        return 2

    conn = db.connect(str(args.db))
    db.init_db(conn)

    try:
        if args.reconcile_only:
            written = reconcile_all(conn)
            conn.commit()
            print(f"recomputed {written} relations")
            return 0

        client = LLMClient()
        pages = tuple(args.pages) if args.pages else None
        totals = {"stored": 0, "relations": 0, "grounded": 0, "ungrounded": 0}

        for path in args.paths:
            if not path.exists():
                print(f"skipping missing file: {path}", file=sys.stderr)
                continue

            started = time.time()
            report = ingest_document(
                path,
                conn=conn,
                client=client,
                extract_model=config.EXTRACT_MODEL,
                metadata_model=config.ADJUDICATE_MODEL or None,
                workers=args.workers,
                pages=pages,
            )
            elapsed = time.time() - started

            if report.already_ingested:
                print(f"{path.name}: already ingested (document {report.doc_id})")
                continue

            totals["stored"] += report.n_stored
            totals["relations"] += report.n_relations
            totals["grounded"] += report.n_grounded
            totals["ungrounded"] += report.n_ungrounded

            print(
                f"{path.name}: {report.n_stored} facts, {report.n_relations} relations "
                f"from {report.n_regions_extracted} regions in {elapsed:.0f}s"
            )
            print(
                f"    grounding rejected {report.grounding_rejection_rate:.1%}"
                f" · evidence repaired {report.repair_rate:.1%}"
                f" · unusable rows {report.n_unusable}"
            )
            if report.rejections:
                top = sorted(report.rejections.items(), key=lambda kv: -kv[1])[:4]
                print("    " + ", ".join(f"{reason}×{n}" for reason, n in top))

        checked = totals["grounded"] + totals["ungrounded"]
        print(
            f"\ntotal: {totals['stored']} facts, {totals['relations']} relations, "
            f"grounding rejected {totals['ungrounded'] / checked:.1%}"
            if checked
            else f"\ntotal: {totals['stored']} facts"
        )
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
