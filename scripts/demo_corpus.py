"""Build the demo knowledge layer used in the README and the video.

Full-corpus ingestion works, but the 511 starter pages cost roughly 9.8M tokens,
which is about twenty hours against a free tier metered at 8,000 tokens per
minute. That is a provider constraint rather than a property of the system --
parsing and reconciling all 511 pages takes about three seconds -- so the demo
ingests the sections that carry the four required cases and leaves whole-corpus
runs to `scripts/ingest.py`.

Each range below is chosen because a specific case lives in it:

  annual report p20-26   standalone vs consolidated revenue in one table,
                         restated in prose alongside it        -> cases 1 and 3
  earnings deck p13-18   the same FY24 revenue in crore rather
                         than million, in a different document -> case 1
  RBI p16-18             FY2025-26 real GDP growth projection
  IMF p12-14             the same projection, different vintage -> case 2
  Economic Survey p20-22  a third estimate of the same figure

Usage:  python scripts/demo_corpus.py [--db data/facts.db]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, db
from app.ingest.pipeline import ingest_document
from app.llm.client import LLMClient

DATASETS = Path("starter-datasets")

PLAN: list[tuple[Path, tuple[int, int], str]] = [
    (
        DATASETS / "delhivery/02-delhivery-annual-report-fy24-excerpt.pdf",
        (20, 26),
        "standalone vs consolidated revenue, plus the prose restating it",
    ),
    (
        DATASETS / "delhivery/03-delhivery-q4-fy24-earnings-presentation.pdf",
        (13, 18),
        "the same FY24 revenue reported in crore",
    ),
    (
        DATASETS / "india-macroeconomy/02-rbi-annual-report-2024-25-excerpt.pdf",
        (16, 18),
        "RBI projection of FY2025-26 real GDP growth",
    ),
    (
        DATASETS / "india-macroeconomy/03-imf-india-2025-article-iv-excerpt.pdf",
        (12, 14),
        "IMF projection of the same figure, six months later",
    ),
    (
        DATASETS / "india-macroeconomy/01-india-economic-survey-2024-25-excerpt.pdf",
        (20, 22),
        "Economic Survey estimate of the same figure",
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="build the demo knowledge layer")
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    args = parser.parse_args()

    if not config.LLM_API_KEY:
        print("LLM_API_KEY is not set; copy .env.example to .env first", file=sys.stderr)
        return 2

    conn = db.connect(str(args.db))
    db.init_db(conn)

    started = time.time()
    totals = {"stored": 0, "relations": 0, "grounded": 0, "ungrounded": 0, "repaired": 0}

    # One client for the whole run: it remembers which keys have hit their daily
    # cap, so a fresh one per document would rediscover that at the cost of a
    # wasted request each time.
    client = LLMClient()

    try:
        for path, pages, why in PLAN:
            if not path.exists():
                print(f"missing: {path}", file=sys.stderr)
                continue

            print(f"\n{path.name}  pages {pages[0]}-{pages[1]}\n  ({why})", flush=True)
            begin = time.time()
            report = ingest_document(
                path,
                conn=conn,
                client=client,
                extract_model=config.EXTRACT_MODEL,
                metadata_model=config.ADJUDICATE_MODEL or None,
                pages=pages,
            )
            if report.already_ingested:
                print("  already ingested", flush=True)
                continue

            for key, value in (
                ("stored", report.n_stored),
                ("relations", report.n_relations),
                ("grounded", report.n_grounded),
                ("ungrounded", report.n_ungrounded),
                ("repaired", report.n_repaired),
            ):
                totals[key] += value

            print(
                f"  {report.n_stored} facts, {report.n_relations} relations "
                f"from {report.n_regions_extracted} regions in {time.time() - begin:.0f}s",
                flush=True,
            )
            print(
                f"  grounding rejected {report.grounding_rejection_rate:.1%}"
                f" · evidence repaired {report.repair_rate:.1%}"
                f" · unusable rows {report.n_unusable}",
                flush=True,
            )

        checked = totals["grounded"] + totals["ungrounded"]
        # Counted from storage rather than summed across documents: reconciling
        # a later document rewrites clusters an earlier one already wrote, so
        # adding up per-document totals counts the same relation twice.
        relations = conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        facts = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        print(
            f"\n{'=' * 62}\n"
            f"{facts} facts · {relations} relations · {time.time() - started:.0f}s\n"
            f"grounding rejected {totals['ungrounded'] / checked:.2%}"
            if checked
            else f"\n{facts} facts"
        )

        print("\nverdicts:")
        for row in conn.execute(
            "SELECT verdict, rule, COUNT(*) n FROM relations "
            "GROUP BY verdict, rule ORDER BY n DESC"
        ):
            print(f"  {row['verdict']:13s} {row['rule']:34s} {row['n']}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
