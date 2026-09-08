"""Have a model review the relations the rules flagged as worth a second look.

Only contradictions and competing forecasts are reviewed -- a few dozen calls
against thousands of facts. The model does not overrule the deterministic
verdict; its assessment is stored alongside so that a disagreement between rule
and reviewer stays visible.

Usage:  python scripts/adjudicate.py [--limit 50]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, db
from app.llm.client import LLMClient
from app.reason.adjudicate import adjudicate, pending


def main() -> int:
    parser = argparse.ArgumentParser(description="review flagged relations")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--db", type=Path, default=config.DB_PATH)
    args = parser.parse_args()

    if not config.LLM_API_KEY:
        print("LLM_API_KEY is not set", file=sys.stderr)
        return 2

    conn = db.connect(str(args.db))
    db.init_db(conn)

    try:
        queue = pending(conn, limit=args.limit)
        if not queue:
            print("nothing flagged for review")
            return 0

        print(f"reviewing {len(queue)} relations...\n")
        results = adjudicate(
            conn,
            client=LLMClient(),
            model=config.ADJUDICATE_MODEL or config.EXTRACT_MODEL,
            limit=args.limit,
        )

        agreed = sum(1 for r in results if r.agrees)
        print(f"{len(results)} reviewed · reviewer agreed with {agreed}\n")

        for result in results:
            row = conn.execute(
                "SELECT verdict, rule FROM relations WHERE id = ?", (result.relation_id,)
            ).fetchone()
            mark = "agrees" if result.agrees else "DISAGREES"
            print(f"[{mark}] {row['verdict']} / {row['rule']}  (confidence {result.confidence:.2f})")
            print(f"  {result.explanation}")
            if result.missing_context:
                print(f"  missing: {result.missing_context}")
            print()

        disputed = [r for r in results if not r.agrees]
        if disputed:
            print(
                f"{len(disputed)} relation(s) where the reviewer disagreed with the rule. "
                "Each is either a missing qualifier in extraction or a missing rule in the "
                "engine, and is worth reading."
            )
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
