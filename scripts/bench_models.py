"""Benchmark candidate extraction models on one hard, real region.

The region is the Delhivery FY24 Directors' Report financial summary. It is a
good probe because getting it right requires three things at once:

  * reading a borderless four-column table
  * picking up the unit from a caption that sits above the table
  * attaching the right Standalone/Consolidated qualifier to each column

A model that returns the numbers but drops the basis qualifier is useless here,
because that qualifier is what separates Case 3 from a false contradiction.

Usage:  python scripts/bench_models.py [model ...]
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI  # noqa: E402

from app import config  # noqa: E402
from app.ingest.parse import parse_pdf  # noqa: E402
from app.ingest.segment import segment_page  # noqa: E402

CANDIDATES = [
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "nvidia/nemotron-nano-3-30b-a3b",
    "writer/palmyra-fin-70b-32k",
    "google/gemma-4-31b-it",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "openai/gpt-oss-20b",
    "deepseek-ai/deepseek-v4-flash-0731",
    "mistralai/mistral-large-2-instruct",
]

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["facts"],
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["metric", "value_raw", "unit_raw", "period_raw", "basis_raw", "quote"],
                "properties": {
                    "metric": {"type": "string"},
                    "value_raw": {"type": "string"},
                    "unit_raw": {"type": "string"},
                    "period_raw": {"type": "string"},
                    "basis_raw": {"type": "string"},
                    "quote": {"type": "string"},
                },
            },
        }
    },
}

PROMPT_TEMPLATE = """Extract every numeric fact from this document excerpt.

Rules:
- Copy values, units, periods and basis EXACTLY as written. Never convert or compute.
- unit_raw: the unit governing the value, often in a caption such as "(Rs in Million)".
- basis_raw: the column qualifier, such as Standalone or Consolidated. Empty string if absent.
- period_raw: the period the value covers, as written.
- quote: a VERBATIM substring of the excerpt containing the value. Never paraphrase.

Return JSON matching the schema.

EXCERPT:
{excerpt}"""


def load_region() -> str:
    doc = parse_pdf("starter-datasets/delhivery/02-delhivery-annual-report-fy24-excerpt.pdf")
    return next(r for r in segment_page(doc.pages[21]) if "81,415.38" in r.text).text


def score(facts: list[dict], excerpt: str) -> dict:
    """Did the model attach the right basis to each revenue figure?"""

    def find(value: str) -> dict | None:
        for f in facts:
            if value in (f.get("value_raw") or "") and "evenue" in (f.get("metric") or ""):
                return f
        return None

    consolidated = find("81,415.38")
    standalone = find("74,540.82")
    grounded = sum(1 for f in facts if (f.get("quote") or "") and f["quote"] in excerpt)

    return {
        "n_facts": len(facts),
        "consolidated_ok": bool(
            consolidated and "consolidat" in (consolidated.get("basis_raw") or "").lower()
        ),
        "standalone_ok": bool(
            standalone and "standalone" in (standalone.get("basis_raw") or "").lower()
        ),
        "unit_ok": bool(consolidated and "illion" in (consolidated.get("unit_raw") or "")),
        "grounded": f"{grounded}/{len(facts)}" if facts else "0/0",
        "grounded_pct": (grounded / len(facts) * 100) if facts else 0.0,
    }


def probe(model: str, excerpt: str) -> dict:
    client = OpenAI(api_key=config.NIM_API_KEY, base_url=config.NIM_BASE_URL, timeout=600.0)
    started = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(excerpt=excerpt)}],
            max_tokens=8000,
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "facts", "schema": SCHEMA, "strict": True},
            },
        )
    except Exception as exc:  # noqa: BLE001 - a benchmark reports failures, it does not raise
        return {"model": model, "error": f"{type(exc).__name__}: {str(exc)[:110]}"}

    elapsed = time.time() - started
    message = response.choices[0].message
    try:
        facts = json.loads(message.content or "{}").get("facts", [])
    except json.JSONDecodeError as exc:
        return {"model": model, "seconds": round(elapsed, 1), "error": f"bad json: {exc}"}

    reasoning = getattr(message, "reasoning_content", "") or ""
    return {
        "model": model,
        "seconds": round(elapsed, 1),
        "in_tokens": response.usage.prompt_tokens,
        "out_tokens": response.usage.completion_tokens,
        "reasoning_chars": len(reasoning),
        **score(facts, excerpt),
    }


def _row(r: dict) -> str:
    if "error" in r:
        return f"{r['model']:42s}  ERROR  {r['error']}"
    basis = f"{int(r['consolidated_ok']) + int(r['standalone_ok'])}/2"
    return (
        f"{r['model']:42s} {r['seconds']:6.1f} {r['n_facts']:6d} {basis:>6s} "
        f"{'yes' if r['unit_ok'] else 'no':>5s} {r['grounded']:>9s}"
    )


HEADER = f"{'model':42s} {'secs':>6s} {'facts':>6s} {'basis':>6s} {'unit':>5s} {'grounded':>9s}"


def main() -> None:
    models = sys.argv[1:] or CANDIDATES
    excerpt = load_region()
    print(f"region: {len(excerpt)} chars\nprobing {len(models)} models concurrently...\n", flush=True)
    print(HEADER, flush=True)
    print("-" * len(HEADER), flush=True)

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(models)) as pool:
        futures = {pool.submit(probe, m, excerpt): m for m in models}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            # Print as each finishes -- a slow model must not hide the fast ones.
            print(_row(result), flush=True)

    print("\n--- ranked ---", flush=True)
    ok = [r for r in results if "error" not in r]
    ok.sort(key=lambda r: (-(r["consolidated_ok"] + r["standalone_ok"]), r["seconds"]))
    print(HEADER)
    for r in ok:
        print(_row(r))


if __name__ == "__main__":
    main()
