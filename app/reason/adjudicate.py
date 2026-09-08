"""Have a model review the pairs the rules found interesting.

This is the only place a language model touches reasoning, and it is given no
authority over the outcome. The deterministic verdict stands; the model adds a
readable account of what is going on, says whether it agrees, and names the
context a reader would need to settle the question.

Recording disagreement rather than resolving it is the point. If the rules say
two figures contradict and the reviewer says they are different vintages of the
same restated number, that gap is worth seeing -- it is either a missing
qualifier in the extractor or a missing rule in the engine, and hiding it behind
an overridden verdict would lose the signal.

Volume is tiny: only contradictions and competing forecasts are reviewed, which
is a few dozen calls against thousands of facts.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

from app.llm.client import LLMClient, LLMError, map_concurrent

ADJUDICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["agrees", "explanation", "confidence", "missing_context"],
    "properties": {
        "agrees": {
            "type": "boolean",
            "description": "Whether the stated verdict is the right reading of the evidence.",
        },
        "explanation": {
            "type": "string",
            "description": (
                "Two or three sentences a reader could act on, referring to the "
                "evidence quoted. Say what each side is measuring and why they differ."
            ),
        },
        "confidence": {"type": "number", "description": "0 to 1."},
        "missing_context": {
            "type": "string",
            "description": (
                "What a reader would need in order to settle this, if anything. "
                "Empty string when the evidence is sufficient."
            ),
        },
    },
}

SYSTEM_PROMPT = (
    "You review automated reconciliations of facts extracted from financial and "
    "economic documents. You judge only what the quoted evidence supports, and you "
    "say plainly when it is insufficient."
)

PROMPT = """Two facts were extracted from documents and compared automatically.

FACT A
  source     : {a_source}
  metric     : {a_metric}
  value      : {a_value} {a_unit}
  period     : {a_period}
  qualifiers : basis={a_basis}, modality={a_modality}, variant={a_variant}
  evidence   : "{a_quote}"

FACT B
  source     : {b_source}
  metric     : {b_metric}
  value      : {b_value} {b_unit}
  period     : {b_period}
  qualifiers : basis={b_basis}, modality={b_modality}, variant={b_variant}
  evidence   : "{b_quote}"

AUTOMATED VERDICT: {verdict} (rule: {rule})
REASONING: {explanation}

Assess it. Consider whether the two are really measuring the same thing: a
difference of scope, accounting basis, definition, vintage or measurement method
explains an apparent conflict without either side being wrong.

Judge only from the evidence quoted. Do not assume figures not shown."""


@dataclass
class Adjudication:
    relation_id: int
    agrees: bool
    explanation: str
    confidence: float
    missing_context: str


# Corroborations are not worth a model call: two figures that agree within their
# own stated precision need no narration.
REVIEWABLE_VERDICTS = ("contradicts",)
REVIEWABLE_RULES = ("differing_forecast",)


_PENDING_SQL = """
SELECT r.id, r.verdict, r.rule, r.explanation,
       a.value_raw AS a_value, a.unit_raw AS a_unit, a.period_raw AS a_period,
       a.basis AS a_basis, a.modality AS a_modality, a.variant AS a_variant,
       a.quote AS a_quote, am.canonical_name AS a_metric,
       ad.publisher AS a_publisher, ad.title AS a_title, ad.published_date AS a_date,
       b.value_raw AS b_value, b.unit_raw AS b_unit, b.period_raw AS b_period,
       b.basis AS b_basis, b.modality AS b_modality, b.variant AS b_variant,
       b.quote AS b_quote, bm.canonical_name AS b_metric,
       bd.publisher AS b_publisher, bd.title AS b_title, bd.published_date AS b_date
FROM relations r
JOIN facts a ON a.id = r.fact_a
JOIN facts b ON b.id = r.fact_b
JOIN documents ad ON ad.id = a.doc_id
JOIN documents bd ON bd.id = b.doc_id
LEFT JOIN metrics am ON am.id = a.metric_id
LEFT JOIN metrics bm ON bm.id = b.metric_id
WHERE r.adjudicated = 0
"""


def _source(publisher: str | None, title: str | None, published: str | None) -> str:
    name = publisher or title or "unknown source"
    return f"{name} ({published[:10]})" if published else name


def pending(conn: sqlite3.Connection, limit: int = 50) -> list[sqlite3.Row]:
    """Relations worth a model's attention, hardest first."""
    placeholders = ",".join("?" for _ in REVIEWABLE_VERDICTS)
    rule_placeholders = ",".join("?" for _ in REVIEWABLE_RULES)
    return conn.execute(
        _PENDING_SQL
        + f" AND (r.verdict IN ({placeholders}) OR r.rule IN ({rule_placeholders}))"
        " ORDER BY CASE r.verdict WHEN 'contradicts' THEN 0 ELSE 1 END, r.id LIMIT ?",
        (*REVIEWABLE_VERDICTS, *REVIEWABLE_RULES, limit),
    ).fetchall()


def adjudicate_one(row: sqlite3.Row, *, client: LLMClient, model: str) -> Adjudication | None:
    prompt = PROMPT.format(
        a_source=_source(row["a_publisher"], row["a_title"], row["a_date"]),
        a_metric=row["a_metric"] or "",
        a_value=row["a_value"],
        a_unit=row["a_unit"] or "",
        a_period=row["a_period"] or "unstated",
        a_basis=row["a_basis"] or "unstated",
        a_modality=row["a_modality"],
        a_variant=row["a_variant"] or "unstated",
        a_quote=(row["a_quote"] or "").replace('"', "'")[:400],
        b_source=_source(row["b_publisher"], row["b_title"], row["b_date"]),
        b_metric=row["b_metric"] or "",
        b_value=row["b_value"],
        b_unit=row["b_unit"] or "",
        b_period=row["b_period"] or "unstated",
        b_basis=row["b_basis"] or "unstated",
        b_modality=row["b_modality"],
        b_variant=row["b_variant"] or "unstated",
        b_quote=(row["b_quote"] or "").replace('"', "'")[:400],
        verdict=row["verdict"],
        rule=row["rule"],
        explanation=row["explanation"] or "",
    )

    try:
        payload = client.complete_json(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=model,
            schema=ADJUDICATION_SCHEMA,
            schema_name="adjudication",
            max_tokens=4000,
        )
    except LLMError:
        return None

    if not isinstance(payload, dict):
        return None

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return Adjudication(
        relation_id=int(row["id"]),
        agrees=bool(payload.get("agrees", True)),
        explanation=str(payload.get("explanation", "")).strip(),
        confidence=max(0.0, min(1.0, confidence)),
        missing_context=str(payload.get("missing_context", "")).strip(),
    )


def adjudicate(
    conn: sqlite3.Connection,
    *,
    client: LLMClient,
    model: str,
    limit: int = 50,
    workers: int = 4,
) -> list[Adjudication]:
    """Review pending relations and record what the model made of them."""
    rows = pending(conn, limit=limit)
    if not rows:
        return []

    results = map_concurrent(
        lambda row: adjudicate_one(row, client=client, model=model), rows, workers=workers
    )

    written = [r for r in results if r is not None]
    for adjudication in written:
        conn.execute(
            "UPDATE relations SET adjudicated = 1, adjudicator_agrees = ?, "
            "adjudicator_note = ?, missing_context = ?, confidence = ? WHERE id = ?",
            (
                int(adjudication.agrees),
                adjudication.explanation,
                adjudication.missing_context,
                adjudication.confidence,
                adjudication.relation_id,
            ),
        )
    conn.commit()
    return written
