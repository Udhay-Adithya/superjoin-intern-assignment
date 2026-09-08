"""Turn a text region into candidate claim frames.

The model's job here is deliberately narrow: find facts, copy them down exactly
as written, and quote the evidence. It does not convert units, resolve fiscal
years, compute growth rates or decide whether two facts agree. Everything that
can be done deterministically is done in Python, where it is testable and where
a weaker model costs recall rather than correctness.

The schema is n-ary rather than a subject/predicate/object triple because
whether two figures conflict depends on qualifiers a triple cannot hold. Revenue
of 74,540.82 and revenue of 81,415.38 for the same company in the same year are
not a contradiction once you know one is standalone and the other consolidated.
That qualifier has to survive extraction or the comparison engine is guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.extract.ground import GroundingError, ground_fact
from app.ingest.segment import Region
from app.llm.client import LLMClient, LLMError

MODALITIES = ["actual", "projected", "estimated", "restated"]

FACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["facts"],
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "subject",
                    "metric",
                    "value_raw",
                    "unit_raw",
                    "period_raw",
                    "basis_raw",
                    "variant_raw",
                    "modality",
                    "quote",
                ],
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "The entity the fact is about, as named in the text.",
                    },
                    "metric": {
                        "type": "string",
                        "description": "What is measured or asserted, as named in the text.",
                    },
                    "value_raw": {
                        "type": "string",
                        "description": "The value exactly as written, digits and all.",
                    },
                    "unit_raw": {
                        "type": "string",
                        "description": (
                            "The unit governing the value, including one given in a "
                            "caption or column header such as '(Rs in Million)'. "
                            "Empty string if none is stated."
                        ),
                    },
                    "period_raw": {
                        "type": "string",
                        "description": "The period the value covers, exactly as written.",
                    },
                    "basis_raw": {
                        "type": "string",
                        "description": (
                            "Scope qualifier such as Standalone or Consolidated, often "
                            "only in a column header. Empty string if none is stated."
                        ),
                    },
                    "variant_raw": {
                        "type": "string",
                        "description": (
                            "Measure variant that changes what is counted, such as "
                            "'real' vs 'nominal', or 'revenue from services' vs "
                            "'revenue from customers'. Empty string if none."
                        ),
                    },
                    "modality": {"type": "string", "enum": MODALITIES},
                    "quote": {
                        "type": "string",
                        "description": (
                            "A verbatim substring of the excerpt containing the value."
                        ),
                    },
                },
            },
        }
    },
}

SYSTEM_PROMPT = (
    "You extract facts from financial and economic documents. "
    "You copy what the document says. You never convert, compute, round or infer values."
)

PROMPT_TEMPLATE = """Extract every fact from the excerpt below that has a definite value.

Rules:
1. Copy `value_raw`, `unit_raw`, `period_raw` and `basis_raw` EXACTLY as written.
   Do not convert units, do not compute growth rates, do not reformat numbers.
2. A table's unit and qualifiers usually sit in a caption or column header above
   the row, not in the row itself. Read them from there and attach them to every
   value in that column. A value whose unit or basis you drop becomes unusable.
3. `basis_raw` captures scope qualifiers such as Standalone or Consolidated.
4. `variant_raw` captures a distinction that changes what is being counted,
   such as real vs nominal, or revenue from services vs revenue from customers.
5. `modality` says what kind of claim it is:
   - "actual" for a reported outturn
   - "projected" for a forecast of a future period
   - "estimated" for a provisional or estimated figure
   - "restated" for a previously reported figure now revised
6. `quote` MUST be an exact substring of the excerpt that contains the value.
   Never paraphrase, never join text from different rows, never invent a quote.
7. Extract only what is present. If the excerpt has no facts, return an empty list.

EXCERPT (from page {page_no}):
{excerpt}"""


@dataclass
class CandidateFact:
    """A fact as the model reported it, before grounding and normalization."""

    subject: str
    metric: str
    value_raw: str
    unit_raw: str
    period_raw: str
    basis_raw: str
    variant_raw: str
    modality: str
    quote: str

    # populated by the grounding gate
    char_start: int | None = None
    char_end: int | None = None
    grounded_quote: str | None = None

    page_no: int = 0
    region_index: int = 0

    @classmethod
    def from_payload(cls, payload: dict, *, page_no: int, region_index: int) -> CandidateFact:
        modality = str(payload.get("modality", "actual")).lower()
        return cls(
            subject=str(payload.get("subject", "")).strip(),
            metric=str(payload.get("metric", "")).strip(),
            value_raw=str(payload.get("value_raw", "")).strip(),
            unit_raw=str(payload.get("unit_raw", "")).strip(),
            period_raw=str(payload.get("period_raw", "")).strip(),
            basis_raw=str(payload.get("basis_raw", "")).strip(),
            variant_raw=str(payload.get("variant_raw", "")).strip(),
            modality=modality if modality in MODALITIES else "actual",
            quote=str(payload.get("quote", "")),
            page_no=page_no,
            region_index=region_index,
        )


@dataclass
class ExtractionResult:
    """Grounded facts plus every rejection, so the failure rate is reportable."""

    facts: list[CandidateFact] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)  # (reason, detail)

    @property
    def grounding_rejection_rate(self) -> float:
        total = len(self.facts) + len(self.rejected)
        return len(self.rejected) / total if total else 0.0


def extract_region(
    region: Region,
    *,
    client: LLMClient,
    model: str,
    region_index: int = 0,
) -> ExtractionResult:
    """Extract and ground the facts in one region."""
    result = ExtractionResult()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": PROMPT_TEMPLATE.format(page_no=region.page_no, excerpt=region.text),
        },
    ]

    try:
        payload = client.complete_json(
            messages, model=model, schema=FACT_SCHEMA, schema_name="facts"
        )
    except LLMError as exc:
        result.rejected.append(("llm_error", str(exc)))
        return result

    for raw in payload.get("facts", []):
        if not isinstance(raw, dict):
            result.rejected.append(("malformed", repr(raw)[:200]))
            continue

        candidate = CandidateFact.from_payload(
            raw, page_no=region.page_no, region_index=region_index
        )
        if not candidate.value_raw or not candidate.metric:
            result.rejected.append(("incomplete", f"{candidate.metric!r}={candidate.value_raw!r}"))
            continue

        try:
            grounded = ground_fact(
                quote=candidate.quote,
                value_raw=candidate.value_raw,
                source=region.text,
                source_offset=region.char_start,
            )
        except GroundingError as exc:
            result.rejected.append((exc.reason, f"{candidate.metric}={candidate.value_raw}"))
            continue

        candidate.char_start = grounded.char_start
        candidate.char_end = grounded.char_end
        candidate.grounded_quote = grounded.quote
        result.facts.append(candidate)

    return result
