"""Establish what a document is, once, before extracting anything from it.

Two things downstream depend on this. Filings refer to their own subject as
"the Company", so facts cannot be attributed without knowing whose report this
is. And conflicting facts are ranked by how authoritative and how recent their
sources are, which needs a publisher and a publication date.

One LLM call per document, not per region, so the cost is negligible.

Source tiers are assigned from document type rather than from a list of known
publishers, so an unfamiliar document still lands somewhere sensible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.ingest.parse import ParsedDoc
from app.llm.client import LLMClient, LLMError

# Higher wins when two facts conflict. Audited statements outrank management
# commentary, which outranks a slide deck.
SOURCE_TIERS = {
    "annual_report": 90,
    "prospectus": 85,
    "institutional_report": 80,
    "regulatory_filing": 85,
    "earnings_presentation": 50,
    "press_release": 40,
    "other": 50,
}

DOC_TYPES = list(SOURCE_TIERS)

METADATA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "publisher", "doc_type", "published_date", "primary_entity"],
    "properties": {
        "title": {"type": "string"},
        "publisher": {
            "type": "string",
            "description": "The organisation that published the document.",
        },
        "doc_type": {"type": "string", "enum": DOC_TYPES},
        "published_date": {
            "type": "string",
            "description": "ISO date (YYYY-MM-DD) of publication, or empty if unstated.",
        },
        "primary_entity": {
            "type": "string",
            "description": (
                "The entity the document is chiefly about -- the company whose "
                "results these are, or the country whose economy is analysed. "
                "This is what 'the Company' refers to elsewhere in the document."
            ),
        },
    },
}

PROMPT = """Identify this document from its opening pages.

`primary_entity` is the entity the document is chiefly about. Elsewhere the
document may call it "the Company" or "your Company"; give its actual name.

`published_date` must be ISO format (YYYY-MM-DD). Use an empty string if the
document does not state one. Do not guess.

OPENING PAGES:
{excerpt}"""


@dataclass
class DocumentMetadata:
    title: str
    publisher: str
    doc_type: str
    published_date: str | None
    primary_entity: str

    @property
    def source_tier(self) -> int:
        return SOURCE_TIERS.get(self.doc_type, SOURCE_TIERS["other"])

    @classmethod
    def unknown(cls, filename: str) -> DocumentMetadata:
        """Fallback so a metadata failure never blocks ingestion."""
        return cls(
            title=filename,
            publisher="",
            doc_type="other",
            published_date=None,
            primary_entity="",
        )


def _opening_excerpt(doc: ParsedDoc, *, max_chars: int = 6000) -> str:
    parts: list[str] = []
    budget = max_chars
    for page in doc.pages[:4]:
        chunk = page.text[:2500]
        parts.append(chunk)
        budget -= len(chunk)
        if budget <= 0:
            break
    return "\n\n".join(parts)[:max_chars]


def extract_metadata(
    doc: ParsedDoc, *, client: LLMClient, model: str
) -> DocumentMetadata:
    """Read the document's identity from its opening pages."""
    excerpt = _opening_excerpt(doc)
    if not excerpt.strip():
        return DocumentMetadata.unknown(doc.path.name)

    try:
        payload = client.complete_json(
            [{"role": "user", "content": PROMPT.format(excerpt=excerpt)}],
            model=model,
            schema=METADATA_SCHEMA,
            schema_name="document_metadata",
            max_tokens=4000,
        )
    except LLMError:
        return DocumentMetadata.unknown(doc.path.name)

    doc_type = str(payload.get("doc_type", "other"))
    published = str(payload.get("published_date", "")).strip()

    return DocumentMetadata(
        title=str(payload.get("title", "")).strip() or doc.path.name,
        publisher=str(payload.get("publisher", "")).strip(),
        doc_type=doc_type if doc_type in SOURCE_TIERS else "other",
        published_date=published or None,
        primary_entity=str(payload.get("primary_entity", "")).strip(),
    )
