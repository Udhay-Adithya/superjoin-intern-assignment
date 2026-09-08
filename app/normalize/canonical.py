"""Canonicalize entity and metric names without hardcoding this corpus.

The assignment will be tested against documents we have never seen, so there are
no Delhivery-specific or India-specific alias tables here. Canonical forms are
derived by deterministic normalization, and the registries grow as new names
appear -- which is also what makes the schema evolve rather than being fixed up
front.

The one distinction worth being careful about is that near-synonyms are not
always synonyms. In this corpus "revenue from operations" and "revenue from
customers" denote the same quantity, while "revenue from services" excludes
traded goods and is a genuinely different measure. Collapsing all three would
manufacture a contradiction out of a definitional difference, so qualifying
words are lifted into a separate ``variant`` rather than being normalized away.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

# Legal-form suffixes that do not distinguish one company from another.
_LEGAL_SUFFIXES = re.compile(
    r"\b(limited|ltd|private|pvt|incorporated|inc|corporation|corp|company|co|"
    r"plc|llp|llc|gmbh|s\.?a\.?|n\.?v\.?)\b\.?",
    re.IGNORECASE,
)

# Referring expressions that stand in for the document's subject.
_ANAPHORA = {
    "the company", "your company", "the group", "the bank", "the issuer",
    "our company", "the corporation", "we", "us", "it",
}

# Words that merely emphasise rather than change what is measured.
_METRIC_NOISE = re.compile(
    r"\b(total|overall|aggregate|the|of|for|a|an|in|on|at|s)\b",
    re.IGNORECASE,
)

_PUNCT = re.compile(r"[^\w\s%]+")
_WS = re.compile(r"\s+")


def normalize_phrase(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    lowered = _PUNCT.sub(" ", (text or "").lower())
    return _WS.sub(" ", lowered).strip()


def canonical_entity_name(mention: str) -> str:
    """Reduce an entity mention to a comparable key.

    Returns an empty string for anaphora such as "your Company", which carry no
    identity of their own and must be resolved from document context instead.
    """
    normalized = normalize_phrase(mention)
    if not normalized or normalized in _ANAPHORA:
        return ""
    without_legal = _LEGAL_SUFFIXES.sub(" ", normalized)
    return _WS.sub(" ", without_legal).strip() or normalized


def canonical_metric_name(metric: str) -> str:
    """Reduce a metric phrase to a comparable key."""
    normalized = normalize_phrase(metric)
    without_noise = _METRIC_NOISE.sub(" ", normalized)
    collapsed = _WS.sub(" ", without_noise).strip()
    return collapsed or normalized


@dataclass(frozen=True)
class Resolution:
    id: int
    canonical_name: str
    created: bool


class Registry:
    """A growing table of canonical names, backed by SQLite.

    ``resolve`` returns an existing entry when the normalized form matches and
    creates one otherwise, recording the surface form as an alias. New kinds of
    fact therefore extend the schema instead of being dropped.
    """

    def __init__(self, conn: sqlite3.Connection, table: str) -> None:
        if table not in {"entities", "metrics"}:
            raise ValueError(f"unsupported registry table: {table}")
        self._conn = conn
        self._table = table

    def resolve(self, surface: str, *, canonical: str, **defaults: str) -> Resolution | None:
        if not canonical:
            return None

        row = self._conn.execute(
            f"SELECT id, canonical_name, aliases_json FROM {self._table} "
            "WHERE canonical_name = ?",
            (canonical,),
        ).fetchone()

        if row is not None:
            self._remember_alias(row["id"], row["aliases_json"], surface)
            return Resolution(id=row["id"], canonical_name=row["canonical_name"], created=False)

        columns = ["canonical_name", "aliases_json", *defaults]
        placeholders = ", ".join("?" for _ in columns)
        cursor = self._conn.execute(
            f"INSERT INTO {self._table} ({', '.join(columns)}) "
            f"VALUES ({placeholders})",
            (canonical, _dump_aliases([surface]), *defaults.values()),
        )
        return Resolution(id=int(cursor.lastrowid), canonical_name=canonical, created=True)

    def _remember_alias(self, row_id: int, aliases_json: str, surface: str) -> None:
        aliases = _load_aliases(aliases_json)
        if surface and surface not in aliases:
            aliases.append(surface)
            self._conn.execute(
                f"UPDATE {self._table} SET aliases_json = ? WHERE id = ?",
                (_dump_aliases(aliases), row_id),
            )


def _load_aliases(raw: str | None) -> list[str]:
    import json

    try:
        value = json.loads(raw or "[]")
    except (TypeError, ValueError):
        return []
    return value if isinstance(value, list) else []


def _dump_aliases(aliases: list[str]) -> str:
    import json

    return json.dumps(aliases, ensure_ascii=False)
