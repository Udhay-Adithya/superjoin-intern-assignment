"""SQLite access. Thin on purpose -- the interesting logic lives above this layer."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from app import config


def connect(path: str | None = None) -> sqlite3.Connection:
    config.ensure_dirs()
    conn = sqlite3.connect(path or config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(config.SCHEMA_PATH.read_text())
    conn.commit()


@contextmanager
def session(path: str | None = None) -> Iterator[sqlite3.Connection]:
    conn = connect(path)
    try:
        init_db(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def insert(conn: sqlite3.Connection, table: str, **values: Any) -> int:
    cols = ", ".join(values)
    placeholders = ", ".join("?" for _ in values)
    cur = conn.execute(
        f"INSERT INTO {table} ({cols}) VALUES ({placeholders})",  # noqa: S608 - fixed call sites
        tuple(values.values()),
    )
    return int(cur.lastrowid)


def record_failure(
    conn: sqlite3.Connection,
    *,
    stage: str,
    reason: str,
    doc_id: int | None = None,
    block_id: int | None = None,
    payload: str | None = None,
) -> int:
    """Every rejection is recorded, never silently dropped.

    The failures table is what makes the honest failure rate reportable.
    """
    return insert(
        conn,
        "failures",
        doc_id=doc_id,
        block_id=block_id,
        stage=stage,
        reason=reason,
        payload=payload,
    )
