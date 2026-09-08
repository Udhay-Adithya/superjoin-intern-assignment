-- Fact knowledge layer storage.
--
-- Two notions of time are kept apart deliberately:
--   valid time       (period_start / period_end) -- when the fact is true in the world
--   transaction time (created_at)                -- when this system learned it
-- Facts are never deleted; they are marked superseded. That combination is what
-- lets a director be "active" in a 2022 document and "resigned" in a 2024 one
-- without either record being destroyed.

CREATE TABLE IF NOT EXISTS documents (
    id             INTEGER PRIMARY KEY,
    filename       TEXT    NOT NULL,
    sha256         TEXT    NOT NULL UNIQUE,
    title          TEXT,
    publisher      TEXT,
    doc_type       TEXT,             -- prospectus | annual_report | presentation | institutional_report
    published_date TEXT,             -- ISO date, best effort
    source_tier    INTEGER NOT NULL DEFAULT 50,  -- higher wins ties; see reason/trust.py
    n_pages        INTEGER NOT NULL DEFAULT 0,
    stored_path    TEXT    NOT NULL,
    ingested_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS pages (
    id         INTEGER PRIMARY KEY,
    doc_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_no    INTEGER NOT NULL,     -- 1-indexed
    text       TEXT    NOT NULL,
    char_start INTEGER NOT NULL,     -- offset into the document-wide text stream
    char_end   INTEGER NOT NULL,
    UNIQUE (doc_id, page_no)
);

-- A block is the unit handed to the extractor. For tables this deliberately spans
-- the caption and header rows too: the unit "(Rs in Million)" and the qualifier
-- columns "Standalone / Consolidated" live there, not in the data row.
CREATE TABLE IF NOT EXISTS blocks (
    id         INTEGER PRIMARY KEY,
    doc_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_no    INTEGER NOT NULL,
    kind       TEXT    NOT NULL,     -- paragraph | table | heading
    text       TEXT    NOT NULL,
    char_start INTEGER NOT NULL,
    char_end   INTEGER NOT NULL,
    bbox_json  TEXT,
    ordinal    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS entities (
    id             INTEGER PRIMARY KEY,
    canonical_name TEXT    NOT NULL UNIQUE,
    kind           TEXT    NOT NULL,   -- company | country | person | other
    aliases_json   TEXT    NOT NULL DEFAULT '[]',
    created_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- Grows as new kinds of facts appear; this is the dynamic-schema story.
CREATE TABLE IF NOT EXISTS metrics (
    id             INTEGER PRIMARY KEY,
    canonical_name TEXT    NOT NULL UNIQUE,
    aliases_json   TEXT    NOT NULL DEFAULT '[]',
    unit_class     TEXT    NOT NULL,   -- currency | percent | count | mass | state | other
    created_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS facts (
    id            INTEGER PRIMARY KEY,
    doc_id        INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    block_id      INTEGER NOT NULL REFERENCES blocks(id)    ON DELETE CASCADE,
    entity_id     INTEGER REFERENCES entities(id),
    metric_id     INTEGER REFERENCES metrics(id),

    -- as written in the document
    value_raw     TEXT    NOT NULL,
    unit_raw      TEXT,
    period_raw    TEXT,
    basis_raw     TEXT,

    -- after deterministic normalization
    value_norm    REAL,               -- canonical magnitude (see normalize/numbers.py)
    unit          TEXT,               -- INR | PERCENT | COUNT | ...
    scale         TEXT,               -- crore | lakh | million | ... (as found)
    currency      TEXT,

    -- valid time: when the fact holds in the world
    period_start  TEXT,
    period_end    TEXT,
    period_kind   TEXT,               -- fiscal_year | fiscal_quarter | calendar_year
                                      -- | calendar_quarter | instant | interval
    -- qualifiers that decide comparability
    basis         TEXT,               -- standalone | consolidated | null
    modality      TEXT NOT NULL DEFAULT 'actual',  -- actual | projected | estimated | restated
    variant       TEXT,               -- e.g. services vs customers; real vs nominal
    polarity      TEXT NOT NULL DEFAULT 'positive',

    -- evidence, verified by the grounding gate
    quote         TEXT    NOT NULL,
    page_no       INTEGER NOT NULL,
    char_start    INTEGER NOT NULL,
    char_end      INTEGER NOT NULL,
    bbox_json     TEXT,

    confidence    REAL    NOT NULL DEFAULT 0.0,
    core_key      TEXT,               -- (entity, metric, period)
    full_key      TEXT,               -- core + (basis, modality, variant)

    -- transaction time + supersession
    status        TEXT    NOT NULL DEFAULT 'active',   -- active | superseded
    superseded_by INTEGER REFERENCES facts(id),
    created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_facts_core   ON facts(core_key);
CREATE INDEX IF NOT EXISTS idx_facts_full   ON facts(full_key);
CREATE INDEX IF NOT EXISTS idx_facts_doc    ON facts(doc_id);
CREATE INDEX IF NOT EXISTS idx_facts_status ON facts(status);

CREATE TABLE IF NOT EXISTS relations (
    id                 INTEGER PRIMARY KEY,
    fact_a             INTEGER NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    fact_b             INTEGER NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    verdict            TEXT    NOT NULL,  -- corroborates | contradicts | reconciled
    rule               TEXT    NOT NULL,  -- which deterministic rule fired
    delta              REAL,              -- relative difference where meaningful
    explanation        TEXT,
    confidence         REAL    NOT NULL DEFAULT 0.0,
    qualifier_inferred INTEGER NOT NULL DEFAULT 0,

    -- A model reviews flagged pairs but does not overrule the deterministic
    -- verdict. Its assessment is recorded alongside so a disagreement between
    -- rule and reviewer is visible rather than silently resolved.
    adjudicated        INTEGER NOT NULL DEFAULT 0,
    adjudicator_agrees INTEGER,
    adjudicator_note   TEXT,
    missing_context    TEXT,
    created_at         TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (fact_a, fact_b)
);

CREATE INDEX IF NOT EXISTS idx_relations_verdict ON relations(verdict);

-- Every rejection is recorded rather than silently dropped. This table is what
-- makes the honest failure metric reportable.
CREATE TABLE IF NOT EXISTS failures (
    id         INTEGER PRIMARY KEY,
    doc_id     INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    block_id   INTEGER REFERENCES blocks(id)    ON DELETE CASCADE,
    stage      TEXT    NOT NULL,   -- extract | ground | normalize | compare
    reason     TEXT    NOT NULL,
    payload    TEXT,
    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_failures_stage ON failures(stage);
