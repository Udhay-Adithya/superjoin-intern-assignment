# Fact Knowledge Layer

Extracts facts from PDFs, ties every fact to the text that supports it, and works
out when facts corroborate, contradict, or only appear to conflict.

> Superjoin VIT 2026 engineering intern assignment.

The central claim, which the rest of the design follows from:

> **A contradiction is not a semantic judgment. It is an arithmetic disagreement
> between two facts that occupy the same normalized frame. The language model's
> job is to fill the frame, not to decide the verdict.**

The model finds facts and quotes its evidence. Deterministic Python normalizes
them and decides how they relate. So verdicts are explainable and reproducible,
and a weaker or cheaper model costs recall rather than correctness.

---

## Setup and Run Instructions

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). Any OpenAI-compatible
LLM endpoint works.

```bash
git clone <this repo> && cd superjoin-intern-assignment
uv venv --python 3.12 && uv sync

cp .env.example .env      # then paste your key into LLM_API_KEY
```

`.env.example` defaults to Groq (free tier, `openai/gpt-oss-120b`). Cerebras and
NVIDIA NIM also work — set `LLM_BASE_URL` and the model names.

`LLM_API_KEY` accepts several comma-separated keys. Rate limits are charged per
key, so a pool multiplies both the per-minute and the per-day allowance, and
each key is retired individually as it hits its daily cap:

```
LLM_API_KEY="gsk_first,gsk_second,gsk_third"
```

**Run the app:**

```bash
.venv/bin/uvicorn app.main:app --port 8000
```

Open <http://localhost:8000>. Upload a PDF from the Upload tab, or build the
demo knowledge layer first:

```bash
.venv/bin/python scripts/demo_corpus.py
```

**Other entry points:**

```bash
.venv/bin/python scripts/ingest.py path/to/file.pdf        # ingest any PDFs
.venv/bin/python scripts/ingest.py --pages 20 26 file.pdf  # just one section
.venv/bin/python scripts/ingest.py --reconcile-only        # recompute relations
.venv/bin/python -m pytest -q                              # the test suite
```

The frontend is TypeScript compiled to plain ES modules. Compiled JS is
committed, so **nothing needs building to run this**. To change the frontend:

```bash
npm install --no-save typescript@5 && ./node_modules/.bin/tsc --watch
```

---

## Video Demo

_TBD_

---

## The Four Required Cases

Every block below is real output from `scripts/demo_corpus.py`, quoted from the
starter documents. No verdict here was decided by a language model.

### 1. A fact corroborated across documents, expressed differently

```
A  Delhivery annual report FY24 · p22
   81,415.38 ₹ in Million   basis=consolidated  variant=operations
   "Revenue from Operations   74,540.82   66,586.61   81,415.38   72,253.01"

B  Delhivery Q4 FY24 earnings deck · p17
   8,142 ₹ Cr               basis=unstated     variant=customers
   "Revenue from customers (A+B)  1,860  2,194  2,076 ... 7,225  8,142  12.7%"

→ Same quantity, written differently: 81,415.38 ₹ in Million and 8,142 ₹ Cr are
  the same figure once scaled, for the same period, despite differing on variant.
```

Two documents, two units, two metric phrasings. They meet because the scale is
normalized, the fiscal periods resolve to the same interval, and the qualifying
tail (`operations` / `customers`) is lifted out of the metric into `variant` so
both reduce to one measure.

### 2. A genuine disagreement

```
A  Reserve Bank of India · p17
   6.5 per cent    modality=projected   period=2025-26
   "real GDP growth for 2025-26 is projected at 6.5 per cent, with risks"

B  International Monetary Fund (2025-11-26) · p13
   6.6 percent     modality=projected   period=FY2025/26
   "real GDP growth is projected at 6.6 percent in FY2025/26, helped by the
    strong 2025Q2 growth outturn, the GST reform..."

→ Competing forecasts, not a contradiction: Reserve Bank of India projects 6.5 %
  and International Monetary Fund (2025-11-26) projects 6.6 percent for the same
  period. Forecasts differ by method and vintage.
```

Note what the system refuses to do: it does not pick a winner. Neither
institution is wrong, because the year has not happened. `2025-26` and
`FY2025/26` are recognised as the same interval despite different notation.

The reviewer model agreed, at 0.97 confidence:

> *Both sources quote projected real GDP growth for FY2025/26: RBI at 6.5% and
> IMF at 6.6%. The evidence shows they are forecasts, not actuals, and the small
> numeric difference is typical of differing projections, not a logical conflict.*

### 3. An apparent contradiction explained by context

```
A  Delhivery annual report FY24 · p22
   4,753.49 ₹ in Million    basis=standalone
B  Delhivery annual report FY24 · p22
   4,526.96 ₹ in Million    basis=consolidated
   "Other Income   4,753.49   3,311.74   4,526.96   3,049.48"

→ Not a contradiction: these measure different things.
  4,753.49 INR is standalone while 4,526.96 INR is consolidated.
```

Same company, same metric, same year, different numbers. The two share a core
key and diverge on `basis`, so the qualifier they differ on *is* the explanation.

A second flavour, resolved by definition rather than scope:

```
A  300,000 Preference Shares of ₹10 each      variant=10
B  4,660,337 Preference Shares of ₹100 each   variant=100

→ Not a contradiction: these measure different things.
```

### 4. An extraction failure, and how it is handled

```
A  Delhivery Q4 deck · p17    13
B  Delhivery Q4 deck · p17   109
   "EBITDA   13   109   46   (58.0%)   242.5%   (452)   127"

→ Same source passage, so not a document conflict: 13 and 109 were both read as
  'ebitda'. They are different quantities whose distinguishing detail was lost
  in extraction.
```

Those are quarterly columns the extractor failed to distinguish. The system will
not claim a document contradicts itself inside one passage — if two figures from
a single region share a full key and disagree, our metric resolution collapsed
two different things, and it says so rather than blaming the document.

This rule was written because of two earlier false contradictions it now catches:
a before/after pair in one sentence ("increasing it from ₹8,703.00 million to
₹8,863.03 million"), and the two preference-share classes above, which are now
correctly separated by `variant` instead.

---

## Approach

### Pipeline

```
PDF
 └─1─ parse       layout-preserving text + word coordinates      (PyMuPDF)
 └─2─ segment     blocks merged into regions that keep a value with its qualifiers
 └─3─ extract     LLM → n-ary claim frames, verbatim quote required
 └─4─ ground      verify the quote exists; verify the value is inside it
 └─5─ normalize   scale · period · entity · metric               (deterministic)
 └─6─ key         core key + full key
 └─7─ compare     CORROBORATES / CONTRADICTS / RECONCILED        (no LLM)
 └─8─ adjudicate  LLM reviews only flagged pairs, without authority to overrule
 └─9─ store       SQLite; facts are superseded, never deleted
```

### Facts are n-ary, not triples

`(Delhivery, revenue, 8142cr)` is unusable, because whether two revenue figures
conflict depends on period, basis, scope and modality — none of which fit in a
triple. Each fact is a frame:

```
subject · metric · value · unit · scale · currency
       · period_start · period_end · basis · modality · variant
       · evidence(doc, page, char span, bbox, verbatim quote)
```

Modelled on XBRL's `context / unit / decimals` and Wikidata's qualifier and
reference split. `modality` matters more than it looks: the RBI's 6.5% is a
*projection* and Delhivery's 81,415.38 is an *audited actual*, and comparing a
forecast to an outturn as though they were the same kind of claim is a category
error the field prevents.

### The grounding gate

Every extracted fact must prove itself:

1. the quoted evidence appears in the source, character for character
2. its offsets are recomputed from the located span
3. the extracted value appears *inside* that quote

Anything failing is dropped and counted. This catches a genuine failure mode —
a real sentence from the document paired with a number that appears nowhere in
it — with `str.find()` rather than another model call.

Matching tolerates whitespace and dash variants, because layout text holds table
columns apart with long space runs and models collapse them when quoting.
Rejecting a fact over invisible whitespace would measure the model's formatting
habits rather than its honesty.

### Comparison keys: the mechanism

Two levels, and the gap between them produces every verdict.

| key | contents | decides |
|---|---|---|
| **core** | entity, metric, period | what is worth comparing |
| **full** | core + basis, modality, variant | whether a difference is real |

| condition | verdict |
|---|---|
| same full key, values agree | **corroborates** |
| same full key, values disagree | **contradicts** |
| same core key, **different** full key | **reconciled** — the differing qualifier *is* the explanation |

Case 3 falls out for free: standalone and consolidated revenue share a core key
and differ on `basis`, so the system reports "not a contradiction, these differ
on basis" mechanically, with no model involved in the judgment.

A missing qualifier is a wildcard rather than a distinct value — an earnings deck
rarely says "consolidated" — and any relation resting on that is flagged
`qualifier_inferred`.

### Tolerance comes from the documents

Two values agree when, **rounded to the coarser of the two implied precisions,
they are the same number**. `8,142` in crore claims precision to the nearest
crore; `81,415.38` in million claims four more digits. The fine value rounds into
the coarse value's place, so they corroborate. No hand-tuned constant anywhere.

The obvious alternative — checking whether the two rounding *bands* overlap —
was implemented first and quietly failed: 6.5% and 6.6% both touch 6.55, so any
two adjacent one-decimal percentages "agreed", erasing the difference between two
institutions' published forecasts.

### Contradiction is reserved for settled facts

Two institutions forecasting different growth for a year that has not happened
are not contradicting each other. Competing projections are `reconciled` with
both publishers and vintages named; only conflicting *actuals* contradict.

### What this system will not claim

A document does not contradict itself inside a single passage. When two figures
from one extracted region share a full key and disagree, our metric resolution
collapsed two different things — so it is reported as `metric_label_collision`,
naming it as a limitation of the system rather than a conflict in the document.
Both real examples are in Limitations below.

### Storage

SQLite, with two notions of time kept apart: `period_start`/`period_end` is
**valid time** (when the fact holds in the world) and `created_at` is
**transaction time** (when this system learned it). Facts are marked superseded,
never deleted. That is bitemporal modelling without a graph server.

The graph is a *view* over the fact and relation tables, not the storage engine —
the assignment is explicit that a graph database alone is not the solution.

### Incremental ingestion

A new document touches a bounded set of core keys, so only those clusters are
recompared. Adding a sixth document does not rebuild the layer. This falls out of
the key design rather than being separate work.

### Things considered and rejected

**RAG.** Stateless and query-time. It never materializes a fact, never normalizes
`₹81,415.38 million` and `₹8,142 Cr` into the same quantity, and never notices two
documents disagree unless you happen to ask a question that surfaces both.
Retrieval survives here only as cheap candidate generation.

**Graphiti / Zep.** The closest existing system, and genuinely good at temporal
invalidation. Rejected because it runs its own extraction with its own schema —
which would outsource the exact thing being evaluated — it needs Neo4j or
FalkorDB standing up before anything runs, and its conflict model invalidates by
*recency*. Pointed at Case 3 it would declare ₹7,454 Cr and ₹8,142 Cr a conflict
and kill the older one. The bitemporal idea was worth taking; the dependency was
not.

### AI tools used

Claude (via Claude Code) for design discussion, implementation and debugging.
`openai/gpt-oss-120b` on Groq for extraction and adjudication at runtime.

---

## Limitations and Next Steps

**Period normalization is the weakest link.** 57.3% of the period labels in this
corpus rely on a convention that is assumed rather than read — `2024-25` is only a
fiscal year because Indian institutions write it that way. Every such fact is
flagged `inferred`.

**Metric resolution is too coarse, and it is the main source of error.** Two real
cases from a live run, both now caught as label collisions rather than reported
as contradictions:

- *"increasing it from ₹8,703.00 million to ₹8,863.03 million"* — a before/after
  pair in one sentence, extracted as two competing values for one metric.
- *"300,000 Preference Shares of ₹10 each"* against *"4,660,337 Preference Shares
  of ₹100 each"* — two share classes whose distinguishing face value was dropped.

The next step is a learned metric matcher over embeddings with an LLM adjudicating
the ambiguous band, replacing today's deterministic phrase normalization.

**Entity resolution is alias-and-normalization, not a learned matcher.** Splink or
a fine-tuned cross-encoder would be the upgrade. The current rule also had to
learn that a subject restating the metric is not an entity — the IMF's projection
arrived with subject "real GDP growth" instead of "India", which silently
prevented it ever meeting the RBI's figure for the same quantity.

**Document metadata is read by the model and is sometimes wrong.** The Economic
Survey excerpt is attributed to the Reserve Bank of India, because RBI is named
throughout the text and the model took the most prominent organisation rather
than the publisher. This matters: `source_tier` is derived from document type,
so a misread type mis-ranks a source when two facts conflict. A stricter
approach would read the publisher from the cover page alone, or verify it
against the document's own header and footer.

**No OCR path.** Every starter PDF has a clean text layer. A scanned PDF yields
close to nothing; the fix is a VLM fallback when the text layer is empty.

**Table structure is inferred from coordinates, not parsed.** Merged and
multi-row headers degrade. `find_tables()` was tried first and found only the
ruled header rows of these borderless financial tables, dropping every data row.

**Throughput is bounded by the provider, not the design.** Parsing and
reconciling all 511 starter pages takes about three seconds, and the demo corpus
rebuilds from cache in eight. Extraction is the cost: roughly 9.8M tokens for
the full corpus against a free tier capped at **200,000 tokens per day** and
8,000 per minute. That is a hard ceiling no amount of engineering removes, so
`scripts/demo_corpus.py` ingests the sections carrying the required cases and
`scripts/ingest.py` runs arbitrary PDFs given a larger quota.

The daily cap is worth calling out because it is invisible until it bites: it
returns the same HTTP 429 as a per-minute breach, and the per-minute headers
keep reporting healthy remaining capacity while every request fails. Hours went
into diagnosing "rate limiting" that was really a daily budget. The client now
distinguishes the two and fails immediately with the numbers, rather than
retrying something that cannot succeed.

**Cross-currency comparison is refused rather than attempted**, since it needs an
FX rate and an as-of date.

---

## Additional Notes

### Bugs worth mentioning, because running it found them

Scanning the corpus rather than trusting hand-written test cases caught a period
parser turning `uploads/2023/04/` — a URL path in a footnote — into a fiscal year
ending in **2104**. The fix generalizes: a fiscal span covers two *consecutive*
years. It now correctly rejects 83 occurrences (URL fragments, multi-year ranges
like `2017-23`, table line-wrap artifacts) with no false rejections.

### Measured on the demo corpus

| | |
|---|---|
| Documents / pages | 5 / 411 |
| Facts extracted | 351 |
| Relations found | 61 |
| Grounding rejection rate | **0.56%** |
| Rebuild from cache | 3s |
| Tests | 138 |

Verdict breakdown:

```
corroborates  agrees_despite_differing_qualifiers  17
reconciled    different_basis                      14
reconciled    different_variant                    12
reconciled    metric_label_collision                7
corroborates  within_implied_precision              4
reconciled    differing_forecast                    1
```

The first live extraction rejected **77%** of facts. The model was assembling
quotes by pairing a row label with one of its values — `"Revenue from Operations
81,415.38"` — a string that does not exist, because the real row carries four
figures. The grounding gate was right to reject it, but it was discarding real
facts over formatting. After tightening the prompt contract and adding a bounded
evidence repair, rejection fell to **0.45%**.

The earnings deck initially produced **zero facts from 51 regions**. A guard
written for two-page A4 spreads — refuse to merge blocks spanning more than 60%
of the page width — shredded every wide landscape slide table into individual
rows stripped of their headers.

### Tests

```bash
.venv/bin/python -m pytest -q
```

The four required cases are written as executable specifications in
`tests/test_compare.py`, using real figures from the starter documents. Most of
the suite runs with no API key: the language model is the one part that cannot be
tested deterministically, so it is stubbed and everything else — real PDFs, real
parsing, real grounding, real reconciliation — is exercised for real.
