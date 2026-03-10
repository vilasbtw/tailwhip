# tailwhip

Semantic and lexical search over Oracle schema metadata.

Find the right table in a Oracle database in seconds.

---

## The Problem

Large Oracle databases, especially ERP systems, have hundreds of tables with
similar names. When a client asks for a report, finding the right table requires
either deep schema knowledge or several minutes of manual searching.

## The Solution

tailwhip ingests an Oracle metadata export (CSV), builds a hybrid search index,
and returns the most relevant tables ranked by semantic similarity + FK centrality.

```bash
$ tailwhip search "part number and invoice value by process" --show-columns
```

---

## How It Works

```
Query
  │
  ├─► QueryExpander    — injects synonyms ("SKU" → also searches "part number")
  │
  ├─► Vector Search    — ChromaDB cosine similarity (understands PT-BR + EN)
  ├─► BM25 Search      — exact token matching
  │
  ├─► RRF              — merges both rankings
  ├─► FK Boost         — promotes central/hub tables (double log, softened)
  ├─► PK Boost         — promotes parent tables over child tables
  └─► Normalize        — final_score in [0, 1]
```

**Why hybrid?** Semantic search understands meaning but misses exact matches.
BM25 catches exact tokens but has no semantic understanding. RRF combines both,
giving the best of each.

**Why FK boost?** A table referenced by 10+ other tables is naturally more
important than a peripheral table with the same vector score.

**Why PK boost?** Without it, a child table with many inbound FKs outranks its
parent for queries about the parent concept. The PK boost detects when a term
is a PK in one table and a FK in another, and promotes the parent accordingly.

---

## Installation

```bash
git clone https://github.com/vilasbtw/tailwhip
cd tailwhip
pip install -e ".[dev]"
```

**Requirements:** Python 3.11+

---

## Setup

Export your Oracle schema to CSV, then index it:

```bash
tailwhip index --file data/db.csv
```

This will:
1. Parse the CSV (latin-1, semicolon-delimited)
2. Build a ChromaDB vector index in `~/.tailwhip/chroma/`
3. Build a BM25 lexical index in `~/.tailwhip/bm25.pkl`

Indexing ~30,000 rows takes around 2 minutes on first run (model download + embedding).
Subsequent runs are faster.

---

## Usage

```bash
# Basic search
tailwhip search "invoice number and freight value"

# Show columns for each result
tailwhip search "part number by process" --show-columns

# Filter by schema owner
tailwhip search "nota fiscal" --owners OWNER_NAME

# Multiple owners
tailwhip search "processo importacao" --owners OWNER_1,OWNER_2

# More results
tailwhip search "declaracao de importacao" --top 20

# Re-index after a new Oracle export
tailwhip refresh --file data/new_db.csv
```

---

## Synonyms

tailwhip ships with bidirectional synonym groups for the International Trade domain.
When you search for "SKU", it also searches for "part number" and vice versa.

The synonym file is at `tailwhip/synonyms.json` and is version-controlled.
Groups are conservative, only true alternative names, not loosely related concepts.

```json
{
  "sku": ["part number", "part_number"],
  "fatura": ["invoice", "num invoice", "numero fatura"],
  "di": ["declaracao importacao", "numero di", "num di"],
  "incoterm": ["terms", "terms of trade"]
}
```

---

## Running Tests

```bash
# All tests
python -m pytest -v

# Unit tests only (fast, no model loading)
python -m pytest tests/unit/ -v

# Integration tests (loads sentence-transformers model)
python -m pytest tests/integration/ -v
```

Current coverage: **77%** — 166 tests passing.

---

## Project Structure

```
tailwhip/
├── tailwhip/
│   ├── cli.py                  Entry point (Typer + Rich)
│   ├── synonyms.json           Bidirectional synonym groups
│   ├── models/                 Pydantic data models
│   ├── ingestion/              CSV parsing and normalization
│   ├── indexing/               ChromaDB + BM25 index management
│   └── search/                 RRF, boosting, orchestration, synonyms
└── tests/
    ├── unit/                   Fast tests with mocks (no model loading)
    └── integration/            Full pipeline tests with real CSV fixtures
```

---

## Tech Stack

| Component | Library |
|---|---|
| CLI | typer + rich |
| Vector search | chromadb (cosine similarity) |
| Lexical search | rank-bm25 (BM25Okapi) |
| Embeddings | sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`) |
| Data models | pydantic v2 |
| CSV parsing | pandas |
| Testing | pytest + pytest-cov |

---

## Limitations

- **Oracle only** — input must be an Oracle-exported CSV in the expected format (more databases coming in the future)
- **Read-only** — tailwhip never connects to the database
- **Batch indexing** — no incremental updates; use `refresh` after schema changes
- **No JOIN suggestions yet** — coming in a future release; suggest the JOIN path between tables for a given query