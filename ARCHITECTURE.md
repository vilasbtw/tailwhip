# Architecture

tailwhip is a hybrid semantic search CLI for Oracle schema metadata.
This document explains how the pipeline works and why each component was designed the way it was.

---

## The Problem

Oracle databases used in enterprise systems can have thousands of tables.
Finding the right table for a given business concept, "invoices", "users", "broker contracts", requires either deep schema knowledge or hours of manual searching.

A pure keyword search fails because table names are abbreviated and inconsistent.
A pure vector search fails because it misses exact token matches and treats all tables as equally relevant regardless of their position in the data model.

tailwhip solves this by combining both approaches and layering domain-aware ranking on top.

---

## Pipeline Overview

```
User query
    │
    ▼
┌─────────────────┐
│  QueryExpander  │  Synonym expansion (e.g. "invoice" → "invoice nota fiscal fatura")
└────────┬────────┘
         │ expanded query
         ├─────────────────────────────────┐
         ▼                                 ▼
┌─────────────────┐             ┌─────────────────┐
│  Vector Search  │             │   BM25 Search   │
│  (ChromaDB)     │             │   (rank-bm25)   │
│  semantic       │             │   lexical        │
└────────┬────────┘             └────────┬────────┘
         │ ranked doc_ids                │ ranked doc_ids
         └──────────────┬────────────────┘
                        ▼
             ┌─────────────────┐
             │      RRF        │  Reciprocal Rank Fusion
             │  1/(k + rank)   │  merges both rankings
             └────────┬────────┘
                      ▼
             ┌─────────────────┐
             │    FK Boost     │  score × log(1 + log(1 + fk_in_count))
             │  (double log)   │  rewards central tables
             └────────┬────────┘
                      ▼
             ┌─────────────────┐
             │    PK Boost     │  score × (1 + 0.5 × pk_match_ratio)
             │                 │  rewards parent tables over child tables
             └────────┬────────┘
                      ▼
             ┌─────────────────┐
             │   Normalize     │  min-max scaling → [0, 1]
             └────────┬────────┘
                      ▼
             ┌─────────────────┐
             │  Owner Filter   │  optional schema filter
             └────────┬────────┘
                      ▼
                 Top-N results
```

---

## Components

### Ingestion

**`OracleAdapter`** reads the Oracle-exported CSV (`sep=';'`, `encoding='latin-1'`) and returns a flat list of `ColumnMetadata`, one entry per column. It parses `CONSTRAINT_TYPE` to detect PKs and FKs, extracting the referenced schema and table from `FK -> OWNER.TABLE_NAME` strings.

**`SchemaNormalizer`** groups columns by `(schema, table)` and computes `fk_in_count`, the number of inbound foreign keys from other tables. Self-references are excluded. This count is the foundation of the FK boost.

### Indexing

**`DocumentBuilder`** converts `(TableMetadata, [ColumnMetadata])` pairs into `SchemaDocument` instances. Each document gets a `doc_id` in the format `SCHEMA.TABLE_NAME` and exposes a `to_embedding_text()` method that assembles a rich text representation:

```
app usuario | user registry
columns: id usuario NUMBER [PK] | nome VARCHAR2(100) | ...
referenced by 5 tables
```

The embedding text is designed to be informative for both the vector model and BM25 tokenization.

**`IndexManager`** manages two indexes persisted under `~/.tailwhip/`:
- **ChromaDB** stores vector embeddings using `paraphrase-multilingual-MiniLM-L12-v2`, a multilingual sentence-transformers model that handles both PT-BR and EN queries.
- **BM25Okapi** tokenizes the embedding text and builds a classic BM25 index for exact token matching.

Both indexes are built from the same `to_embedding_text()` output, ensuring consistency.

### Search

**`QueryExpander`** loads `synonyms.json` and expands the query before it hits either index. This allows "nota fiscal" to match tables indexed with "invoice" and vice versa.

**`SearchOrchestrator`** coordinates the full pipeline. It receives an `IndexManager` via dependency injection, making it fully testable with mocks.

---

## Ranking Design Decisions

### Why hybrid search (vector + BM25)?

Vector search excels at semantic similarity but struggles with exact names and abbreviations.
BM25 excels at exact token matching but misses synonyms and semantic intent.
Combining both via RRF consistently outperforms either alone, especially in schema search, where table names are both abbreviated identifiers and business concepts.

### Why Reciprocal Rank Fusion?

RRF (`1 / (k + rank)`) is simple, parameter-light, and robust. The damping constant `k=60` (the classic value from the original paper) prevents early positions from dominating and gives meaningful weight to documents that appear consistently across both rankings.

An alternative would be weighted score fusion, but that requires calibrating weights per query type. RRF works well out of the box.

### Why FK boost with double log?

Tables with many inbound foreign keys are typically central business entities, the tables you're most likely looking for. `fk_in_count` is a structural signal of importance.

The naive approach would be a linear or single-log boost, but that creates too large a gap between a highly referenced table (e.g. 100 refs) and a moderately referenced one (e.g. 5 refs), making it hard for other signals to compete.

Double log softens this:

```
boost = log(1 + log(1 + fk_in_count))

fk=1   → 0.53
fk=5   → 0.90
fk=20  → 1.14
fk=100 → 1.37
```

The differences are meaningful but not overwhelming. A table with 100 refs doesn't score 20× a table with 5 refs, it scores about 1.5×.

**Note:** `fk_in_count=0` results in `log(1) = 0`, zeroing the score. This is intentional: tables with no inbound FKs are likely lookup tables or leaf nodes, not the business entities a user is searching for.

### Why PK boost?

The FK boost alone creates a problem: child tables (many columns, many inbound FKs from their own sub-tables) can outscore parent tables when the query matches the parent concept.

Classic example: a query for "invoice" might rank `ORDER_ITEMS` (which has `ID_INVOICE` as a FK and 20 inbound refs) above `INVOICES` (which has `ID_INVOICE` as a PK and 5 inbound refs).

The PK boost addresses this by rewarding tables where query tokens match PK column names:

```
score × (1 + 0.5 × pk_match_ratio)
```

`pk_match_ratio` is the fraction of PK columns whose tokenized name intersects the query (after removing stopwords like `id`, `cod`, `num`). A table where the searched concept is the PK gets boosted; a table where it's just a FK does not.

### Why normalize after all boosts?

`SearchResult.final_score` must be in `[0, 1]` to satisfy the model validation constraint. Min-max normalization is applied last, after all boosts have been combined, ensuring the scale constraint is met without affecting the relative ordering.

---

## Module Structure

```
tailwhip/
├── models/              # Pure data classes (no business logic)
│   ├── column_metadata.py
│   ├── table_metadata.py
│   ├── schema_document.py
│   └── search_result.py
├── ingestion/           # CSV parsing and schema normalization
│   ├── oracle_adapter.py
│   └── schema_normalizer.py
├── indexing/            # Document building and index management
│   ├── document_builder.py
│   └── index_manager.py
├── search/              # Query expansion, ranking, orchestration
│   ├── query_expander.py
│   ├── rank_fusion.py
│   └── orchestrator.py
├── synonyms.json        # 36 synonym groups (bidirectional)
└── cli.py               # Typer CLI (index, refresh, search commands)

tests/
├── unit/                # Fast, isolated, mock-heavy
├── integration/         # Full pipeline with real ChromaDB and BM25
└── fixtures/            # oracle_sample.csv, fictional schema for tests
```

### Design principles

**Dependency injection throughout.** `SearchOrchestrator` receives `IndexManager` as a constructor argument. This makes the entire search pipeline testable without touching disk or loading ML models.

**Pure functions for ranking.** `rank_fusion.py` contains only pure functions, no state, no I/O. Every ranking decision is independently testable with simple dict inputs.

**Models validate their own invariants.** `SearchResult` enforces `0 <= final_score <= 1` at construction time. Invalid scores fail fast rather than silently corrupting results.

**TDD from the start.** Each module was written test-first. The test suite documents the intended behaviour of every branch, including defensive edge cases.

---

## Storage

All index data is persisted under `~/.tailwhip/`:

```
~/.tailwhip/
├── chroma/          # ChromaDB vector store (HNSW, cosine similarity)
├── bm25.pkl         # Serialized BM25Okapi instance
└── documents.pkl    # Dict[doc_id, SchemaDocument]
```

`IndexManager` loads lazily, indexes are only read from disk on the first search call, not at construction time.

---

## Performance Notes

- **Indexing** is the slow step: embedding 2,000+ tables takes 1–3 minutes on CPU, depending on hardware. It runs once and is persisted.
- **Search** is fast: vector query + BM25 scoring + all ranking steps complete in under 1 second after the index is loaded into memory.
- **Embeddings** are batched in groups of 256 to avoid memory overflow on large schemas.
- The sentence-transformers model (`~90MB`) is loaded once per `IndexManager` instance and reused across searches.