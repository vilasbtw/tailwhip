from pathlib import Path

from tailwhip.models.search_result import SearchResult
from tailwhip.models.schema_document import tokenize_name
from tailwhip.search.rank_fusion import reciprocal_rank_fusion, apply_fk_boost, apply_pk_boost, normalize_scores
from tailwhip.search.query_expander import QueryExpander

_DEFAULT_SYNONYMS = Path(__file__).parent.parent / "synonyms.json"


class SearchOrchestrator:
    """
    Facade that coordinates the full search pipeline:

    1. Synonym expansion  (QueryExpander)
    2. Vector search      (semantic, via ChromaDB)
    3. BM25 search        (lexical, exact token matching)
    4. Reciprocal Rank Fusion — merges both rankings
    5. FK boost           (double log — softened centrality boost)
    6. PK boost           — favours parent tables over child tables
    7. Score normalization — ensures final_score is in [0, 1]
    8. Owner filter       (optional)
    9. Return top-N SearchResult

    Receives IndexManager via dependency injection for testability.
    """

    def __init__(self, index_manager, expander: QueryExpander | None = None):
        self._index = index_manager
        self._expander = expander or QueryExpander(synonyms_path=_DEFAULT_SYNONYMS)

    def search(
        self,
        query: str,
        owners: list[str] | None = None,
        top_n: int = 10,
    ) -> list[SearchResult]:
        # 1. Synonym expansion
        query = self._expander.expand(query)

        # 2. Query both indexes
        vector_ranking = self._index.vector_search(query)
        bm25_ranking   = self._index.bm25_search(query)

        # 3. Merge rankings with RRF
        rrf_scores = reciprocal_rank_fusion([vector_ranking, bm25_ranking])

        if not rrf_scores:
            return []

        doc_map = self._index.get_documents()

        # 4. FK boost (double log — softened)
        fk_counts = {
            doc_id: doc_map[doc_id].table.fk_in_count
            for doc_id in rrf_scores
            if doc_id in doc_map
        }
        boosted = apply_fk_boost(rrf_scores, fk_counts)

        # 5. PK boost — favours parent tables over child tables
        query_tokens = tokenize_name(query).split()
        boosted = apply_pk_boost(boosted, doc_map, query_tokens)

        # 6. Normalize to [0, 1]
        normalized = normalize_scores(boosted)

        # 7. Sort by score descending
        ranked = sorted(normalized.items(), key=lambda x: x[1], reverse=True)

        # 8. Apply owner filter
        results = []
        for doc_id, score in ranked:
            doc = doc_map.get(doc_id)
            if doc is None:
                continue
            if owners and doc.table.schema_name not in owners:
                continue
            results.append((doc, score))

        # 9. Slice top-N and build SearchResult objects
        return [
            SearchResult(
                rank=i + 1,
                doc_id=doc.doc_id,
                table=doc.table,
                relevant_columns=doc.columns,
                final_score=score,
            )
            for i, (doc, score) in enumerate(results[:top_n])
        ]
