import math

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tailwhip.models.schema_document import SchemaDocument


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = 60
) -> dict[str, float]:
    """
    Combine multiple rankings using Reciprocal Rank Fusion.

    RRF(d) = Σ 1 / (k + rank_i(d))

    Args:
        rankings: list of rankings, each being an ordered list of doc_ids
        k: damping constant (default 60, classic value from the literature)

    Returns:
        dict mapping doc_id -> accumulated RRF score
    """
    scores: dict[str, float] = {}

    for ranking in rankings:
        for position, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + position)

    return scores


def apply_fk_boost(
    scores: dict[str, float], fk_counts: dict[str, int]
) -> dict[str, float]:
    """
    Apply FK centrality boost on top of RRF scores.

    Uses double log to soften the advantage of heavily referenced tables,
    preventing child tables (many inbound FKs) from outranking parent tables
    when the query targets the parent concept.

    score_final = rrf_score * log(1 + log(1 + fk_in_count))

    Args:
        scores: dict of doc_id -> rrf_score
        fk_counts: dict of doc_id -> fk_in_count (missing entries treated as 0)

    Returns:
        dict of doc_id -> boosted score
    """
    return {
        doc_id: score * math.log(1 + math.log(1 + fk_counts.get(doc_id, 0)))
        for doc_id, score in scores.items()
    }


def apply_pk_boost(
    scores: dict[str, float],
    documents: "dict[str, SchemaDocument]",
    query_tokens: list[str],
    weight: float = 0.5,
) -> dict[str, float]:
    """
    Boost tables whose PK columns match tokens in the query.

    Favours parent tables (concept is a PK) over child tables (concept is a FK).
    Classic case this resolves:
        INVOICE  (ID_INVOICE is PK) vs
        INVOICE_ITENS (ID_INVOICE is FK)

    score_final = score * (1 + weight * pk_match_ratio)
    pk_match_ratio = PK columns whose tokens intersect the query / total PKs

    Args:
        scores: dict of doc_id -> current score
        documents: dict of doc_id -> SchemaDocument
        query_tokens: tokenized query (already lowercased)
        weight: boost weight (default 0.5)

    Returns:
        dict of doc_id -> boosted score
    """
    if not scores:
        return {}

    # Generic schema tokens that alone do not identify a business concept
    STOPWORDS = {"id", "cod", "num", "numero", "codigo", "s", "n", "d", "c", "i"}

    query_set = set(query_tokens) - STOPWORDS

    # If the query contains only stopwords, skip the boost entirely
    if not query_set:
        return dict(scores)

    boosted = {}

    for doc_id, score in scores.items():
        doc = documents.get(doc_id)
        if doc is None:
            boosted[doc_id] = score
            continue

        pk_columns = [col for col in doc.columns if col.is_pk]

        if not pk_columns:
            boosted[doc_id] = score
            continue

        from tailwhip.models.schema_document import tokenize_name
        matched = sum(
            1 for col in pk_columns
            if (set(tokenize_name(col.column_name).split()) - STOPWORDS) & query_set
        )

        pk_match_ratio = matched / len(pk_columns)
        boosted[doc_id] = score * (1 + weight * pk_match_ratio)

    return boosted


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    """
    Normalize scores to the [0, 1] range using min-max scaling.

    Required to ensure SearchResult.final_score satisfies the model
    validation constraint (0 <= score <= 1).

    When all scores are equal, returns 0 for all entries to avoid
    division by zero.

    Args:
        scores: dict of doc_id -> score

    Returns:
        dict of doc_id -> normalized score in [0, 1]
    """
    if not scores:
        return {}

    min_score = min(scores.values())
    max_score = max(scores.values())
    spread = max_score - min_score

    if spread == 0:
        return {doc_id: 0.0 for doc_id in scores}

    return {
        doc_id: (score - min_score) / spread
        for doc_id, score in scores.items()
    }
