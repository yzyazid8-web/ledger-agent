"""Recherche hybride : fusion vectorielle + BM25 (RRF), reclassement, filtres en cascade."""

from collections.abc import Sequence

from ledger_agent.retrieval.fallback import search_with_fallback
from ledger_agent.retrieval.query_analysis import filters_from_question
from ledger_agent.retrieval.reranker import Reranker
from ledger_agent.retrieval.types import RetrievedChunk, Retriever, SearchFilters, SearchOutcome

# Constante usuelle de la Reciprocal Rank Fusion : elle atténue l'écart entre les premiers rangs
RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievedChunk]], weights: Sequence[float] | None = None
) -> list[RetrievedChunk]:
    """Fusionne plusieurs classements : chaque chunk reçoit la somme de poids / (RRF_K + rang)."""
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = {}
    chunks: dict[str, RetrievedChunk] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, item in enumerate(ranking):
            key = item.chunk.chunk_id
            scores[key] = scores.get(key, 0.0) + weight / (RRF_K + rank + 1)
            chunks.setdefault(key, item)
    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [
        RetrievedChunk(chunks[key].chunk, scores[key], rank) for rank, key in enumerate(ordered)
    ]


class HybridRetriever:
    def __init__(
        self,
        dense: Retriever,
        lexical: Retriever | None = None,
        reranker: Reranker | None = None,
        candidates: int = 30,
        lexical_weight: float = 1.0,
    ) -> None:
        self._dense = dense
        self._lexical = lexical
        self._reranker = reranker
        self._candidates = candidates
        self._lexical_weight = lexical_weight

    def search(
        self, query: str, k: int, filters: SearchFilters | None = None
    ) -> list[RetrievedChunk]:
        depth = max(k, self._candidates)
        rankings = [self._dense.search(query, depth, filters)]
        weights = [1.0]
        if self._lexical is not None:
            rankings.append(self._lexical.search(query, depth, filters))
            weights.append(self._lexical_weight)
        fused = reciprocal_rank_fusion(rankings, weights)[:depth]

        if self._reranker is None or not fused:
            return fused[:k]
        scores = self._reranker.score(query, [c.chunk.embedding_text for c in fused])
        reranked = sorted(zip(fused, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [
            RetrievedChunk(item.chunk, score, rank)
            for rank, (item, score) in enumerate(reranked[:k])
        ]


class AutoFilterRetriever:
    """Applique les filtres repérés dans la question, en élargissant si les résultats manquent."""

    def __init__(self, inner: Retriever) -> None:
        self._inner = inner

    def search_with_outcome(self, query: str, k: int) -> SearchOutcome:
        return search_with_fallback(self._inner, query, k, filters_from_question(query))

    def search(
        self, query: str, k: int, filters: SearchFilters | None = None
    ) -> list[RetrievedChunk]:
        if filters is not None:
            return search_with_fallback(self._inner, query, k, filters).results
        return self.search_with_outcome(query, k).results
