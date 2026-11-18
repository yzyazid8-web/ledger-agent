"""Recherche vectorielle dans Qdrant, avec filtres de métadonnées optionnels."""

from qdrant_client import QdrantClient, models

from ledger_agent.ingestion.indexer import COLLECTION_NAME, Embedder
from ledger_agent.retrieval.store import ChunkStore
from ledger_agent.retrieval.types import RetrievedChunk, SearchFilters


def to_qdrant_filter(filters: SearchFilters | None) -> models.Filter | None:
    if filters is None:
        return None
    conditions: list[models.Condition] = [
        models.FieldCondition(key=key, match=models.MatchValue(value=value))
        for key, value in (
            ("doc_name", filters.doc_name),
            ("ticker", filters.ticker),
            ("fiscal_year", filters.fiscal_year),
        )
        if value is not None
    ]
    return models.Filter(must=conditions) if conditions else None


class DenseRetriever:
    def __init__(self, client: QdrantClient, embedder: Embedder, store: ChunkStore) -> None:
        self._client = client
        self._embedder = embedder
        self._store = store

    def search(
        self, query: str, k: int, filters: SearchFilters | None = None
    ) -> list[RetrievedChunk]:
        points = self._client.query_points(
            COLLECTION_NAME,
            query=self._embedder.embed_query(query),
            query_filter=to_qdrant_filter(filters),
            limit=k,
            with_payload=False,
        ).points
        return [
            RetrievedChunk(chunk=self._store.get(str(p.id)), score=float(p.score), rank=rank)
            for rank, p in enumerate(points)
            if str(p.id) in self._store
        ]
