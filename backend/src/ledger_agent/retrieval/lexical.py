"""Recherche lexicale BM25, avec les mêmes filtres de métadonnées que la recherche vectorielle."""

import json
from pathlib import Path

import bm25s
import numpy as np

from ledger_agent.ingestion.indexer import BM25_IDS_FILE, tokenize
from ledger_agent.retrieval.store import ChunkStore
from ledger_agent.retrieval.types import RetrievedChunk, SearchFilters


class LexicalRetriever:
    def __init__(self, index: bm25s.BM25, chunk_ids: list[str], store: ChunkStore) -> None:
        self._index = index
        self._chunk_ids = chunk_ids
        self._store = store
        # Métadonnées alignées sur les positions de l'index, pour filtrer sans boucle Python
        chunks = [store.get(i) for i in chunk_ids]
        self._doc_names = np.array([c.doc_name for c in chunks])
        self._tickers = np.array([c.ticker or "" for c in chunks])
        self._years = np.array([c.fiscal_year for c in chunks])

    @classmethod
    def load(cls, index_dir: Path, store: ChunkStore) -> "LexicalRetriever":
        index = bm25s.BM25.load(str(index_dir))
        chunk_ids = json.loads((index_dir / BM25_IDS_FILE).read_text(encoding="utf-8"))
        return cls(index, chunk_ids, store)

    def _mask(self, filters: SearchFilters | None) -> np.ndarray | None:
        if filters is None:
            return None
        mask = np.ones(len(self._chunk_ids), dtype=bool)
        if filters.doc_name is not None:
            mask &= self._doc_names == filters.doc_name
        if filters.ticker is not None:
            mask &= self._tickers == filters.ticker
        if filters.fiscal_year is not None:
            mask &= self._years == filters.fiscal_year
        return mask

    def search(
        self, query: str, k: int, filters: SearchFilters | None = None
    ) -> list[RetrievedChunk]:
        mask = self._mask(filters)
        if mask is not None and not mask.any():
            return []
        allowed = int(mask.sum()) if mask is not None else len(self._chunk_ids)
        positions, scores = self._index.retrieve(
            tokenize([query]),
            k=min(k, allowed),
            weight_mask=mask.astype(float) if mask is not None else None,
            show_progress=False,
        )
        results: list[RetrievedChunk] = []
        for position, score in zip(positions[0], scores[0], strict=True):
            # Un score nul signifie qu'aucun mot de la requête n'apparaît dans le chunk
            if score <= 0 or (mask is not None and not mask[position]):
                continue
            chunk = self._store.get(self._chunk_ids[int(position)])
            results.append(RetrievedChunk(chunk=chunk, score=float(score), rank=len(results)))
        return results
