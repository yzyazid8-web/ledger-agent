"""Construction des index de recherche : vectoriel (Qdrant) et lexical (BM25)."""

import json
import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Protocol

import bm25s
import Stemmer
from qdrant_client import QdrantClient, models

from ledger_agent.ingestion.chunking import Chunk

logger = logging.getLogger(__name__)

COLLECTION_NAME = "filings"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
BM25_IDS_FILE = "chunk_ids.json"


class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    """Embeddings ONNX sur CPU, sans PyTorch."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        cache_dir: Path | None = None,
        batch_size: int = 64,
    ) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name, cache_dir=str(cache_dir) if cache_dir else None)
        self._batch_size = batch_size
        self._dimension = len(self.embed_query("dimension probe"))

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.passage_embed(list(texts), batch_size=self._batch_size)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        # bge ajoute un préfixe d'instruction aux requêtes, que query_embed gère
        vector = next(iter(self._model.query_embed(text)))
        return [float(x) for x in vector]


def chunk_payload(chunk: Chunk) -> dict[str, object]:
    return {
        "doc_name": chunk.doc_name,
        "company": chunk.company,
        "ticker": chunk.ticker,
        "doc_type": chunk.doc_type,
        "fiscal_year": chunk.fiscal_year,
        "fiscal_quarter": chunk.fiscal_quarter,
        "page_index": chunk.page_index,
    }


def ensure_collection(client: QdrantClient, dimension: int, recreate: bool = False) -> None:
    if recreate and client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            COLLECTION_NAME,
            vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
        )


def indexed_ids(client: QdrantClient) -> set[str]:
    ids: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            COLLECTION_NAME, limit=1000, offset=offset, with_payload=False, with_vectors=False
        )
        ids.update(str(point.id) for point in points)
        if offset is None:
            return ids


def _batches(items: Sequence[Chunk], size: int) -> Iterator[Sequence[Chunk]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def sync_dense_index(
    client: QdrantClient,
    chunks: Sequence[Chunk],
    embedder: Embedder,
    batch_size: int = 256,
) -> tuple[int, int]:
    """Aligne la collection sur `chunks`. Renvoie (chunks ajoutés, chunks supprimés)."""
    existing = indexed_ids(client)
    wanted = {c.chunk_id for c in chunks}

    stale = sorted(existing - wanted)
    if stale:
        client.delete(COLLECTION_NAME, points_selector=models.PointIdsList(points=list(stale)))

    missing = [c for c in chunks if c.chunk_id not in existing]
    for done, batch in enumerate(_batches(missing, batch_size), start=1):
        vectors = embedder.embed_documents([c.embedding_text for c in batch])
        client.upsert(
            COLLECTION_NAME,
            points=[
                models.PointStruct(id=c.chunk_id, vector=v, payload=chunk_payload(c))
                for c, v in zip(batch, vectors, strict=True)
            ],
        )
        logger.info(
            "Embedded %d / %d new chunks", min(done * batch_size, len(missing)), len(missing)
        )
    return len(missing), len(stale)


def tokenize(texts: Sequence[str]) -> list[list[str]]:
    stemmer = Stemmer.Stemmer("english")
    tokens: list[list[str]] = bm25s.tokenize(
        list(texts), stopwords="en", stemmer=stemmer, return_ids=False, show_progress=False
    )
    return tokens


def build_bm25_index(chunks: Sequence[Chunk], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    retriever = bm25s.BM25()
    retriever.index(tokenize([c.embedding_text for c in chunks]), show_progress=False)
    retriever.save(str(out_dir), show_progress=False)
    # BM25 renvoie des positions : ce fichier les traduit en identifiants de chunks
    (out_dir / BM25_IDS_FILE).write_text(json.dumps([c.chunk_id for c in chunks]), encoding="utf-8")
