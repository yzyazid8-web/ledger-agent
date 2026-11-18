import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import bm25s
from qdrant_client import QdrantClient

from ledger_agent.ingestion.chunking import Chunk, chunk_document
from ledger_agent.ingestion.financebench import FinanceBenchDocument
from ledger_agent.ingestion.indexer import (
    BM25_IDS_FILE,
    COLLECTION_NAME,
    build_bm25_index,
    ensure_collection,
    indexed_ids,
    sync_dense_index,
    tokenize,
)
from ledger_agent.ingestion.parsing import PageText


class FakeEmbedder:
    """Vecteurs déterministes dérivés d'un hash : pas de modèle à télécharger."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def dimension(self) -> int:
        return 8

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += len(texts)
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [b / 255 + 0.01 for b in digest[:8]]


def make_chunks() -> list[Chunk]:
    document = FinanceBenchDocument("3M_2018_10K", "3M", "10k", 2018)
    pages = [
        PageText(59, "Consolidated Statement of Cash Flows. Purchases of property plant equipment"),
        PageText(10, "Risk factors: the company faces litigation related to PFAS chemicals"),
        PageText(20, "Revenue by segment: Safety and Industrial, Health Care, Consumer"),
    ]
    return chunk_document(document, "MMM", pages)


def test_dense_sync_is_incremental_and_removes_stale_chunks() -> None:
    client = QdrantClient(":memory:")
    embedder = FakeEmbedder()
    chunks = make_chunks()
    ensure_collection(client, embedder.dimension)

    assert sync_dense_index(client, chunks, embedder) == (3, 0)
    # Deuxième passage : rien à vectoriser
    assert sync_dense_index(client, chunks, embedder) == (0, 0)
    assert embedder.calls == 3

    # Un chunk disparaît de la source : il est supprimé de l'index
    assert sync_dense_index(client, chunks[:2], embedder) == (0, 1)
    assert indexed_ids(client) == {c.chunk_id for c in chunks[:2]}
    assert client.count(COLLECTION_NAME).count == 2


def test_bm25_index_finds_the_right_chunk(tmp_path: Path) -> None:
    chunks = make_chunks()
    build_bm25_index(chunks, tmp_path)

    retriever = bm25s.BM25.load(str(tmp_path))
    ids = json.loads((tmp_path / BM25_IDS_FILE).read_text(encoding="utf-8"))
    positions, _ = retriever.retrieve(tokenize(["PFAS litigation"]), k=1, show_progress=False)

    assert ids[positions[0][0]] == chunks[1].chunk_id
