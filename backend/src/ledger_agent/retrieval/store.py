"""Accès au texte des chunks à partir de leur identifiant (lu depuis chunks.parquet)."""

from collections.abc import Iterable
from pathlib import Path

import pyarrow.parquet as pq

from ledger_agent.ingestion.chunking import Chunk, table_to_chunks


class ChunkStore:
    def __init__(self, chunks: Iterable[Chunk]) -> None:
        self._by_id = {c.chunk_id: c for c in chunks}

    @classmethod
    def from_parquet(cls, path: Path) -> "ChunkStore":
        return cls(table_to_chunks(pq.read_table(path)))

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, chunk_id: object) -> bool:
        return chunk_id in self._by_id

    def get(self, chunk_id: str) -> Chunk:
        return self._by_id[chunk_id]
