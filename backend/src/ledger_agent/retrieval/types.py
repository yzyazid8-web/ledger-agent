"""Types partagés par les méthodes de recherche."""

from dataclasses import dataclass
from typing import Protocol

from ledger_agent.ingestion.chunking import Chunk


@dataclass(frozen=True, slots=True)
class SearchFilters:
    doc_name: str | None = None
    ticker: str | None = None
    fiscal_year: int | None = None


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int


class Retriever(Protocol):
    def search(
        self, query: str, k: int, filters: SearchFilters | None = None
    ) -> list[RetrievedChunk]: ...
