"""Types partagés par les méthodes de recherche."""

from dataclasses import asdict, dataclass
from typing import Protocol, runtime_checkable

from ledger_agent.ingestion.chunking import Chunk


@dataclass(frozen=True, slots=True)
class SearchFilters:
    doc_name: str | None = None
    ticker: str | None = None
    fiscal_year: int | None = None

    def as_dict(self) -> dict[str, str | int]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    results: list[RetrievedChunk]
    requested: SearchFilters | None
    applied: SearchFilters | None

    @property
    def relaxed(self) -> bool:
        """Vrai si les filtres demandés ont dû être élargis faute de résultats."""
        return self.requested != self.applied


class Retriever(Protocol):
    def search(
        self, query: str, k: int, filters: SearchFilters | None = None
    ) -> list[RetrievedChunk]: ...


@runtime_checkable
class ReportingRetriever(Protocol):
    """Recherche qui indique aussi les filtres réellement appliqués."""

    def search_with_outcome(self, query: str, k: int) -> SearchOutcome: ...
