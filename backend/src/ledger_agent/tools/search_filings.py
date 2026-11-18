"""Outil de recherche de passages dans les rapports, qui dit quels filtres il a appliqués."""

from dataclasses import dataclass

from ledger_agent.retrieval.fallback import search_with_fallback
from ledger_agent.retrieval.types import RetrievedChunk, Retriever, SearchFilters


@dataclass(frozen=True, slots=True)
class Passage:
    source_id: str
    doc_name: str
    page_number: int
    header: str
    text: str
    score: float


@dataclass(frozen=True, slots=True)
class SearchFilingsResult:
    passages: list[Passage]
    requested_filters: dict[str, str | int]
    applied_filters: dict[str, str | int]

    @property
    def relaxed(self) -> bool:
        """Vrai si les passages ne respectent pas tous les filtres demandés."""
        return self.requested_filters != self.applied_filters


def to_passage(item: RetrievedChunk) -> Passage:
    chunk = item.chunk
    return Passage(
        # Numéro de page affiché à partir de 1, comme dans un lecteur PDF
        source_id=f"doc:{chunk.doc_name}:p{chunk.page_index + 1}",
        doc_name=chunk.doc_name,
        page_number=chunk.page_index + 1,
        header=chunk.header,
        text=chunk.text,
        score=item.score,
    )


class SearchFilingsTool:
    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever

    def run(
        self,
        query: str,
        ticker: str | None = None,
        fiscal_year: int | None = None,
        doc_name: str | None = None,
        k: int = 5,
    ) -> SearchFilingsResult:
        """Cherche avec les filtres donnés, les relâche si besoin, et le signale."""
        requested = (
            SearchFilters(
                doc_name=doc_name,
                ticker=ticker.upper() if ticker else None,
                fiscal_year=fiscal_year,
            )
            if doc_name or ticker or fiscal_year
            else None
        )
        outcome = search_with_fallback(self._retriever, query, k, requested)
        return SearchFilingsResult(
            passages=[to_passage(r) for r in outcome.results],
            requested_filters=requested.as_dict() if requested else {},
            applied_filters=outcome.applied.as_dict() if outcome.applied else {},
        )
