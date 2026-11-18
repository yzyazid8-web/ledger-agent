"""Outil de recherche de passages dans les rapports, avec filtres explicites."""

from dataclasses import dataclass

from ledger_agent.retrieval.types import RetrievedChunk, Retriever, SearchFilters


@dataclass(frozen=True, slots=True)
class Passage:
    source_id: str
    doc_name: str
    page_number: int
    header: str
    text: str
    score: float


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
    ) -> list[Passage]:
        """Cherche avec les filtres donnés, puis les relâche un à un si les résultats manquent."""
        cascade: list[SearchFilters | None] = []
        if doc_name or ticker or fiscal_year:
            cascade.append(SearchFilters(doc_name=doc_name, ticker=ticker, fiscal_year=fiscal_year))
        if ticker and (doc_name or fiscal_year):
            cascade.append(SearchFilters(ticker=ticker))
        cascade.append(None)

        results: list[RetrievedChunk] = []
        for filters in cascade:
            results = self._retriever.search(query, k, filters)
            if len(results) >= k:
                break
        return [to_passage(r) for r in results]
