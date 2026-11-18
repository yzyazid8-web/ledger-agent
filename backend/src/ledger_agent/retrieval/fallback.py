"""Relâchement progressif des filtres, partagé par l'évaluation et l'outil de l'agent."""

from ledger_agent.retrieval.types import Retriever, SearchFilters, SearchOutcome


def relaxation_cascade(filters: SearchFilters | None) -> list[SearchFilters | None]:
    """Filtres à essayer dans l'ordre : tels quels, entreprise seule, aucun."""
    cascade: list[SearchFilters | None] = [filters]
    if filters is not None:
        if filters.ticker and (filters.doc_name or filters.fiscal_year):
            cascade.append(SearchFilters(ticker=filters.ticker))
        cascade.append(None)
    return list(dict.fromkeys(cascade))


def search_with_fallback(
    retriever: Retriever, query: str, k: int, filters: SearchFilters | None
) -> SearchOutcome:
    """Cherche avec `filters`, puis élargit tant qu'il manque des résultats."""
    results = []
    applied: SearchFilters | None = filters
    for candidate in relaxation_cascade(filters):
        applied = candidate
        results = retriever.search(query, k, candidate)
        if len(results) >= k:
            break
    return SearchOutcome(results=results, requested=filters, applied=applied)
