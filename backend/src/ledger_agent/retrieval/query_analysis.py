"""Repérage par règles de l'entreprise et de l'exercice cités dans une question."""

import re
from dataclasses import dataclass
from functools import cache

from ledger_agent.ingestion.universe import Company, load_universe
from ledger_agent.retrieval.fallback import relaxation_cascade
from ledger_agent.retrieval.types import SearchFilters

# « FY2022 », « fiscal year 2022 », « 2022 », « FY2023Q1 » ; « FY22 » est traité à part.
# Pas de \b final : dans « 2023Q1 », rien ne sépare l'année du trimestre
_FULL_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_SHORT_YEAR = re.compile(r"\bFY\s?'?(\d{2})\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class QueryEntities:
    tickers: tuple[str, ...]
    years: tuple[int, ...]


@cache
def _company_patterns() -> tuple[tuple[Company, re.Pattern[str]], ...]:
    patterns = []
    for company in load_universe():
        names = sorted({company.name, *company.aliases}, key=len, reverse=True)
        alternatives = "|".join(re.escape(n) for n in names)
        # Le ticker n'est reconnu qu'en majuscules, pour ne pas confondre « BA » et « ba »
        pattern = re.compile(
            rf"(?<![\w&])(?:(?i:{alternatives})|{re.escape(company.ticker)})(?![\w&])"
        )
        patterns.append((company, pattern))
    return tuple(patterns)


def extract_entities(question: str) -> QueryEntities:
    tickers = tuple(c.ticker for c, pattern in _company_patterns() if pattern.search(question))
    years = {int(y) for y in _FULL_YEAR.findall(question)}
    years |= {2000 + int(y) for y in _SHORT_YEAR.findall(question)}
    return QueryEntities(tickers=tickers, years=tuple(sorted(years)))


def filters_from_question(question: str) -> SearchFilters | None:
    """Entreprise et exercice cités, ou None si l'entreprise est absente ou ambiguë."""
    entities = extract_entities(question)
    if len(entities.tickers) != 1:
        return None
    # Plusieurs années (« FY2021 vs FY2022 ») : le rapport le plus récent contient les comparatifs
    year = max(entities.years) if entities.years else None
    return SearchFilters(ticker=entities.tickers[0], fiscal_year=year)


def filter_cascade(question: str) -> list[SearchFilters | None]:
    """Filtres à essayer dans l'ordre : entreprise + exercice, entreprise seule, aucun."""
    return relaxation_cascade(filters_from_question(question))
