"""Entreprises couvertes par le projet (l'univers FinanceBench)."""

import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from importlib.resources import files


@dataclass(frozen=True, slots=True)
class Company:
    name: str
    ticker: str
    cik: int
    # Autres façons de nommer l'entreprise dans une question (« J&J », « Ulta »…)
    aliases: tuple[str, ...] = ()

    @property
    def cik10(self) -> str:
        """CIK complété à 10 chiffres par des zéros, comme dans les URL EDGAR."""
        return f"{self.cik:010d}"


@cache
def load_universe() -> tuple[Company, ...]:
    raw = (files("ledger_agent.ingestion") / "universe.toml").read_text(encoding="utf-8")
    entries = tomllib.loads(raw)["companies"]
    return tuple(
        Company(
            name=e["name"],
            ticker=e["ticker"],
            cik=int(e["cik"]),
            aliases=tuple(e.get("aliases", ())),
        )
        for e in entries
    )


def find_company(name: str) -> Company | None:
    """Retrouve une entreprise par son nom FinanceBench, ou None si elle est hors univers."""
    return next((c for c in load_universe() if c.name == name), None)


def select_companies(tickers: Iterable[str] | None = None) -> tuple[Company, ...]:
    """Renvoie tout l'univers, ou seulement les tickers demandés (insensible à la casse)."""
    universe = load_universe()
    if not tickers:
        return universe
    wanted = {t.upper() for t in tickers}
    unknown = wanted - {c.ticker for c in universe}
    if unknown:
        raise ValueError(f"Unknown tickers: {sorted(unknown)}")
    return tuple(c for c in universe if c.ticker in wanted)
