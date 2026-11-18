"""Accès typé aux chiffres XBRL : recherche de concepts et lecture d'un fait précis."""

import re
from dataclasses import dataclass
from pathlib import Path

import duckdb

# Libellés courants vers les concepts us-gaap qui les portent, par ordre de préférence.
# Une même notion est souvent déclarée sous plusieurs concepts selon l'entreprise.
_REVENUE = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
)
_COGS = ("CostOfGoodsAndServicesSold", "CostOfGoodsSold", "CostOfRevenue")
_CAPEX = ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets")
_DA = (
    "DepreciationDepletionAndAmortization",
    "DepreciationAndAmortization",
    "DepreciationAmortizationAndAccretionNet",
)
METRIC_SYNONYMS: dict[str, tuple[str, ...]] = {
    "revenue": _REVENUE,
    "net sales": _REVENUE,
    "sales": _REVENUE,
    "cost of goods sold": _COGS,
    "cogs": _COGS,
    "cost of revenue": _COGS,
    "cost of sales": _COGS,
    "gross profit": ("GrossProfit",),
    "operating income": ("OperatingIncomeLoss",),
    "net income": ("NetIncomeLoss", "ProfitLoss"),
    "diluted eps": ("EarningsPerShareDiluted",),
    "basic eps": ("EarningsPerShareBasic",),
    "eps": ("EarningsPerShareDiluted", "EarningsPerShareBasic"),
    "capital expenditure": _CAPEX,
    "capex": _CAPEX,
    "depreciation and amortization": _DA,
    "d&a": _DA,
    "operating cash flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "cash from operations": ("NetCashProvidedByUsedInOperatingActivities",),
    "total assets": ("Assets",),
    "total liabilities": ("Liabilities",),
    "current assets": ("AssetsCurrent",),
    "current liabilities": ("LiabilitiesCurrent",),
    "inventory": ("InventoryNet", "InventoryFinishedGoods"),
    "accounts receivable": ("AccountsReceivableNetCurrent",),
    "accounts payable": ("AccountsPayableCurrent",),
    "cash and cash equivalents": ("CashAndCashEquivalentsAtCarryingValue",),
    "shareholders equity": ("StockholdersEquity",),
    "total equity": (
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "StockholdersEquity",
    ),
    "long-term debt": ("LongTermDebtNoncurrent", "LongTermDebt"),
    "interest expense": ("InterestExpense", "InterestExpenseNonoperating"),
    "income tax": ("IncomeTaxExpenseBenefit",),
    "research and development": ("ResearchAndDevelopmentExpense",),
    "sg&a": ("SellingGeneralAndAdministrativeExpense",),
    "dividends paid": ("PaymentsOfDividends", "PaymentsOfDividendsCommonStock"),
    "dividends per share": ("CommonStockDividendsPerShareDeclared",),
    "share repurchases": ("PaymentsForRepurchaseOfCommonStock",),
}

_TOKEN = re.compile(r"[a-z]{3,}")
_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])")

# Types de période acceptés pour un exercice ou un trimestre
_PERIOD_TYPES = {"FY": ("annual", "instant"), "Q4": ("quarter",)}
_QUARTER_TYPES = ("quarter", "instant")
VALID_PERIODS = ("FY", "Q1", "Q2", "Q3", "Q4")


@dataclass(frozen=True, slots=True)
class ConceptMatch:
    taxonomy: str
    concept: str
    label: str
    facts: int
    score: float


@dataclass(frozen=True, slots=True)
class FactResult:
    ticker: str
    taxonomy: str
    concept: str
    value: float
    unit: str
    fiscal_year: int
    fiscal_period: str
    period_start: str | None
    period_end: str
    form: str
    accn: str
    filed: str
    # Valeur la plus récente si le chiffre a été retraité dans un dépôt ultérieur
    latest_value: float | None = None

    @property
    def source_id(self) -> str:
        return f"xbrl:{self.ticker}:{self.concept}:{self.fiscal_year}{self.fiscal_period}"


def _words(text: str) -> set[str]:
    return set(_TOKEN.findall(_CAMEL.sub(" ", text).lower()))


class FactStore:
    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self._con = connection
        rows = self._con.execute(
            """
            SELECT c.taxonomy, c.concept, coalesce(c.label, c.concept), f.ticker, count(*)
            FROM concepts c JOIN facts f USING (taxonomy, concept)
            GROUP BY ALL
            """
        ).fetchall()
        self._labels: dict[tuple[str, str], str] = {}
        self._counts: dict[tuple[str, str, str], int] = {}
        for taxonomy, concept, label, ticker, count in rows:
            self._labels[(taxonomy, concept)] = label
            self._counts[(ticker, taxonomy, concept)] = count

    @classmethod
    def open(cls, path: Path) -> "FactStore":
        return cls(duckdb.connect(str(path), read_only=True))

    def close(self) -> None:
        self._con.close()

    def _facts_for(self, ticker: str | None, taxonomy: str, concept: str) -> int:
        if ticker is None:
            return sum(
                n for (_, tx, c), n in self._counts.items() if (tx, c) == (taxonomy, concept)
            )
        return self._counts.get((ticker, taxonomy, concept), 0)

    def search_concepts(
        self, query: str, ticker: str | None = None, limit: int = 8
    ) -> list[ConceptMatch]:
        """Concepts XBRL correspondant à un libellé, limités à ceux déclarés par l'entreprise."""
        lowered = query.lower()
        scores: dict[tuple[str, str], float] = {}

        # 1. Synonymes connus, prioritaires et dans leur ordre de préférence
        for phrase, concepts in METRIC_SYNONYMS.items():
            if re.search(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", lowered):
                for rank, concept in enumerate(concepts):
                    key = ("us-gaap", concept)
                    scores[key] = max(scores.get(key, 0.0), 10.0 - rank + len(phrase) / 100)

        # 2. Recouvrement de mots entre la requête et le libellé ou le nom du concept
        query_words = _words(query)
        if query_words:
            for key, label in self._labels.items():
                overlap = len(query_words & (_words(label) | _words(key[1])))
                if overlap:
                    scores[key] = max(scores.get(key, 0.0), overlap / len(query_words))

        matches = [
            ConceptMatch(tx, c, self._labels.get((tx, c), c), self._facts_for(ticker, tx, c), s)
            for (tx, c), s in scores.items()
        ]
        matches = [m for m in matches if m.facts > 0]
        # À score égal, le concept le plus renseigné passe devant
        matches.sort(key=lambda m: (m.score, m.facts), reverse=True)
        return matches[:limit]

    def get_fact(
        self,
        ticker: str,
        concept: str,
        fiscal_year: int,
        fiscal_period: str = "FY",
        taxonomy: str = "us-gaap",
    ) -> FactResult | None:
        """Le chiffre tel que publié à l'origine pour cet exercice, ou None s'il n'existe pas."""
        ticker, fiscal_period = ticker.upper(), fiscal_period.upper()
        if fiscal_period not in VALID_PERIODS:
            raise ValueError(f"Période inconnue : {fiscal_period} (attendu : {VALID_PERIODS})")
        period_types = _PERIOD_TYPES.get(fiscal_period, _QUARTER_TYPES)
        placeholders = ", ".join("?" for _ in period_types)
        rows = self._con.execute(
            f"""
            SELECT value, unit, period_start, period_end, form, accn, filed,
                   is_current_period, is_latest
            FROM facts
            WHERE ticker = ? AND taxonomy = ? AND concept = ?
              AND fiscal_year = ? AND fiscal_period = ?
              AND period_type IN ({placeholders})
            ORDER BY is_current_period DESC, filed ASC, period_type
            """,
            [ticker, taxonomy, concept, fiscal_year, fiscal_period, *period_types],
        ).fetchall()
        if not rows:
            return None

        value, unit, start, end, form, accn, filed, _, _ = rows[0]
        latest = next((r[0] for r in rows if r[8] and r[3] == end), None)
        return FactResult(
            ticker=ticker,
            taxonomy=taxonomy,
            concept=concept,
            value=float(value),
            unit=unit,
            fiscal_year=fiscal_year,
            fiscal_period=fiscal_period,
            period_start=str(start) if start else None,
            period_end=str(end),
            form=form,
            accn=accn,
            filed=str(filed),
            latest_value=float(latest) if latest is not None and latest != value else None,
        )
