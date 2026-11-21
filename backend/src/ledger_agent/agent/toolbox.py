"""Outils exposés au LLM dans la boucle ReAct, et registre des preuves collectées.

Chaque appel renvoie une « observation » que le modèle lit avant de décider de la suite.
Les erreurs sont renvoyées comme observations, pas levées : le modèle peut se corriger.
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ledger_agent.agent.evidence import Evidence, fact_evidence, passage_evidence
from ledger_agent.agent.verify import FORMULA_CONSTANTS, formula_literals, input_matches_evidence
from ledger_agent.tools.calculator import CalculationError, CalculationResult, calculate
from ledger_agent.tools.financial_facts import VALID_PERIODS, FactStore
from ledger_agent.tools.search_filings import SearchFilingsTool

logger = logging.getLogger(__name__)

OBSERVATION_PASSAGE_CHARS = 1200
MAX_CONCEPTS = 5
TERMINAL_TOOLS = frozenset({"submit_answer", "abstain"})


def _function(
    name: str, description: str, properties: dict[str, Any], required: list[str]
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


# Descriptions courtes : elles sont renvoyées au modèle à chaque tour, et les tokens
# comptent dans la limite de débit (8 000 par minute sur l'offre gratuite de Groq)
SEARCH_FILINGS = _function(
    "search_filings",
    "Search passages of SEC filings, for anything not in XBRL. Returns passages E<n>.",
    {
        "query": {"type": "string"},
        "ticker": {"type": "string"},
        "fiscal_year": {"type": "integer"},
    },
    ["query"],
)
FIND_CONCEPTS = _function(
    "find_concepts",
    "Find the XBRL concept names a company uses for a line item. Call before get_fact.",
    {"label": {"type": "string"}, "ticker": {"type": "string"}},
    ["label", "ticker"],
)
GET_FACT = _function(
    "get_fact",
    "Official XBRL value of a concept, as filed, in raw units. Returns a fact E<n>.",
    {
        "ticker": {"type": "string"},
        "concept": {"type": "string"},
        "fiscal_year": {"type": "integer"},
        "fiscal_period": {"type": "string", "enum": list(VALID_PERIODS)},
    },
    ["ticker", "concept", "fiscal_year"],
)
CALCULATE = _function(
    "calculate",
    "Evaluate an expression over named inputs, each citing its source E<n> or C<n> and the "
    "exact value shown there. Returns a result C<n>.",
    {
        "expression": {"type": "string", "description": "e.g. (a - b) / b * 100"},
        "inputs": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {"source": {"type": "string"}, "value": {"type": "number"}},
                "required": ["source", "value"],
            },
        },
    },
    ["expression", "inputs"],
)
SUBMIT_ANSWER = _function(
    "submit_answer",
    "Submit the answer, citing evidence after each fact it uses.",
    {
        "answer": {"type": "string"},
        "final_answer": {"type": "string", "description": "Concise answer"},
        "cited": {"type": "array", "items": {"type": "string"}},
    },
    ["answer", "final_answer", "cited"],
)
ABSTAIN = _function(
    "abstain",
    "Give up when the information is not available.",
    {"reason": {"type": "string"}},
    ["reason"],
)


@dataclass
class Toolbox:
    """Outils d'une seule question : le registre des preuves repart de zéro à chaque question."""

    search: SearchFilingsTool
    facts: FactStore | None
    passages_per_search: int = 3
    evidence: list[Evidence] = field(default_factory=list)
    calculations: dict[str, CalculationResult] = field(default_factory=dict)
    _seen: dict[str, str] = field(default_factory=dict)

    def definitions(self, final_only: bool = False) -> tuple[dict[str, Any], ...]:
        if final_only:
            return (SUBMIT_ANSWER, ABSTAIN)
        tools = [SEARCH_FILINGS]
        if self.facts is not None:
            tools += [FIND_CONCEPTS, GET_FACT]
        return (*tools, CALCULATE, SUBMIT_ANSWER, ABSTAIN)

    def _register(self, make: Any, key: str) -> Evidence:
        if key in self._seen:
            return next(e for e in self.evidence if e.id == self._seen[key])
        item: Evidence = make(f"E{len(self.evidence) + 1}")
        self.evidence.append(item)
        self._seen[key] = item.id
        return item

    def execute(self, name: str, arguments: str) -> dict[str, Any]:
        # Liste fermée : le nom choisi par le modèle ne peut pas atteindre une autre méthode
        handlers: dict[str, Callable[..., dict[str, Any]]] = {
            "search_filings": self._search_filings,
            "calculate": self._calculate,
        }
        if self.facts is not None:
            handlers |= {"find_concepts": self._find_concepts, "get_fact": self._get_fact}
        handler = handlers.get(name)
        if handler is None:
            return {"error": f"outil inconnu : {name}"}
        try:
            args = json.loads(arguments or "{}")
            if not isinstance(args, dict):
                raise ValueError("les arguments doivent être un objet JSON")
            result: dict[str, Any] = handler(**args)
            return result
        except (TypeError, ValueError) as exc:
            return {"error": f"arguments invalides pour {name} : {exc}"}

    # Outils

    def _search_filings(
        self, query: str, ticker: str | None = None, fiscal_year: int | None = None
    ) -> dict[str, Any]:
        result = self.search.run(
            query, ticker=ticker, fiscal_year=fiscal_year, k=self.passages_per_search
        )
        passages = []
        for passage in result.passages:
            item = self._register(lambda i, p=passage: passage_evidence(i, p), passage.chunk_id)
            passages.append(
                {
                    "id": item.id,
                    "source": passage.header,
                    "text": passage.text[:OBSERVATION_PASSAGE_CHARS],
                }
            )
        observation: dict[str, Any] = {"passages": passages}
        if result.relaxed:
            # Le modèle doit savoir que certains passages ne respectent pas ses filtres
            applied = result.applied_filters or "none"
            observation["warning"] = f"filters relaxed from {result.requested_filters} to {applied}"
        return observation

    def _find_concepts(self, label: str, ticker: str) -> dict[str, Any]:
        assert self.facts is not None
        matches = self.facts.search_concepts(label, ticker.upper(), limit=MAX_CONCEPTS)
        if not matches:
            return {"error": f"aucun concept XBRL trouvé pour « {label} » chez {ticker}"}
        return {
            "candidates": [
                {"concept": m.concept, "label": m.label, "facts": m.facts} for m in matches
            ]
        }

    def _get_fact(
        self, ticker: str, concept: str, fiscal_year: int, fiscal_period: str = "FY"
    ) -> dict[str, Any]:
        assert self.facts is not None
        fact = self.facts.get_fact(ticker, concept, int(fiscal_year), fiscal_period)
        if fact is None:
            where = f"{ticker}, {fiscal_year} {fiscal_period}"
            return {"error": f"aucune valeur pour {concept} chez {where}"}
        item = self._register(lambda i: fact_evidence(i, fact), fact.source_id)
        observation: dict[str, Any] = {
            "id": item.id,
            "value": fact.value,
            "unit": fact.unit,
            "label": fact.label,
            "period": f"{fact.period_start} to {fact.period_end}"
            if fact.period_start
            else fact.period_end,
            "filed_in": f"{fact.form} filed {fact.filed}",
        }
        if fact.latest_value is not None:
            observation["restated_to"] = fact.latest_value
        return observation

    def _calculate(self, expression: str, inputs: dict[str, Any]) -> dict[str, Any]:
        by_id = {e.id: e for e in self.evidence}
        # Un chiffre écrit en clair dans la formule échapperait au contrôle des sources
        smuggled = [n for n in formula_literals(expression) if n not in FORMULA_CONSTANTS]
        if smuggled:
            return {
                "error": f"les nombres {smuggled} doivent être passés en entrées (inputs) avec "
                "leur source E<n> ou C<n>, pas écrits dans la formule"
            }
        variables: dict[str, float] = {}
        for name, source in inputs.items():
            if not isinstance(source, dict) or "source" not in source or "value" not in source:
                return {"error": f"l'entrée {name} doit être {{source, value}}"}
            ref, value = str(source["source"]), float(source["value"])
            if ref in self.calculations:
                if abs(self.calculations[ref].value - value) > max(abs(value) * 0.005, 1e-9):
                    return {"error": f"{name} : {value} ne correspond pas au résultat de {ref}"}
            elif ref in by_id:
                if not input_matches_evidence(value, by_id[ref]):
                    return {"error": f"{name} : {value} n'apparaît pas dans {ref}"}
            else:
                return {"error": f"{name} : source inconnue {ref}"}
            variables[name] = value
        try:
            result = calculate(expression, variables)
        except CalculationError as exc:
            return {"error": str(exc)}
        calc_id = f"C{len(self.calculations) + 1}"
        self.calculations[calc_id] = result
        return {"id": calc_id, "value": result.value}
