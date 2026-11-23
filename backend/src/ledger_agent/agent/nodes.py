"""Noeuds du graphe de l'agent : planifier, collecter, juger, synthétiser, vérifier."""

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, ValidationError

from ledger_agent.agent.evidence import (
    Evidence,
    fact_evidence,
    passage_evidence,
    render_evidence,
)
from ledger_agent.agent.prompts import GRADER_PROMPT, SYNTHESIZER_PROMPT, planner_prompt
from ledger_agent.agent.schemas import Grade, Plan, Synthesis
from ledger_agent.agent.verify import input_matches_evidence, verify_numbers
from ledger_agent.answer_format import ABSTAIN_MARKER
from ledger_agent.ingestion.universe import load_universe
from ledger_agent.llm.types import CompletionParams, LLMClient, LLMResponse, Message
from ledger_agent.retrieval.query_analysis import extract_entities, filters_from_question
from ledger_agent.tools.calculator import CalculationError, CalculationResult, calculate
from ledger_agent.tools.financial_facts import FactStore
from ledger_agent.tools.search_filings import SearchFilingsResult, SearchFilingsTool

logger = logging.getLogger(__name__)

# « {growth} » ou avec un format Python : « {ebitda_margin:.2%} »
_PLACEHOLDER = re.compile(r"\{([a-z][a-z0-9_]*)(?::([^{}]*))?\}")
# Le juge de suffisance n'a pas besoin des passages entiers
GRADER_PASSAGE_CHARS = 600


def _concat(left: list[Any], right: list[Any]) -> list[Any]:
    return left + right


class AgentState(TypedDict, total=False):
    question: str
    plan: Plan
    evidence: list[Evidence]
    pending_queries: list[str]
    queries_done: list[str]
    search_round: int
    first_search: SearchFilingsResult | None
    synthesis: Synthesis | None
    results: list[CalculationResult]
    problems: list[str]
    attempts: int
    retry: bool
    text: str
    abstained: bool
    retrieval_s: float
    llm_calls: Annotated[list[LLMResponse], _concat]
    trace: Annotated[list[dict[str, Any]], _concat]


@dataclass(frozen=True, slots=True)
class AgentConfig:
    use_xbrl: bool = True
    use_grader: bool = True
    use_verifier: bool = True
    max_search_rounds: int = 2
    passages_per_query: int = 3
    max_passages: int = 8
    max_synthesis_attempts: int = 2


@dataclass(slots=True)
class AgentRuntime:
    planner_llm: LLMClient
    writer_llm: LLMClient
    search: SearchFilingsTool
    facts: FactStore | None
    config: AgentConfig
    planner_params: CompletionParams
    writer_params: CompletionParams


def json_call[ModelT: BaseModel](
    llm: LLMClient, messages: list[Message], params: CompletionParams, schema: type[ModelT]
) -> tuple[ModelT | None, list[LLMResponse]]:
    """Appel en mode JSON, avec une seconde chance si la sortie ne respecte pas le schéma."""
    json_params = CompletionParams(
        max_tokens=params.max_tokens,
        temperature=params.temperature,
        reasoning_effort=params.reasoning_effort,
        json_mode=True,
    )
    calls: list[LLMResponse] = []
    for _ in range(2):
        response = llm.complete(messages, json_params)
        calls.append(response)
        try:
            return schema.model_validate_json(response.text), calls
        except ValidationError as exc:
            messages = [
                *messages,
                {"role": "assistant", "content": response.text},
                {
                    "role": "user",
                    "content": f"Invalid output: {str(exc)[:400]}. Return valid JSON only.",
                },
            ]
    return None, calls


def _event(node: str, started: float, **detail: Any) -> list[dict[str, Any]]:
    return [{"node": node, "ms": round((time.perf_counter() - started) * 1000), **detail}]


# ① Planifier


def make_plan_node(rt: AgentRuntime) -> Callable[[AgentState], dict[str, Any]]:
    tickers = {c.ticker for c in load_universe()}

    def plan(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        question = state["question"]
        entities = extract_entities(question)
        messages: list[Message] = [
            {"role": "system", "content": planner_prompt()},
            {
                "role": "user",
                "content": f"Question: {question}\n"
                f"Detected by rules: tickers={list(entities.tickers)}, "
                f"years={list(entities.years)}",
            },
        ]
        result, calls = json_call(rt.planner_llm, messages, rt.planner_params, Plan)

        # Les règles servent de filet si le LLM échoue ou invente un ticker
        rules = filters_from_question(question)
        plan_out = result or Plan()
        ticker = (plan_out.ticker or "").upper()
        if ticker not in tickers:
            ticker = rules.ticker if rules and rules.ticker else ""
        plan_out = plan_out.model_copy(
            update={
                "ticker": ticker or None,
                "fiscal_year": plan_out.fiscal_year or (rules.fiscal_year if rules else None),
                "search_queries": plan_out.search_queries or [question],
                "metrics": plan_out.metrics if rt.config.use_xbrl else [],
            }
        )
        return {
            "plan": plan_out,
            "pending_queries": list(plan_out.search_queries),
            "queries_done": [],
            "search_round": 0,
            "evidence": [],
            "attempts": 0,
            "retrieval_s": 0.0,
            "llm_calls": calls,
            "trace": _event("plan", started, plan=plan_out.model_dump(), fallback=result is None),
        }

    return plan


# ② Collecter


def make_gather_node(rt: AgentRuntime) -> Callable[[AgentState], dict[str, Any]]:
    def gather(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        plan = state["plan"]
        evidence = list(state.get("evidence", []))
        seen = {e.source_id for e in evidence} | {e.chunk_id for e in evidence if e.chunk_id}
        lookups: list[dict[str, Any]] = []

        def next_id() -> str:
            return f"E{len(evidence) + 1}"

        # Faits XBRL, au premier tour seulement : ils ne dépendent pas des requêtes texte
        if state.get("search_round", 0) == 0 and rt.facts is not None and plan.ticker:
            for metric in plan.metrics:
                year = metric.fiscal_year or plan.fiscal_year
                if year is None:
                    continue
                found = None
                for concept in rt.facts.search_concepts(metric.label, plan.ticker, limit=3):
                    fact = rt.facts.get_fact(
                        plan.ticker, concept.concept, year, metric.fiscal_period, concept.taxonomy
                    )
                    if fact is not None:
                        found = fact
                        break
                lookups.append(
                    {
                        "metric": metric.label,
                        "year": year,
                        "found": found.source_id if found else None,
                    }
                )
                if found is not None and found.source_id not in seen:
                    evidence.append(fact_evidence(next_id(), found))
                    seen.add(found.source_id)

        first_search = state.get("first_search")
        searches: list[dict[str, Any]] = []
        passages = sum(e.kind == "passage" for e in evidence)
        for query in state.get("pending_queries", []):
            result = rt.search.run(
                query,
                ticker=plan.ticker,
                fiscal_year=plan.fiscal_year,
                k=rt.config.passages_per_query,
            )
            first_search = first_search or result
            searches.append(
                {"query": query, "applied": result.applied_filters, "relaxed": result.relaxed}
            )
            for passage in result.passages:
                if passage.chunk_id in seen or passages >= rt.config.max_passages:
                    continue
                evidence.append(passage_evidence(next_id(), passage))
                seen.add(passage.chunk_id)
                passages += 1

        return {
            "evidence": evidence,
            "queries_done": state.get("queries_done", []) + state.get("pending_queries", []),
            "pending_queries": [],
            "search_round": state.get("search_round", 0) + 1,
            "first_search": first_search,
            "retrieval_s": state.get("retrieval_s", 0.0) + time.perf_counter() - started,
            "trace": _event(
                "gather", started, facts=lookups, searches=searches, evidence=len(evidence)
            ),
        }

    return gather


# ③ Juger la suffisance des preuves


def make_grade_node(rt: AgentRuntime) -> Callable[[AgentState], dict[str, Any]]:
    def grade(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        short = [_shorten(e) for e in state.get("evidence", [])]
        messages: list[Message] = [
            {"role": "system", "content": GRADER_PROMPT},
            {
                "role": "user",
                "content": f"Question: {state['question']}\n\nEvidence:\n{render_evidence(short)}",
            },
        ]
        verdict, calls = json_call(rt.planner_llm, messages, rt.planner_params, Grade)
        done = set(state.get("queries_done", []))
        new = [q for q in (verdict.new_queries if verdict else []) if q not in done]
        more = (
            verdict is not None
            and not verdict.sufficient
            and bool(new)
            and state.get("search_round", 0) < rt.config.max_search_rounds
        )
        return {
            "pending_queries": new if more else [],
            "llm_calls": calls,
            "trace": _event(
                "grade",
                started,
                sufficient=verdict.sufficient if verdict else None,
                missing=verdict.missing if verdict else "",
                new_queries=new if more else [],
            ),
        }

    return grade


def _shorten(evidence: Evidence) -> Evidence:
    if evidence.kind == "xbrl":
        return evidence
    return Evidence(
        id=evidence.id,
        kind=evidence.kind,
        source_id=evidence.source_id,
        title=evidence.title,
        text=evidence.text[:GRADER_PASSAGE_CHARS],
    )


# ④ Synthétiser, en exécutant les calculs déclarés


def _format(value: float) -> str:
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"


def run_calculations(
    synthesis: Synthesis, evidence: list[Evidence]
) -> tuple[dict[str, CalculationResult], list[str]]:
    by_id = {e.id: e for e in evidence}
    results: dict[str, CalculationResult] = {}
    problems: list[str] = []
    for calc in synthesis.calculations:
        # Un calcul peut réutiliser le résultat d'un calcul précédent par son nom
        # (ex. actif moyen, puis rentabilité des actifs) : la valeur reste traçable
        variables: dict[str, float] = {name: r.value for name, r in results.items()}
        for name, source in calc.inputs.items():
            item = by_id.get(source.evidence)
            if item is None:
                problems.append(f"{calc.name}: unknown evidence {source.evidence}")
            elif not input_matches_evidence(source.value, item):
                problems.append(f"{calc.name}: {source.value} does not appear in {item.id}")
            variables[name] = source.value
        try:
            results[calc.name] = calculate(calc.expression, variables)
        except CalculationError as exc:
            problems.append(f"{calc.name}: {exc}")
    return results, problems


def fill_placeholders(text: str, results: dict[str, CalculationResult]) -> tuple[str, list[str]]:
    missing: list[str] = []

    def replace(match: re.Match[str]) -> str:
        result = results.get(match.group(1))
        if result is None:
            missing.append(match.group(1))
            return match.group(0)
        spec = match.group(2)
        if spec:
            # Un format inconnu ne doit pas laisser le gabarit brut dans la réponse
            try:
                return format(result.value, spec)
            except ValueError:
                pass
        return _format(result.value)

    return _PLACEHOLDER.sub(replace, text), [f"unknown calculation {{{m}}}" for m in missing]


def make_synthesize_node(rt: AgentRuntime) -> Callable[[AgentState], dict[str, Any]]:
    def synthesize(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        evidence = state.get("evidence", [])
        feedback = ""
        if state.get("problems"):
            feedback = (
                "\n\nYour previous answer was rejected: "
                + "; ".join(state["problems"])
                + ". Fix it: use only numbers from the evidence or declared calculations."
            )
        messages: list[Message] = [
            {"role": "system", "content": SYNTHESIZER_PROMPT},
            {
                "role": "user",
                "content": f"Question: {state['question']}\n\nEvidence:\n"
                f"{render_evidence(evidence)}{feedback}",
            },
        ]
        synthesis, calls = json_call(rt.writer_llm, messages, rt.writer_params, Synthesis)
        attempts = state.get("attempts", 0) + 1

        if synthesis is None or not synthesis.sufficient:
            reason = "invalid output" if synthesis is None else "insufficient evidence"
            return {
                "synthesis": synthesis,
                "text": ABSTAIN_MARKER,
                "abstained": True,
                "results": [],
                "problems": [],
                "attempts": attempts,
                "llm_calls": calls,
                "trace": _event("synthesize", started, abstained=True, reason=reason),
            }

        results, problems = run_calculations(synthesis, evidence)
        answer, missing_a = fill_placeholders(synthesis.answer, results)
        final, missing_f = fill_placeholders(synthesis.final_answer, results)
        problems += missing_a + missing_f
        return {
            "synthesis": synthesis,
            "text": f"{answer}\n\nFinal answer: {final}",
            "abstained": False,
            "results": list(results.values()),
            "problems": problems,
            "attempts": attempts,
            "llm_calls": calls,
            "trace": _event(
                "synthesize",
                started,
                calculations=[
                    {"name": n, "expression": r.expression, "value": r.value, "source": r.source_id}
                    for n, r in results.items()
                ],
                cited=synthesis.cited,
            ),
        }

    return synthesize


# ⑤ Vérifier par règles


def make_verify_node(rt: AgentRuntime) -> Callable[[AgentState], dict[str, Any]]:
    def verify(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        if state.get("abstained"):
            return {"retry": False, "trace": _event("verify", started, skipped="abstention")}
        check = verify_numbers(state["text"], state.get("evidence", []), state.get("results", []))
        problems = list(state.get("problems", [])) + check.problems()
        if not problems:
            return {"retry": False, "trace": _event("verify", started, ok=True)}
        if state.get("attempts", 0) < rt.config.max_synthesis_attempts:
            return {
                "retry": True,
                "problems": problems,
                "trace": _event("verify", started, ok=False, problems=problems, retry=True),
            }
        # Promesse du projet : un chiffre non traçable entraîne l'abstention
        unverified = ", ".join(check.unverified + check.sign_conflicts) or "see trace"
        return {
            "retry": False,
            "abstained": True,
            "text": f"{ABSTAIN_MARKER}\n(Unverified figures: {unverified})",
            "trace": _event("verify", started, ok=False, problems=problems, abstained=True),
        }

    return verify
