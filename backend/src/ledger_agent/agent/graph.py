"""Assemblage du graphe LangGraph et exécution de l'agent."""

import time
from collections.abc import Callable, Iterator
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ledger_agent.agent.nodes import (
    AgentRuntime,
    AgentState,
    make_gather_node,
    make_grade_node,
    make_plan_node,
    make_synthesize_node,
    make_verify_node,
)
from ledger_agent.answer_format import extract_final_answer, is_abstention
from ledger_agent.pipeline import PipelineAnswer, RetrievedPage
from ledger_agent.retrieval.types import SearchFilters, SearchOutcome


def _node(fn: Callable[[AgentState], dict[str, Any]]) -> Any:
    # Les surcharges typées de LangGraph ne reconnaissent pas une simple fonction comme nœud
    return fn


# plan --> gather --> (grade gather) --> synthesize --> (verify synthesize) --> fin
RECURSION_LIMIT = 25


def build_graph(rt: AgentRuntime) -> CompiledStateGraph:  # type: ignore[type-arg]
    graph = StateGraph(AgentState)
    graph.add_node("plan", _node(make_plan_node(rt)))
    graph.add_node("gather", _node(make_gather_node(rt)))
    graph.add_node("synthesize", _node(make_synthesize_node(rt)))
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "gather")

    if rt.config.use_grader:
        graph.add_node("grade", _node(make_grade_node(rt)))
        graph.add_edge("gather", "grade")
        graph.add_conditional_edges(
            "grade",
            lambda s: "gather" if s.get("pending_queries") else "synthesize",
            ["gather", "synthesize"],
        )
    else:
        graph.add_edge("gather", "synthesize")

    if rt.config.use_verifier:
        graph.add_node("verify", _node(make_verify_node(rt)))
        graph.add_edge("synthesize", "verify")
        graph.add_conditional_edges(
            "verify", lambda s: "synthesize" if s.get("retry") else END, ["synthesize", END]
        )
    else:
        graph.add_edge("synthesize", END)
    return graph.compile()


def _to_search_outcome(state: dict[str, Any]) -> SearchOutcome | None:
    result = state.get("first_search")
    if result is None:
        return None
    return SearchOutcome(
        results=[],
        requested=SearchFilters(**result.requested_filters) if result.requested_filters else None,
        applied=SearchFilters(**result.applied_filters) if result.applied_filters else None,
    )


class FinancialAgent:
    def __init__(self, rt: AgentRuntime) -> None:
        self._graph = build_graph(rt)

    def invoke(self, question: str) -> dict[str, Any]:
        state: dict[str, Any] = self._graph.invoke(
            {"question": question, "llm_calls": [], "trace": []},
            {"recursion_limit": RECURSION_LIMIT},
        )
        return state

    def stream(self, question: str) -> Iterator[dict[str, Any]]:
        """Un événement « step » par étape franchie, puis un événement « answer » final."""
        started = time.perf_counter()
        final: dict[str, Any] = {}
        # Avec plusieurs modes, LangGraph renvoie des paires (mode, contenu)
        chunks: Iterator[tuple[str, dict[str, Any]]] = self._graph.stream(  # type: ignore[assignment]
            {"question": question, "llm_calls": [], "trace": []},
            {"recursion_limit": RECURSION_LIMIT},
            stream_mode=["updates", "values"],
        )
        for mode, chunk in chunks:
            if mode == "values":
                final = chunk
                continue
            for update in chunk.values():
                for event in (update or {}).get("trace", []):
                    yield {"type": "step", **_jsonable(event)}
        yield answer_event(final, time.perf_counter() - started)

    def run(self, question: str) -> PipelineAnswer:
        started = time.perf_counter()
        state = self.invoke(question)
        return pipeline_answer(state, time.perf_counter() - started)


def _cited(state: dict[str, Any]) -> list[str]:
    # Agent ReAct : citations transmises avec submit_answer ; workflow : sortie de la synthèse
    if "cited" in state:
        return list(state["cited"] or [])
    synthesis = state.get("synthesis")
    return list(synthesis.cited) if synthesis else []


def pipeline_answer(state: dict[str, Any], elapsed_s: float) -> PipelineAnswer:
    """Réponse au format commun de l'évaluation, quel que soit le type d'agent."""
    evidence = state.get("evidence", [])
    passages = [e for e in evidence if e.kind == "passage"]
    cited_ids = set(_cited(state))
    text = state.get("text", "")
    trace = [*state.get("trace", []), {"node": "total", "ms": round(elapsed_s * 1000)}]
    return PipelineAnswer(
        text=text,
        abstained=bool(state.get("abstained")) or is_abstention(text),
        retrieved=[
            RetrievedPage(e.chunk_id or "", e.doc_name or "", e.page_index or 0, e.score)
            for e in passages
        ],
        cited=[(e.doc_name or "", e.page_index or 0) for e in passages if e.id in cited_ids],
        llm_calls=state.get("llm_calls", []),
        retrieval_s=state.get("retrieval_s", 0.0),
        search=_to_search_outcome(state),
        trace=[_jsonable(t) for t in trace],
    )


EVIDENCE_PREVIEW_CHARS = 1500


def answer_event(state: dict[str, Any], elapsed_s: float) -> dict[str, Any]:
    """Réponse finale avec ses preuves et ses calculs, prête à afficher dans l'interface."""
    text = state.get("text", "")
    abstained = bool(state.get("abstained")) or is_abstention(text)
    calculations: list[dict[str, Any]] = state.get("calculations") or next(
        (t["calculations"] for t in reversed(state.get("trace", [])) if "calculations" in t), []
    )
    calls = state.get("llm_calls", [])
    return {
        "type": "answer",
        "text": text,
        "final_answer": None if abstained else extract_final_answer(text),
        "abstained": abstained,
        "cited": _cited(state),
        "evidence": [
            {
                "id": e.id,
                "kind": e.kind,
                "source_id": e.source_id,
                "title": e.title,
                "text": e.text[:EVIDENCE_PREVIEW_CHARS],
                "value": e.value,
                "doc_name": e.doc_name,
                "page_number": e.page_index + 1 if e.page_index is not None else None,
            }
            for e in state.get("evidence", [])
        ],
        "calculations": calculations,
        "llm_calls": len(calls),
        "tokens": sum(c.prompt_tokens + c.completion_tokens for c in calls),
        "elapsed_s": round(elapsed_s, 2),
    }


def _jsonable(event: dict[str, Any]) -> dict[str, Any]:
    # Les traces sont enregistrées dans predictions.jsonl : uniquement des types JSON
    return {
        k: v if isinstance(v, str | int | float | bool | list | dict | type(None)) else str(v)
        for k, v in event.items()
    }
