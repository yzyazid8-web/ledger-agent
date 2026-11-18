"""Assemblage du graphe LangGraph et exécution de l'agent."""

import time
from collections.abc import Callable
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
from ledger_agent.answer_format import is_abstention
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

    def run(self, question: str) -> PipelineAnswer:
        started = time.perf_counter()
        state = self.invoke(question)
        evidence = state.get("evidence", [])
        passages = [e for e in evidence if e.kind == "passage"]
        cited_ids = set(state["synthesis"].cited) if state.get("synthesis") else set()
        text = state.get("text", "")
        trace = [
            *state.get("trace", []),
            {"node": "total", "ms": round((time.perf_counter() - started) * 1000)},
        ]
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


def _jsonable(event: dict[str, Any]) -> dict[str, Any]:
    # Les traces sont enregistrées dans predictions.jsonl : uniquement des types JSON
    return {
        k: v if isinstance(v, str | int | float | bool | list | dict | type(None)) else str(v)
        for k, v in event.items()
    }
