"""Assemblage du graphe LangGraph et exécution de l'agent."""

import time
from collections.abc import Callable, Iterator
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ledger_agent.agent.answer import answer_event, jsonable, pipeline_answer
from ledger_agent.agent.nodes import (
    AgentRuntime,
    AgentState,
    make_gather_node,
    make_grade_node,
    make_plan_node,
    make_synthesize_node,
    make_verify_node,
)
from ledger_agent.pipeline import PipelineAnswer


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
                    yield {"type": "step", **jsonable(event)}
        yield answer_event(final, time.perf_counter() - started)

    def run(self, question: str) -> PipelineAnswer:
        started = time.perf_counter()
        state = self.invoke(question)
        return pipeline_answer(state, time.perf_counter() - started)
