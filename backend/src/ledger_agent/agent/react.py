"""Agent ReAct : le LLM choisit ses outils, lit leurs résultats et décide de la suite."""

import json
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ledger_agent.agent.graph import answer_event, pipeline_answer
from ledger_agent.agent.react_prompts import (
    BUDGET_EXHAUSTED,
    NO_TOOL_CALLED,
    invalid_tool_call,
    react_system_prompt,
    verification_failed,
)
from ledger_agent.agent.toolbox import TERMINAL_TOOLS, Toolbox
from ledger_agent.agent.verify import verify_numbers
from ledger_agent.answer_format import ABSTAIN_MARKER
from ledger_agent.llm.types import (
    CompletionParams,
    LLMClient,
    LLMResponse,
    LLMToolCallError,
    Message,
)
from ledger_agent.pipeline import PipelineAnswer
from ledger_agent.tools.financial_facts import FactStore
from ledger_agent.tools.search_filings import SearchFilingsTool

RECURSION_LIMIT = 60
ARGUMENT_PREVIEW_CHARS = 200
EVIDENCE_ID = re.compile(r"E\d+")


def _concat(left: list[Any], right: list[Any]) -> list[Any]:
    return left + right


class ReactState(TypedDict, total=False):
    question: str
    messages: list[Message]
    pending: list[dict[str, Any]]
    steps: int
    attempts: int
    force_tool: bool
    submitted: dict[str, Any] | None
    text: str
    abstained: bool
    cited: list[str]
    retrieval_s: float
    llm_calls: Annotated[list[LLMResponse], _concat]
    trace: Annotated[list[dict[str, Any]], _concat]


@dataclass(frozen=True, slots=True)
class ReactConfig:
    use_xbrl: bool = True
    use_verifier: bool = True
    max_steps: int = 8
    passages_per_search: int = 3
    max_answer_attempts: int = 2


def _event(node: str, started: float, **detail: Any) -> list[dict[str, Any]]:
    return [{"node": node, "ms": round((time.perf_counter() - started) * 1000), **detail}]


def _preview(arguments: str) -> Any:
    try:
        return json.loads(arguments)
    except ValueError:
        return arguments[:ARGUMENT_PREVIEW_CHARS]


def _summary(name: str, observation: dict[str, Any]) -> str:
    """Résumé lisible d'une observation, pour la trace affichée dans l'interface."""
    if "error" in observation:
        return f"erreur : {observation['error']}"
    if name == "search_filings":
        ids = [p["id"] for p in observation.get("passages", [])]
        warning = " (filtres élargis)" if "warning" in observation else ""
        return f"{len(ids)} passages {', '.join(ids)}{warning}"
    if name == "find_concepts":
        return ", ".join(c["concept"] for c in observation.get("candidates", [])[:3])
    if name == "get_fact":
        return f"{observation['id']} = {observation['value']:,.0f} {observation['unit']}"
    if name == "calculate":
        return f"{observation['id']} = {observation['value']:,.4g}"
    return ""


class ReactAgent:
    def __init__(
        self,
        llm: LLMClient,
        search: SearchFilingsTool,
        facts: FactStore | None,
        config: ReactConfig,
        params: CompletionParams,
    ) -> None:
        self._llm = llm
        self._search = search
        self._facts = facts if config.use_xbrl else None
        self._config = config
        self._params = params

    # Construction du graphe, avec une boîte à outils neuve pour chaque question

    def _graph(self, box: Toolbox) -> Any:
        cfg, llm = self._config, self._llm

        def agent(state: ReactState) -> dict[str, Any]:
            started = time.perf_counter()
            steps = state.get("steps", 0)
            messages = list(state["messages"])
            exhausted = steps >= cfg.max_steps
            if exhausted:
                messages.append({"role": "user", "content": BUDGET_EXHAUSTED})
            params = CompletionParams(
                max_tokens=self._params.max_tokens,
                temperature=self._params.temperature,
                reasoning_effort=self._params.reasoning_effort,
                tools=box.definitions(final_only=exhausted),
                # Après une réponse en texte libre, ou au dernier tour, un appel d'outil est
                # imposé : le modèle doit soumettre sa réponse au lieu de la rédiger hors outil
                tool_choice="required" if exhausted or state.get("force_tool") else "auto",
            )
            try:
                response = llm.complete(messages, params)
            except LLMToolCallError as exc:
                # Appel mal formé refusé par le fournisseur : le modèle est prévenu et réessaie,
                # dans la limite du budget d'étapes
                messages.append({"role": "user", "content": invalid_tool_call(str(exc))})
                return {
                    "messages": messages,
                    "pending": [],
                    "steps": steps + 1,
                    "trace": _event("agent", started, step=steps + 1, actions=[], error=str(exc)),
                }
            calls = response.tool_calls
            if calls:
                messages.append(
                    {"role": "assistant", "content": response.text or None, "tool_calls": calls}
                )
            else:
                messages += [
                    {"role": "assistant", "content": response.text},
                    {"role": "user", "content": NO_TOOL_CALLED},
                ]
            return {
                "messages": messages,
                "pending": [dict(c) for c in calls],
                "steps": steps + 1,
                "force_tool": not calls,
                "llm_calls": [response],
                "trace": _event(
                    "agent",
                    started,
                    step=steps + 1,
                    thought=response.text[:300],
                    actions=[
                        {
                            "tool": c["function"]["name"],
                            "args": _preview(c["function"]["arguments"]),
                        }
                        for c in calls
                    ],
                ),
            }

        def tools(state: ReactState) -> dict[str, Any]:
            started = time.perf_counter()
            messages = list(state["messages"])
            results: list[dict[str, Any]] = []
            submitted: dict[str, Any] | None = None
            for call in state.get("pending", []):
                name, arguments = call["function"]["name"], call["function"]["arguments"]
                if name in TERMINAL_TOOLS:
                    args = _preview(arguments)
                    submitted = {"tool": name, **(args if isinstance(args, dict) else {})}
                    observation: dict[str, Any] = {"status": "received"}
                else:
                    observation = box.execute(name, arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(observation, ensure_ascii=False, default=str),
                    }
                )
                results.append({"tool": name, "summary": _summary(name, observation)})
            return {
                "messages": messages,
                "pending": [],
                "submitted": submitted,
                "retrieval_s": state.get("retrieval_s", 0.0) + time.perf_counter() - started,
                "trace": _event("tool", started, results=results),
            }

        def finish(state: ReactState) -> dict[str, Any]:
            submitted = state.get("submitted") or {}
            if submitted.get("tool") == "abstain":
                return {
                    "text": f"{ABSTAIN_MARKER}\n({submitted.get('reason', '')})",
                    "abstained": True,
                    "cited": [],
                }
            answer = str(submitted.get("answer", "")).rstrip()
            final = str(submitted.get("final_answer", ""))
            cited = [str(c) for c in submitted.get("cited", []) if isinstance(c, str)]
            # Une source citée doit apparaître dans le texte, où l'interface la rend cliquable
            missing = [c for c in cited if EVIDENCE_ID.fullmatch(c) and c not in answer]
            if missing:
                answer += " " + " ".join(f"[{c}]" for c in missing)
            return {
                "text": f"{answer}\n\nFinal answer: {final}",
                "abstained": False,
                "cited": cited,
            }

        def verify(state: ReactState) -> dict[str, Any]:
            started = time.perf_counter()
            if state.get("abstained"):
                return {"trace": _event("verify", started, skipped="abstention")}
            check = verify_numbers(state["text"], box.evidence, list(box.calculations.values()))
            # Citer un calcul (C1) est aussi légitime que citer une preuve (E1)
            known = {e.id for e in box.evidence} | set(box.calculations)
            problems = [f"untraceable number {n}" for n in check.unverified] + [
                f"unknown citation {c}" for c in state.get("cited", []) if c not in known
            ]
            if not problems:
                return {
                    "attempts": state.get("attempts", 0) + 1,
                    "trace": _event("verify", started, ok=True),
                }
            attempts = state.get("attempts", 0) + 1
            if attempts < cfg.max_answer_attempts:
                # L'agent reprend la main : il peut chercher ou calculer ce qui manque
                messages = [
                    *state["messages"],
                    {"role": "user", "content": verification_failed(problems)},
                ]
                return {
                    "attempts": attempts,
                    "messages": messages,
                    "submitted": None,
                    "trace": _event("verify", started, ok=False, problems=problems, retry=True),
                }
            unverified = ", ".join(check.unverified) or "citations"
            return {
                "attempts": attempts,
                "abstained": True,
                "text": f"{ABSTAIN_MARKER}\n(Unverified figures: {unverified})",
                "trace": _event("verify", started, ok=False, problems=problems, abstained=True),
            }

        def after_agent(state: ReactState) -> str:
            # Sans appel d'outil, l'agent est relancé ; au-delà du budget, il s'abstient
            if state.get("pending"):
                return "tools"
            return "agent" if state.get("steps", 0) <= cfg.max_steps else "give_up"

        def after_tools(state: ReactState) -> str:
            if state.get("submitted"):
                return "finish"
            # Budget dépassé et toujours pas de réponse : on arrête plutôt que de tourner en rond
            return "agent" if state.get("steps", 0) <= cfg.max_steps else "give_up"

        def after_verify(state: ReactState) -> str:
            return "agent" if state.get("submitted") is None and not state.get("abstained") else END

        def give_up(state: ReactState) -> dict[str, Any]:
            return {
                "text": f"{ABSTAIN_MARKER}\n(no answer within the step budget)",
                "abstained": True,
            }

        graph = StateGraph(ReactState)
        for name, fn in (
            ("agent", agent),
            ("tools", tools),
            ("finish", finish),
            ("give_up", give_up),
        ):
            graph.add_node(name, _node(fn))
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", after_agent, ["tools", "agent", "give_up"])
        graph.add_conditional_edges("tools", after_tools, ["finish", "agent", "give_up"])
        graph.add_edge("give_up", END)
        if cfg.use_verifier:
            graph.add_node("verify", _node(verify))
            graph.add_edge("finish", "verify")
            graph.add_conditional_edges("verify", after_verify, ["agent", END])
        else:
            graph.add_edge("finish", END)
        return graph.compile()

    def _start(self, question: str) -> tuple[Toolbox, Any, dict[str, Any]]:
        box = Toolbox(
            self._search, self._facts, passages_per_search=self._config.passages_per_search
        )
        initial = {
            "question": question,
            "messages": [
                {"role": "system", "content": react_system_prompt(self._config.max_steps)},
                {"role": "user", "content": question},
            ],
            "llm_calls": [],
            "trace": [],
        }
        return box, self._graph(box), initial

    @staticmethod
    def _with_tools(state: dict[str, Any], box: Toolbox) -> dict[str, Any]:
        # Les preuves et calculs vivent dans la boîte à outils : on les joint à l'état final
        return {
            **state,
            "evidence": box.evidence,
            "calculations": [
                {"name": cid, "expression": r.expression, "value": r.value, "source": r.source_id}
                for cid, r in box.calculations.items()
            ],
        }

    def stream(self, question: str) -> Iterator[dict[str, Any]]:
        started = time.perf_counter()
        box, graph, initial = self._start(question)
        final: dict[str, Any] = {}
        for mode, chunk in graph.stream(
            initial, {"recursion_limit": RECURSION_LIMIT}, stream_mode=["updates", "values"]
        ):
            if mode == "values":
                final = chunk
                continue
            for update in chunk.values():
                for event in (update or {}).get("trace", []):
                    yield {"type": "step", **event}
        yield answer_event(self._with_tools(final, box), time.perf_counter() - started)

    def run(self, question: str) -> PipelineAnswer:
        started = time.perf_counter()
        box, graph, initial = self._start(question)
        state = graph.invoke(initial, {"recursion_limit": RECURSION_LIMIT})
        return pipeline_answer(self._with_tools(state, box), time.perf_counter() - started)


def _node(fn: Callable[[ReactState], dict[str, Any]]) -> Any:
    # Les surcharges typées de LangGraph ne reconnaissent pas une simple fonction comme nœud
    return fn
