"""Agent ReAct : le LLM choisit ses outils, lit leurs résultats et décide de la suite."""

import json
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ledger_agent.agent.answer import answer_event, pipeline_answer
from ledger_agent.agent.react_prompts import (
    BUDGET_EXHAUSTED,
    NO_TOOL_CALLED,
    invalid_tool_call,
    react_system_prompt,
    verification_failed,
)
from ledger_agent.agent.toolbox import TERMINAL_TOOLS, Toolbox
from ledger_agent.agent.verify import evidence_for_numbers, verify_numbers
from ledger_agent.answer_format import ABSTAIN_MARKER, extract_final_answer, is_abstention
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
    provider_failures: int
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
    max_steps: int = 12
    passages_per_search: int = 3
    max_answer_attempts: int = 2
    # Sorties mal formées tolérées hors budget d'étapes, avant de compter comme des étapes
    max_provider_failures: int = 3


def _event(node: str, started: float, **detail: Any) -> list[dict[str, Any]]:
    return [{"node": node, "ms": round((time.perf_counter() - started) * 1000), **detail}]


# Champs qui doivent être remplis pour qu'un outil de fin soit accepté
_REQUIRED_FINAL_FIELDS = {"submit_answer": ("answer", "final_answer"), "abstain": ("reason",)}


def _missing_fields(name: str, args: Any) -> list[str]:
    required = _REQUIRED_FINAL_FIELDS.get(name, ())
    if not isinstance(args, dict):
        return list(required)
    return [f for f in required if not str(args.get(f) or "").strip()]


def _canonical(arguments: str) -> str:
    # Deux appels identiques à l'ordre des clés près ont la même empreinte
    try:
        return json.dumps(json.loads(arguments), sort_keys=True)
    except ValueError:
        return arguments


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


_FINAL_MARKER = re.compile(r"final answer\s*:", re.IGNORECASE)
_REFERENCE = re.compile(r"\b[EC]\d+\b")


def _unescape(text: str) -> str:
    # Séquences littérales « \n » (barre oblique inverse puis n), pas de vrais sauts de ligne
    return text.replace(r"\r\n", "\n").replace(r"\n", "\n").replace(r"\t", " ")


def text_submission(text: str) -> dict[str, Any] | None:
    """Réponse rédigée sans outil, reconnue à sa ligne « Final answer: » ; None sinon."""
    markers = list(_FINAL_MARKER.finditer(text))
    if not markers:
        # Sans ce repère, le texte peut n'être qu'une étape de raisonnement
        return None
    final = extract_final_answer(text)
    if is_abstention(text):
        return {"tool": "abstain", "reason": final}
    return {
        "tool": "submit_answer",
        "answer": text[: markers[-1].start()].strip(),
        "final_answer": final,
        "cited": list(dict.fromkeys(_REFERENCE.findall(text))),
    }


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
        # Appels d'outils déjà en échec pour cette question, pour repérer les répétitions
        failed_calls: set[str] = set()

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
                # Sortie mal formée refusée par le fournisseur : le modèle est prévenu et réessaie.
                # Ces ratés ne consomment pas le budget de raisonnement, dans une limite à part ;
                # au-delà, ils comptent comme des étapes, ce qui garantit l'arrêt
                failures = state.get("provider_failures", 0) + 1
                counted = failures > cfg.max_provider_failures
                messages.append({"role": "user", "content": invalid_tool_call(str(exc))})
                return {
                    "messages": messages,
                    "pending": [],
                    "steps": steps + 1 if counted else steps,
                    "provider_failures": failures,
                    "trace": _event("agent", started, step=steps + 1, actions=[], error=str(exc)),
                }
            calls = response.tool_calls
            if calls:
                messages.append(
                    {"role": "assistant", "content": response.text or None, "tool_calls": calls}
                )
            else:
                submission = text_submission(response.text)
                if submission is not None:
                    # Réponse rédigée hors outil : acceptée telle quelle, puis vérifiée
                    # comme une réponse soumise par submit_answer
                    messages.append({"role": "assistant", "content": response.text})
                    return {
                        "messages": messages,
                        "pending": [],
                        "steps": steps + 1,
                        "force_tool": False,
                        "submitted": submission,
                        "llm_calls": [response],
                        "trace": _event(
                            "agent",
                            started,
                            step=steps + 1,
                            actions=[{"tool": submission["tool"], "args": {"en_texte": True}}],
                        ),
                    }
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
                    missing = _missing_fields(name, args)
                    if missing:
                        # Une réponse sans contenu ne doit jamais être acceptée : on la refuse
                        # comme une erreur d'outil, et le modèle la soumet à nouveau
                        observation: dict[str, Any] = {
                            "error": f"{name} requires non-empty {', '.join(missing)}"
                        }
                    else:
                        submitted = {"tool": name, **args}
                        observation = {"status": "received"}
                else:
                    observation = box.execute(name, arguments)
                    # Le même appel, déjà en échec : on pousse le modèle à changer d'approche
                    key = f"{name}:{_canonical(arguments)}"
                    if "error" in observation and key in failed_calls:
                        observation["hint"] = (
                            "You already made this exact call and it failed. Change the query, "
                            "concept, year, source or values instead of repeating it."
                        )
                    if "error" in observation:
                        failed_calls.add(key)
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
            # Sauts de ligne échappés deux fois par le modèle (« \\n » écrit en toutes lettres)
            answer = _unescape(str(submitted.get("answer", ""))).rstrip()
            final = _unescape(str(submitted.get("final_answer", ""))).strip()
            cited = [str(c) for c in submitted.get("cited", []) if isinstance(c, str)]
            if not any(EVIDENCE_ID.fullmatch(c) for c in cited):
                # Le modèle n'a cité aucune preuve : on cite celles d'où viennent ses chiffres,
                # selon la même correspondance que celle qui valide les chiffres au vérificateur
                cited += evidence_for_numbers(f"{answer}\n{final}", box.evidence)
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
            problems = check.problems() + [
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
            unverified = ", ".join(check.unverified + check.sign_conflicts) or "citations"
            return {
                "attempts": attempts,
                "abstained": True,
                "text": f"{ABSTAIN_MARKER}\n(Unverified figures: {unverified})",
                "trace": _event("verify", started, ok=False, problems=problems, abstained=True),
            }

        def after_agent(state: ReactState) -> str:
            # Sans appel d'outil ni réponse, l'agent est relancé ; au-delà du budget, il s'abstient
            if state.get("pending"):
                return "tools"
            if state.get("submitted"):
                return "finish"
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
        graph.add_conditional_edges("agent", after_agent, ["tools", "finish", "agent", "give_up"])
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

    def stream(self, question: str, callbacks: list[Any] | None = None) -> Iterator[dict[str, Any]]:
        started = time.perf_counter()
        box, graph, initial = self._start(question)
        final: dict[str, Any] = {}
        # Les callbacks (suivi en production) observent le graphe sans en changer le déroulé
        run_config: dict[str, Any] = {"recursion_limit": RECURSION_LIMIT}
        if callbacks:
            run_config["callbacks"] = callbacks
        for mode, chunk in graph.stream(initial, run_config, stream_mode=["updates", "values"]):
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
