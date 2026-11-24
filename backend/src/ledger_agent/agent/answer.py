"""Réponse finale au format commun, quel que soit l'agent (ReAct ou workflow à étapes fixes).

Partagé par les deux architectures : l'agent ReAct ne dépend ainsi pas du code du workflow.
"""

from typing import Any

from ledger_agent.answer_format import extract_final_answer, is_abstention
from ledger_agent.pipeline import PipelineAnswer, RetrievedPage
from ledger_agent.retrieval.types import SearchFilters, SearchOutcome


def _to_search_outcome(state: dict[str, Any]) -> SearchOutcome | None:
    result = state.get("first_search")
    if result is None:
        return None
    return SearchOutcome(
        results=[],
        requested=SearchFilters(**result.requested_filters) if result.requested_filters else None,
        applied=SearchFilters(**result.applied_filters) if result.applied_filters else None,
    )


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
        trace=[jsonable(t) for t in trace],
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


def jsonable(event: dict[str, Any]) -> dict[str, Any]:
    # Les traces sont enregistrées dans predictions.jsonl : uniquement des types JSON
    return {
        k: v if isinstance(v, str | int | float | bool | list | dict | type(None)) else str(v)
        for k, v in event.items()
    }
