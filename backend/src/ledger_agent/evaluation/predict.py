"""Phase 1 : génération des réponses, question par question, avec reprise sûre."""

import json
import logging
import time
from pathlib import Path
from typing import Any

from ledger_agent.config import get_settings
from ledger_agent.evaluation.components import build_pipeline, pipeline_prompt_sha
from ledger_agent.evaluation.config import EVAL_DIR, RunConfig
from ledger_agent.evaluation.dataset import (
    Question,
    Split,
    load_or_create_split,
    select_split,
    to_question,
)
from ledger_agent.evaluation.runs import PREDICTIONS_FILE, latest_records, prepare_run
from ledger_agent.ingestion.financebench import load_questions
from ledger_agent.llm.types import LLMError, LLMQuotaError
from ledger_agent.pipeline import PipelineAnswer

logger = logging.getLogger(__name__)

SPLITS_FILE = EVAL_DIR / "splits.json"


def load_split_questions(split: Split) -> list[Question]:
    settings = get_settings()
    rows = load_questions(settings.data_dir / "raw" / "financebench")
    questions = [to_question(r) for r in rows]
    return select_split(questions, split, load_or_create_split(SPLITS_FILE, questions))


def _base_record(question: Question) -> dict[str, Any]:
    return {
        "id": question.id,
        "question_type": question.question_type,
        "question": question.question,
        "gold_answer": question.answer,
        "gold_pages": [[e.doc_name, e.page_index] for e in question.evidence],
    }


def prediction_record(question: Question, answer: PipelineAnswer, total_s: float) -> dict[str, Any]:
    search = answer.search
    return {
        **_base_record(question),
        "prediction": answer.text,
        "retrieved": [
            {
                "chunk_id": r.chunk_id,
                "doc_name": r.doc_name,
                "page_index": r.page_index,
                "score": round(r.score, 4),
            }
            for r in answer.retrieved
        ],
        "cited": [list(page) for page in answer.cited],
        "search": {
            "requested": search.requested.as_dict() if search and search.requested else None,
            "applied": search.applied.as_dict() if search and search.applied else None,
            "relaxed": search.relaxed if search else False,
        },
        "llm": {
            "models": sorted({c.model for c in answer.llm_calls}),
            "calls": len(answer.llm_calls),
            "prompt_tokens": answer.prompt_tokens,
            "completion_tokens": answer.completion_tokens,
            "cached": answer.all_cached,
        },
        # Latence de bout en bout : recherche comprise, pas seulement les appels au LLM
        "timing": {
            "retrieval_s": round(answer.retrieval_s, 3),
            "llm_s": round(answer.llm_s, 3),
            "total_s": round(total_s, 3),
        },
        "trace": answer.trace,
    }


def run_predictions(config: RunConfig, split: Split, limit: int | None = None) -> Path:
    """Génère les réponses manquantes ; s'arrête proprement si le quota est épuisé."""
    settings = get_settings()
    full = load_split_questions(split)
    selected = full[:limit] if limit is not None else full
    # L'empreinte porte sur le jeu complet : un essai avec --limit peut être prolongé ensuite
    run_dir = prepare_run(
        config,
        split,
        [q.id for q in full],
        settings,
        expected=len(selected),
        prompt_sha=pipeline_prompt_sha(config),
    )
    predictions_path = run_dir / PREDICTIONS_FILE

    # Les questions en erreur lors d'une session précédente sont retentées
    done = {i for i, r in latest_records(predictions_path).items() if "error" not in r}
    todo = [q for q in selected if q.id not in done]
    logger.info("%d questions, %d déjà faites, %d à traiter", len(selected), len(done), len(todo))

    with (
        build_pipeline(config, settings) as pipeline,
        predictions_path.open("a", encoding="utf-8") as out,
    ):
        for i, question in enumerate(todo, start=1):
            started = time.perf_counter()
            try:
                answer = pipeline.run(question.question)
            except LLMQuotaError as exc:
                logger.warning(
                    "Arrêt : %s. Relance la même commande plus tard pour reprendre.", exc
                )
                break
            except LLMError as exc:
                # Une question en échec n'arrête pas les autres ; elle sera retentée à la reprise
                logger.warning("[%d/%d] %s en erreur : %s", i, len(todo), question.id, exc)
                record = {**_base_record(question), "error": str(exc)[:300]}
            else:
                record = prediction_record(question, answer, time.perf_counter() - started)
                logger.info(
                    "[%d/%d] %s %s tokens=%d%s",
                    i,
                    len(todo),
                    question.id,
                    "ABSTENTION" if answer.abstained else "ok",
                    answer.prompt_tokens + answer.completion_tokens,
                    " (cache)" if answer.all_cached else "",
                )
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
    return run_dir
