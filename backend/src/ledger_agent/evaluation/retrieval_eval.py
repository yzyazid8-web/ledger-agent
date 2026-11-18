"""Évaluation de la recherche seule (hit@k), gratuite : aucun appel au LLM."""

import time
from contextlib import ExitStack
from typing import Any

from ledger_agent.config import get_settings
from ledger_agent.evaluation.components import build_retriever
from ledger_agent.evaluation.config import RunConfig
from ledger_agent.evaluation.dataset import Split
from ledger_agent.evaluation.metrics import page_hit
from ledger_agent.evaluation.predict import load_split_questions
from ledger_agent.evaluation.runs import run_dir_for, write_json
from ledger_agent.evaluation.stats import bootstrap_ci

RETRIEVAL_KS = (1, 3, 5, 10)


def evaluate_retrieval(config: RunConfig, split: Split) -> dict[str, Any]:
    """Part des questions dont une page justificative figure dans les k premiers résultats."""
    questions = load_split_questions(split)
    hits: dict[int, list[bool]] = {k: [] for k in RETRIEVAL_KS}

    with ExitStack() as stack:
        retriever = build_retriever(config.retrieval, get_settings(), stack)
        started = time.perf_counter()
        for question in questions:
            gold = {(e.doc_name, e.page_index) for e in question.evidence}
            found = [
                (r.chunk.doc_name, r.chunk.page_index)
                for r in retriever.search(question.question, max(RETRIEVAL_KS))
            ]
            for k in RETRIEVAL_KS:
                hits[k].append(page_hit(found[:k], gold))
        elapsed = time.perf_counter() - started

    low, high = bootstrap_ci(hits[5])
    summary: dict[str, Any] = {
        "run": f"{config.name}-{split}",
        "questions": len(questions),
        **{f"hit@{k}": round(sum(v) / len(v), 4) for k, v in hits.items()},
        "hit@5_ci95": [round(low, 4), round(high, 4)],
        "ms_per_question": round(elapsed / len(questions) * 1000),
        "retrieval": config.retrieval.model_dump(),
    }
    write_json(run_dir_for(config.name, split) / "retrieval.json", summary)
    return summary
