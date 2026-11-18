"""Exécution d'une évaluation en deux phases.

1. `run_predictions` : interroge le système question par question et ajoute chaque
   résultat à predictions.jsonl. Une exécution interrompue reprend là où elle s'est
   arrêtée, et le cache LLM évite de repayer les questions déjà traitées.
2. `score_run` : calcule les métriques (règles + juge LLM) et écrit le rapport.
"""

import json
import logging
import statistics
from collections import defaultdict
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from ledger_agent.baselines.naive_rag import NaiveRAG, RAGAnswer
from ledger_agent.config import Settings, get_settings
from ledger_agent.evaluation.config import EVAL_DIR, JudgeConfig, LLMConfig, RunConfig
from ledger_agent.evaluation.dataset import (
    Question,
    Split,
    load_or_create_split,
    select_split,
    to_question,
)
from ledger_agent.evaluation.judge import LLMJudge
from ledger_agent.evaluation.metrics import (
    citation_precision,
    extract_final_answer,
    is_abstention,
    is_numeric_answer,
    numeric_correct,
    page_hit,
)
from ledger_agent.ingestion.financebench import load_questions
from ledger_agent.ingestion.indexer import FastEmbedEmbedder
from ledger_agent.llm.cache import CachedLLM
from ledger_agent.llm.gemini import GeminiClient
from ledger_agent.llm.openai_compatible import OpenAICompatibleClient
from ledger_agent.llm.types import LLMClient, LLMError
from ledger_agent.retrieval.dense import DenseRetriever
from ledger_agent.retrieval.store import ChunkStore

logger = logging.getLogger(__name__)

SPLITS_FILE = EVAL_DIR / "splits.json"


def run_dir_for(config: RunConfig, split: Split) -> Path:
    return EVAL_DIR / "runs" / f"{config.name}-{split}"


def load_all_questions(settings: Settings) -> list[Question]:
    return [to_question(r) for r in load_questions(settings.data_dir / "raw" / "financebench")]


# Construction des composants


def build_llm(config: LLMConfig, settings: Settings, stack: ExitStack) -> LLMClient:
    if config.provider != "groq":
        raise ValueError(f"Unsupported LLM provider: {config.provider}")
    key = settings.groq_api_key.get_secret_value() if settings.groq_api_key else ""
    client = OpenAICompatibleClient(key, config.model, tokens_per_minute=config.tokens_per_minute)
    stack.callback(client.close)
    cached = CachedLLM(client, settings.data_dir / "cache" / "llm.sqlite")
    stack.callback(cached.close)
    return cached


def build_judge(config: JudgeConfig, settings: Settings, stack: ExitStack) -> LLMJudge | None:
    if config.provider != "gemini":
        raise ValueError(f"Unsupported judge provider: {config.provider}")
    if settings.gemini_api_key is None or not settings.gemini_api_key.get_secret_value():
        logger.warning("GEMINI_API_KEY is empty: written answers will stay unjudged")
        return None
    client = GeminiClient(
        settings.gemini_api_key.get_secret_value(),
        config.model,
        requests_per_minute=config.requests_per_minute,
    )
    stack.callback(client.close)
    cached = CachedLLM(client, settings.data_dir / "cache" / "judge.sqlite")
    stack.callback(cached.close)
    return LLMJudge(cached)


@contextmanager
def build_pipeline(config: RunConfig, settings: Settings) -> Iterator[NaiveRAG]:
    if config.pipeline != "naive_rag":
        raise ValueError(f"Unsupported pipeline: {config.pipeline}")
    with ExitStack() as stack:
        store = ChunkStore.from_parquet(settings.data_dir / "processed" / "chunks.parquet")
        qdrant = QdrantClient(path=str(settings.data_dir / "indexes" / "qdrant"))
        stack.callback(qdrant.close)
        embedder = FastEmbedEmbedder(cache_dir=settings.data_dir / "models")
        retriever = DenseRetriever(qdrant, embedder, store)
        llm = build_llm(config.llm, settings, stack)
        yield NaiveRAG(retriever, llm, top_k=config.top_k, params=config.llm.params())


# Phase 1 : prédictions


def prediction_record(question: Question, answer: RAGAnswer) -> dict[str, Any]:
    return {
        "id": question.id,
        "question_type": question.question_type,
        "question": question.question,
        "gold_answer": question.answer,
        "gold_pages": [[e.doc_name, e.page_index] for e in question.evidence],
        "prediction": answer.text,
        "retrieved": [
            {
                "chunk_id": r.chunk.chunk_id,
                "doc_name": r.chunk.doc_name,
                "page_index": r.chunk.page_index,
                "score": round(r.score, 4),
            }
            for r in answer.retrieved
        ],
        "cited": [[c.chunk.doc_name, c.chunk.page_index] for c in answer.cited],
        "llm": {
            "model": answer.llm.model,
            "prompt_tokens": answer.llm.prompt_tokens,
            "completion_tokens": answer.llm.completion_tokens,
            "latency_s": round(answer.llm.latency_s, 3),
            "cached": answer.llm.cached,
        },
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_predictions(config: RunConfig, split: Split, limit: int | None = None) -> Path:
    settings = get_settings()
    questions = load_all_questions(settings)
    selected = select_split(questions, split, load_or_create_split(SPLITS_FILE, questions))
    if limit is not None:
        selected = selected[:limit]

    run_dir = run_dir_for(config, split)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8"
    )
    predictions_path = run_dir / "predictions.jsonl"
    done = {r["id"] for r in read_jsonl(predictions_path)}
    todo = [q for q in selected if q.id not in done]
    logger.info(
        "%d questions selected, %d already done, %d to run", len(selected), len(done), len(todo)
    )

    with (
        build_pipeline(config, settings) as pipeline,
        predictions_path.open("a", encoding="utf-8") as out,
    ):
        for i, question in enumerate(todo, start=1):
            answer = pipeline.answer(question.question)
            out.write(json.dumps(prediction_record(question, answer), ensure_ascii=False) + "\n")
            out.flush()
            logger.info(
                "[%d/%d] %s %s tokens=%d%s",
                i,
                len(todo),
                question.id,
                "ABSTAIN" if answer.abstained else "ok",
                answer.llm.total_tokens,
                " (cache)" if answer.llm.cached else "",
            )
    return run_dir


# Phase 2 : métriques


def score_record(record: dict[str, Any], judge: LLMJudge | None) -> dict[str, Any]:
    gold_pages = [tuple(p) for p in record["gold_pages"]]
    retrieved_pages = [(r["doc_name"], r["page_index"]) for r in record["retrieved"]]
    cited_pages = [tuple(p) for p in record["cited"]]
    prediction = record["prediction"]
    abstained = is_abstention(prediction)
    kind = "numeric" if is_numeric_answer(record["gold_answer"]) else "text"

    correct: bool | None
    judge_reason = None
    if abstained:
        correct = False
    elif kind == "numeric":
        correct = numeric_correct(record["gold_answer"], prediction)
    elif judge is not None:
        # Une panne du juge laisse la question non jugée au lieu d'interrompre le calcul
        try:
            verdict = judge.grade(record["question"], record["gold_answer"], prediction)
        except LLMError as exc:
            logger.warning("Judge failed on %s: %s", record["id"], exc)
            correct, judge_reason = None, f"judge error: {str(exc)[:120]}"
        else:
            correct = verdict.verdict == "correct" if verdict.verdict != "invalid" else None
            judge_reason = verdict.reason
    else:
        correct = None

    return {
        "id": record["id"],
        "question_type": record["question_type"],
        "kind": kind,
        "correct": correct,
        "abstained": abstained,
        "final_answer": extract_final_answer(prediction)[:300],
        "gold_answer": record["gold_answer"],
        "page_hit": page_hit(retrieved_pages, gold_pages),
        "citation_precision": citation_precision(cited_pages, gold_pages),
        "judge_reason": judge_reason,
        "latency_s": record["llm"]["latency_s"],
        "tokens": record["llm"]["prompt_tokens"] + record["llm"]["completion_tokens"],
    }


def _rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def summarize(scores: list[dict[str, Any]]) -> dict[str, Any]:
    judged = [s for s in scores if s["correct"] is not None]
    by_kind: dict[str, list[bool]] = defaultdict(list)
    by_type: dict[str, list[bool]] = defaultdict(list)
    for s in judged:
        by_kind[s["kind"]].append(s["correct"])
        by_type[s["question_type"]].append(s["correct"])
    precisions = [s["citation_precision"] for s in scores if s["citation_precision"] is not None]
    latencies = sorted(s["latency_s"] for s in scores)

    return {
        "questions": len(scores),
        "judged": len(judged),
        "unjudged": len(scores) - len(judged),
        "accuracy": _rate([s["correct"] for s in judged]),
        "accuracy_numeric": _rate(by_kind["numeric"]),
        "accuracy_text": _rate(by_kind["text"]),
        "accuracy_by_type": {k: _rate(v) for k, v in sorted(by_type.items())},
        "abstention_rate": _rate([s["abstained"] for s in scores]),
        "page_hit_rate": _rate([s["page_hit"] for s in scores]),
        "citation_precision": round(statistics.mean(precisions), 4) if precisions else None,
        "latency_p50_s": round(statistics.median(latencies), 2) if latencies else None,
        "latency_p95_s": round(latencies[int(0.95 * (len(latencies) - 1))], 2)
        if latencies
        else None,
        "tokens_mean": round(statistics.mean(s["tokens"] for s in scores)) if scores else None,
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def summary_markdown(name: str, summary: dict[str, Any]) -> str:
    lines = [
        f"# Résultats : {name}",
        "",
        f"Questions : {summary['questions']} (jugées : {summary['judged']}, "
        f"non jugées : {summary['unjudged']})",
        "",
        "| Métrique | Valeur |",
        "|---|---|",
        f"| Exactitude (questions jugées) | {_pct(summary['accuracy'])} |",
        f"| Exactitude, réponses numériques | {_pct(summary['accuracy_numeric'])} |",
        f"| Exactitude, réponses rédigées | {_pct(summary['accuracy_text'])} |",
        f"| Taux d'abstention | {_pct(summary['abstention_rate'])} |",
        f"| Page justificative retrouvée | {_pct(summary['page_hit_rate'])} |",
        f"| Précision des citations | {_pct(summary['citation_precision'])} |",
        f"| Latence p50 / p95 | {summary['latency_p50_s']} s / {summary['latency_p95_s']} s |",
        f"| Tokens moyens par question | {summary['tokens_mean']} |",
        "",
        "| Type de question | Exactitude |",
        "|---|---|",
    ]
    lines += [f"| {k} | {_pct(v)} |" for k, v in summary["accuracy_by_type"].items()]
    return "\n".join(lines) + "\n"


def score_run(run_dir: Path, config: RunConfig, use_judge: bool = True) -> dict[str, Any]:
    settings = get_settings()
    records = read_jsonl(run_dir / "predictions.jsonl")
    with ExitStack() as stack:
        judge = build_judge(config.judge, settings, stack) if use_judge else None
        scores = [score_record(r, judge) for r in records]

    summary = summarize(scores)
    summary["run"] = run_dir.name
    summary["judge_model"] = config.judge.model if use_judge else None
    with (run_dir / "scores.jsonl").open("w", encoding="utf-8") as out:
        for s in scores:
            out.write(json.dumps(s, ensure_ascii=False) + "\n")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(summary_markdown(run_dir.name, summary), encoding="utf-8")
    return summary
