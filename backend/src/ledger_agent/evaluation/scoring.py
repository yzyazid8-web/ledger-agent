"""Phase 2 : métriques par question, résumé avec intervalles de confiance, rapport."""

import json
import logging
import statistics
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from ledger_agent.answer_format import extract_final_answer, is_abstention
from ledger_agent.config import get_settings
from ledger_agent.evaluation.components import build_judge
from ledger_agent.evaluation.config import JudgeConfig
from ledger_agent.evaluation.judge import JUDGE_PROMPT_SHA, LLMJudge
from ledger_agent.evaluation.metrics import (
    citation_precision,
    is_numeric_answer,
    numeric_correct,
    page_hit,
)
from ledger_agent.evaluation.runs import PREDICTIONS_FILE, latest_records, load_run_meta, write_json
from ledger_agent.evaluation.stats import bootstrap_ci, cohen_kappa
from ledger_agent.llm.types import LLMError

logger = logging.getLogger(__name__)


def _judge(judge: LLMJudge, record: dict[str, Any]) -> tuple[bool | None, str]:
    """Verdict du juge, ou None s'il échoue ou répond hors format."""
    try:
        verdict = judge.grade(record["question"], record["gold_answer"], record["prediction"])
    except LLMError as exc:
        logger.warning("Juge en échec sur %s : %s", record["id"], exc)
        return None, f"erreur du juge : {str(exc)[:120]}"
    if verdict.verdict == "invalid":
        return None, f"verdict invalide : {verdict.reason[:120]}"
    return verdict.verdict == "correct", verdict.reason


def score_record(record: dict[str, Any], judge: LLMJudge | None) -> dict[str, Any]:
    kind = "numeric" if is_numeric_answer(record["gold_answer"]) else "text"
    base = {
        "id": record["id"],
        "question_type": record["question_type"],
        "kind": kind,
        "gold_answer": record["gold_answer"],
    }
    if "error" in record:
        # Une question en erreur compte comme fausse : l'exclure gonflerait l'exactitude
        return {
            **base,
            "correct": False,
            "error": True,
            "abstained": False,
            "page_hit": False,
            "citation_precision": None,
            "judge_correct": None,
            "judge_reason": None,
            "final_answer": "",
            "latency_s": None,
            "tokens": 0,
        }

    prediction = record["prediction"]
    gold_pages = [tuple(p) for p in record["gold_pages"]]
    abstained = is_abstention(prediction)
    judge_correct: bool | None = None
    judge_reason: str | None = None
    if judge is not None and not abstained:
        judge_correct, judge_reason = _judge(judge, record)

    correct: bool | None
    if abstained:
        correct = False
    elif kind == "numeric":
        # Le juge note aussi les réponses chiffrées, pour mesurer son accord avec les règles
        correct = numeric_correct(record["gold_answer"], prediction)
    else:
        correct = judge_correct

    timing = record.get("timing") or {}
    return {
        **base,
        "correct": correct,
        "error": False,
        "abstained": abstained,
        "page_hit": page_hit(
            [(r["doc_name"], r["page_index"]) for r in record["retrieved"]], gold_pages
        ),
        "citation_precision": citation_precision([tuple(p) for p in record["cited"]], gold_pages),
        "judge_correct": judge_correct,
        "judge_reason": judge_reason,
        "final_answer": extract_final_answer(prediction)[:300],
        "filters_relaxed": (record.get("search") or {}).get("relaxed"),
        "latency_s": timing.get("total_s", (record.get("llm") or {}).get("latency_s")),
        "tokens": record["llm"]["prompt_tokens"] + record["llm"]["completion_tokens"],
    }


def _rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _ci(values: list[bool]) -> list[float] | None:
    if not values:
        return None
    low, high = bootstrap_ci(values)
    return [round(low, 4), round(high, 4)]


def summarize(scores: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    # Prudent : une question non jugée ou en erreur compte comme non réussie
    hits = [s["correct"] is True for s in scores]
    by_kind: dict[str, list[bool]] = defaultdict(list)
    by_type: dict[str, list[bool]] = defaultdict(list)
    for s, ok in zip(scores, hits, strict=True):
        by_kind[s["kind"]].append(ok)
        by_type[s["question_type"]].append(ok)

    # Accord juge / règles sur les réponses chiffrées jugées par les deux
    pairs = [
        (s["correct"], s["judge_correct"])
        for s in scores
        if s["kind"] == "numeric" and s["correct"] is not None and s["judge_correct"] is not None
    ]
    precisions = [s["citation_precision"] for s in scores if s["citation_precision"] is not None]
    latencies = sorted(s["latency_s"] for s in scores if s["latency_s"] is not None)
    answered = [s for s in scores if not s["error"]]

    return {
        "questions": len(scores),
        "questions_expected": expected,
        "complete": len(scores) == expected,
        "errors": sum(s["error"] for s in scores),
        "unjudged": sum(s["correct"] is None for s in scores),
        "accuracy": _rate(hits),
        "accuracy_ci95": _ci(hits),
        "accuracy_numeric": _rate(by_kind["numeric"]),
        "accuracy_text": _rate(by_kind["text"]),
        "accuracy_by_type": {k: _rate(v) for k, v in sorted(by_type.items())},
        "abstention_rate": _rate([s["abstained"] for s in scores]),
        "wrong_answer_rate": _rate([s["correct"] is False and not s["abstained"] for s in scores]),
        "page_hit_rate": _rate([s["page_hit"] for s in scores]),
        "citation_precision": round(statistics.mean(precisions), 4) if precisions else None,
        "filters_relaxed_rate": _rate([bool(s.get("filters_relaxed")) for s in answered]),
        "judge_agreement": {
            "n": len(pairs),
            "agreement": _rate([a == b for a, b in pairs]),
            "kappa": round(k, 3)
            if (k := cohen_kappa(*zip(*pairs, strict=True))) is not None
            else None,
        }
        if pairs
        else None,
        "latency_p50_s": round(statistics.median(latencies), 2) if latencies else None,
        "latency_p95_s": round(latencies[int(0.95 * (len(latencies) - 1))], 2)
        if latencies
        else None,
        "tokens_mean": round(statistics.mean(s["tokens"] for s in answered)) if answered else None,
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def summary_markdown(name: str, summary: dict[str, Any]) -> str:
    ci = summary["accuracy_ci95"]
    ci_text = f" (IC 95 % : {ci[0]:.0%} à {ci[1]:.0%})" if ci else ""
    agreement = summary["judge_agreement"]
    lines = [f"# Résultats : {name}", ""]
    if not summary["complete"]:
        lines += [
            f"> ⚠️ Exécution incomplète : {summary['questions']} questions sur "
            f"{summary['questions_expected']}. Ces chiffres ne sont pas comparables.",
            "",
        ]
    lines += [
        f"Questions : {summary['questions']} · erreurs : {summary['errors']} · "
        f"non jugées : {summary['unjudged']} (comptées comme non réussies)",
        "",
        "| Métrique | Valeur |",
        "|---|---|",
        f"| Exactitude | {_pct(summary['accuracy'])}{ci_text} |",
        f"| Exactitude, réponses chiffrées | {_pct(summary['accuracy_numeric'])} |",
        f"| Exactitude, réponses rédigées | {_pct(summary['accuracy_text'])} |",
        f"| Taux d'abstention | {_pct(summary['abstention_rate'])} |",
        f"| Réponses fausses (hors abstention) | {_pct(summary['wrong_answer_rate'])} |",
        f"| Page justificative retrouvée | {_pct(summary['page_hit_rate'])} |",
        f"| Précision des citations | {_pct(summary['citation_precision'])} |",
        f"| Filtres de recherche élargis | {_pct(summary['filters_relaxed_rate'])} |",
        f"| Latence de bout en bout p50 / p95 | {summary['latency_p50_s']} s / "
        f"{summary['latency_p95_s']} s |",
        f"| Tokens moyens par question | {summary['tokens_mean']} |",
    ]
    if agreement:
        lines.append(
            f"| Accord juge / règles (réponses chiffrées) | {_pct(agreement['agreement'])}, "
            f"kappa {agreement['kappa']} (n = {agreement['n']}) |"
        )
    lines += ["", "| Type de question | Exactitude |", "|---|---|"]
    lines += [f"| {k} | {_pct(v)} |" for k, v in summary["accuracy_by_type"].items()]
    return "\n".join(lines) + "\n"


def score_run(run_dir: Path, use_judge: bool = True) -> dict[str, Any]:
    """Note une exécution avec le juge enregistré dans son run.json, pas celui du TOML actuel."""
    meta = load_run_meta(run_dir)
    judge_config = JudgeConfig.model_validate(meta["config"]["judge"])
    records = list(latest_records(run_dir / PREDICTIONS_FILE).values())
    with ExitStack() as stack:
        judge = build_judge(judge_config, get_settings(), stack) if use_judge else None
        scores = [score_record(r, judge) for r in records]

    summary = summarize(scores, meta["questions_expected"])
    summary.update(
        run=run_dir.name,
        generation_fingerprint=meta["generation_fingerprint"],
        git=meta["git"],
        judge={"model": judge_config.model, "prompt_sha": JUDGE_PROMPT_SHA} if judge else None,
    )
    with (run_dir / "scores.jsonl").open("w", encoding="utf-8") as out:
        for s in scores:
            out.write(json.dumps(s, ensure_ascii=False) + "\n")
    write_json(run_dir / "summary.json", summary)
    (run_dir / "summary.md").write_text(summary_markdown(run_dir.name, summary), encoding="utf-8")
    return summary
