"""Comparaison appariée de deux exécutions sur les mêmes questions (test de McNemar)."""

from pathlib import Path
from typing import Any

from ledger_agent.evaluation.config import EVAL_DIR
from ledger_agent.evaluation.dataset import Split
from ledger_agent.evaluation.runs import read_jsonl, run_dir_for, write_json
from ledger_agent.evaluation.stats import compare_paired

SIGNIFICANCE = 0.05


class ComparisonError(ValueError):
    """Les deux exécutions ne portent pas sur les mêmes questions."""


def _scores(name: str, split: Split) -> dict[str, bool]:
    path = run_dir_for(name, split) / "scores.jsonl"
    if not path.exists():
        raise ComparisonError(f"{path} introuvable : lance d'abord le calcul des scores")
    return {s["id"]: s["correct"] is True for s in read_jsonl(path)}


def compare_runs(name_a: str, name_b: str, split: Split) -> dict[str, Any]:
    a, b = _scores(name_a, split), _scores(name_b, split)
    if set(a) != set(b):
        raise ComparisonError(
            f"{name_a} ({len(a)} questions) et {name_b} ({len(b)} questions) ne portent pas "
            "sur les mêmes questions : termine les deux exécutions avant de comparer"
        )
    ids = sorted(a)
    result = compare_paired([a[i] for i in ids], [b[i] for i in ids])
    report: dict[str, Any] = {
        "a": name_a,
        "b": name_b,
        "split": split,
        "questions": result.n,
        "accuracy_a": round(result.accuracy_a, 4),
        "accuracy_b": round(result.accuracy_b, 4),
        "difference": round(result.difference, 4),
        "difference_ci95": [round(x, 4) for x in result.diff_ci],
        "won_by_b": result.only_b,
        "won_by_a": result.only_a,
        "mcnemar_p": round(result.p_value, 4),
        "significant": result.p_value < SIGNIFICANCE,
    }
    out = EVAL_DIR / "reports" / f"compare-{name_a}-vs-{name_b}-{split}"
    write_json(out.with_suffix(".json"), report)
    out.with_suffix(".md").write_text(comparison_markdown(report), encoding="utf-8")
    return report


def comparison_markdown(r: dict[str, Any]) -> str:
    low, high = r["difference_ci95"]
    verdict = (
        "La différence est **statistiquement significative** (p < 0,05)."
        if r["significant"]
        else "La différence **n'est pas statistiquement significative** (p ≥ 0,05) : "
        "elle peut s'expliquer par le hasard sur cet échantillon."
    )
    return (
        f"# Comparaison : {r['a']} contre {r['b']} ({r['split']})\n\n"
        f"| | {r['a']} | {r['b']} |\n|---|---|---|\n"
        f"| Exactitude | {r['accuracy_a']:.1%} | {r['accuracy_b']:.1%} |\n\n"
        f"- Questions : {r['questions']} (les mêmes pour les deux)\n"
        f"- Écart : {r['difference']:+.1%} (IC 95 % : {low:+.1%} à {high:+.1%})\n"
        f"- Questions gagnées par {r['b']} : {r['won_by_b']} ; par {r['a']} : {r['won_by_a']}\n"
        f"- Test de McNemar exact : p = {r['mcnemar_p']}\n\n{verdict}\n"
    )


def report_path(name_a: str, name_b: str, split: Split) -> Path:
    return EVAL_DIR / "reports" / f"compare-{name_a}-vs-{name_b}-{split}.md"
