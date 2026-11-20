"""Tableau d'ablation : toutes les configurations d'un jeu, et leurs comparaisons appariées."""

import json
from pathlib import Path
from typing import Any

from ledger_agent.evaluation.compare import SIGNIFICANCE
from ledger_agent.evaluation.config import EVAL_DIR
from ledger_agent.evaluation.dataset import Split
from ledger_agent.evaluation.runs import read_jsonl, write_json
from ledger_agent.evaluation.stats import compare_paired

# Ordre de lecture : de la référence la plus simple à l'agent complet
ORDER = (
    "naive",
    "hybrid",
    "hybrid_rerank",
    "workflow",
    "agent_sans_xbrl",
    "agent_sans_verif",
    "agent",
)
# Comparaisons qui répondent aux questions de l'étude : (référence, configuration comparée)
COMPARISONS = (
    ("naive", "hybrid"),
    ("hybrid", "workflow"),
    ("hybrid", "agent"),
    ("workflow", "agent"),
    ("naive", "agent"),
    ("agent_sans_xbrl", "agent"),
    ("agent_sans_verif", "agent"),
)


def _runs(split: Split) -> dict[str, Path]:
    found = {
        path.parent.name.removesuffix(f"-{split}"): path.parent
        for path in (EVAL_DIR / "runs").glob(f"*-{split}/summary.json")
    }
    ordered = [n for n in ORDER if n in found] + sorted(set(found) - set(ORDER))
    return {name: found[name] for name in ordered}


def _scores(run_dir: Path) -> dict[str, bool]:
    return {s["id"]: s["correct"] is True for s in read_jsonl(run_dir / "scores.jsonl")}


def build_report(split: Split) -> dict[str, Any]:
    runs = _runs(split)
    rows = []
    for name, run_dir in runs.items():
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        rows.append({"config": name, **{k: summary.get(k) for k in _COLUMNS}})

    comparisons: list[dict[str, Any]] = []
    for a, b in COMPARISONS:
        if a not in runs or b not in runs:
            continue
        scores_a, scores_b = _scores(runs[a]), _scores(runs[b])
        complete = all(r["complete"] for r in rows if r["config"] in (a, b))
        if not complete or set(scores_a) != set(scores_b):
            comparisons.append({"a": a, "b": b, "status": "en attente : exécutions incomplètes"})
            continue
        ids = sorted(scores_a)
        result = compare_paired([scores_a[i] for i in ids], [scores_b[i] for i in ids])
        comparisons.append(
            {
                "a": a,
                "b": b,
                "status": "ok",
                "difference": round(result.difference, 4),
                "difference_ci95": [round(x, 4) for x in result.diff_ci],
                "won_by_b": result.only_b,
                "won_by_a": result.only_a,
                "mcnemar_p": round(result.p_value, 4),
                "significant": result.p_value < SIGNIFICANCE,
            }
        )

    report = {"split": split, "rows": rows, "comparisons": comparisons}
    out = EVAL_DIR / "reports" / f"ablation-{split}"
    write_json(out.with_suffix(".json"), report)
    out.with_suffix(".md").write_text(report_markdown(report), encoding="utf-8")
    return report


MISSING = "n/a"

_COLUMNS = (
    "questions",
    "questions_expected",
    "complete",
    "accuracy",
    "accuracy_ci95",
    "wrong_answer_rate",
    "abstention_rate",
    "page_hit_rate",
    "tokens_mean",
    "latency_p50_s",
)


def _pct(value: float | None) -> str:
    return MISSING if value is None else f"{value:.0%}"


def report_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Étude d'ablation ({report['split']})",
        "",
        "| Configuration | Questions | Exactitude (IC 95 %) | Réponses fausses | Abstention "
        "| Page retrouvée | Tokens / question | Latence p50 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in report["rows"]:
        ci = r["accuracy_ci95"]
        accuracy = f"{_pct(r['accuracy'])} ({_pct(ci[0])} à {_pct(ci[1])})" if ci else MISSING
        done = f"{r['questions']}/{r['questions_expected']}" + ("" if r["complete"] else " ⚠️")
        lines.append(
            f"| `{r['config']}` | {done} | {accuracy} | {_pct(r['wrong_answer_rate'])} "
            f"| {_pct(r['abstention_rate'])} | {_pct(r['page_hit_rate'])} "
            f"| {r['tokens_mean'] or MISSING} | {r['latency_p50_s'] or MISSING} s |"
        )
    lines += [
        "",
        "⚠️ = exécution incomplète : ses chiffres ne sont pas comparables aux autres.",
        "",
        "## Comparaisons appariées (test de McNemar exact)",
        "",
        "| Comparaison | Écart d'exactitude (IC 95 %) | Gagnées / perdues | p | Significatif |",
        "|---|---|---|---|---|",
    ]
    for c in report["comparisons"]:
        name = f"`{c['a']}` → `{c['b']}`"
        if c["status"] != "ok":
            lines.append(f"| {name} | {c['status']} | | | |")
            continue
        low, high = c["difference_ci95"]
        lines.append(
            f"| {name} | {c['difference']:+.0%} ({low:+.0%} à {high:+.0%}) "
            f"| {c['won_by_b']} / {c['won_by_a']} | {c['mcnemar_p']} "
            f"| {'oui' if c['significant'] else 'non'} |"
        )
    return "\n".join(lines) + "\n"
