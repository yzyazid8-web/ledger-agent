"""Ligne de commande d'évaluation.

Utilisation :
    python -m ledger_agent.evaluation run naive --split dev [--limit 10]
    python -m ledger_agent.evaluation score naive --split dev [--no-judge]
    python -m ledger_agent.evaluation retrieval hybrid --split dev
    python -m ledger_agent.evaluation compare naive hybrid --split dev
"""

import argparse
import json
import logging
import sys

from ledger_agent.config import get_settings
from ledger_agent.evaluation.compare import ComparisonError, compare_runs
from ledger_agent.evaluation.config import load_run_config
from ledger_agent.evaluation.dataset import SplitError
from ledger_agent.evaluation.predict import run_predictions
from ledger_agent.evaluation.retrieval_eval import evaluate_retrieval
from ledger_agent.evaluation.runs import RunMismatchError, run_dir_for
from ledger_agent.evaluation.scoring import score_run


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m ledger_agent.evaluation")
    sub = parser.add_subparsers(dest="command", required=True)

    for command, help_text in (
        ("run", "Génère les réponses puis calcule les métriques"),
        ("score", "Recalcule les métriques d'une exécution existante"),
        ("retrieval", "Mesure la recherche seule (hit@k), sans appel au LLM"),
    ):
        p = sub.add_parser(command, help=help_text)
        p.add_argument("config", help="Nom de configuration (ex. naive) ou chemin .toml")
        p.add_argument("--split", choices=["dev", "test"], default="dev")
        if command in ("run", "score"):
            p.add_argument("--no-judge", action="store_true", help="N'appelle pas le juge LLM")
        if command == "run":
            p.add_argument("--limit", type=int, help="Ne traite que les N premières questions")

    compare = sub.add_parser("compare", help="Compare deux exécutions (test de McNemar)")
    compare.add_argument("a", help="Configuration de référence")
    compare.add_argument("b", help="Configuration comparée")
    compare.add_argument("--split", choices=["dev", "test"], default="dev")

    args = parser.parse_args()
    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Les journaux HTTP ligne par ligne noient la progression
    logging.getLogger("httpx").setLevel(logging.WARNING)

    try:
        if args.command == "compare":
            summary = compare_runs(args.a, args.b, args.split)
        else:
            config = load_run_config(args.config)
            if args.command == "retrieval":
                summary = evaluate_retrieval(config, args.split)
            else:
                if args.command == "run":
                    run_dir = run_predictions(config, args.split, args.limit)
                else:
                    run_dir = run_dir_for(config.name, args.split)
                summary = score_run(run_dir, use_judge=not args.no_judge)
    except (RunMismatchError, ComparisonError, SplitError, FileNotFoundError) as exc:
        sys.exit(f"Erreur : {exc}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
