"""Ligne de commande d'évaluation.

Utilisation :
    python -m ledger_agent.evaluation run naive --split dev [--limit 10]
    python -m ledger_agent.evaluation score naive --split dev [--no-judge]
"""

import argparse
import json
import logging

from ledger_agent.config import get_settings
from ledger_agent.evaluation.config import load_run_config
from ledger_agent.evaluation.runner import run_dir_for, run_predictions, score_run


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m ledger_agent.evaluation")
    sub = parser.add_subparsers(dest="command", required=True)

    for command, help_text in (
        ("run", "Génère les réponses puis calcule les métriques"),
        ("score", "Recalcule les métriques d'une exécution existante"),
    ):
        p = sub.add_parser(command, help=help_text)
        p.add_argument("config", help="Nom de configuration (ex. naive) ou chemin .toml")
        p.add_argument("--split", choices=["dev", "test", "all"], default="dev")
        p.add_argument("--no-judge", action="store_true", help="N'appelle pas le juge LLM")
        if command == "run":
            p.add_argument("--limit", type=int, help="Ne traite que les N premières questions")

    args = parser.parse_args()
    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Les journaux HTTP ligne par ligne noient la progression
    logging.getLogger("httpx").setLevel(logging.WARNING)

    config = load_run_config(args.config)
    if args.command == "run":
        run_dir = run_predictions(config, args.split, args.limit)
    else:
        run_dir = run_dir_for(config, args.split)

    summary = score_run(run_dir, config, use_judge=not args.no_judge)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
