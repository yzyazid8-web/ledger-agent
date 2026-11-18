"""Dossiers d'exécution : fichiers, empreintes et métadonnées de reproductibilité."""

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ledger_agent.baselines.naive_rag import PROMPT_SHA
from ledger_agent.config import REPO_ROOT, Settings
from ledger_agent.evaluation.config import EVAL_DIR, RunConfig
from ledger_agent.evaluation.dataset import Split

RUN_FILE = "run.json"
PREDICTIONS_FILE = "predictions.jsonl"


class RunMismatchError(RuntimeError):
    """Reprise refusée : la configuration a changé depuis le début de l'exécution."""


def run_dir_for(name: str, split: Split) -> Path:
    return EVAL_DIR / "runs" / f"{name}-{split}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def latest_records(path: Path) -> dict[str, dict[str, Any]]:
    """Dernier enregistrement par question : une erreur relancée remplace la précédente."""
    return {record["id"]: record for record in read_jsonl(path)}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def corpus_fingerprint(settings: Settings) -> str:
    """Empreinte du corpus indexé, tirée des manifestes d'ingestion (hors dates)."""
    entries = []
    for name in ("documents_manifest.json", "facts_manifest.json"):
        path = settings.data_dir / "processed" / name
        if path.exists():
            entries.append(json.loads(path.read_text(encoding="utf-8")).get("entries", []))
    return _sha(entries) if entries else "inconnu"


def git_state() -> dict[str, Any]:
    """Commit courant et présence de modifications non commitées (lecture seule)."""

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    status = git("status", "--porcelain", "--", "backend/src")
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def prepare_run(
    config: RunConfig, split: Split, question_ids: list[str], settings: Settings, expected: int
) -> Path:
    """Crée ou reprend un dossier d'exécution, en refusant toute reprise incohérente."""
    run_dir = run_dir_for(config.name, split)
    corpus_sha = corpus_fingerprint(settings)
    split_sha = _sha(sorted(question_ids))
    fingerprint = config.generation_fingerprint(PROMPT_SHA, corpus_sha, split_sha)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    run_file = run_dir / RUN_FILE

    if run_file.exists():
        meta: dict[str, Any] = json.loads(run_file.read_text(encoding="utf-8"))
        if meta.get("generation_fingerprint") != fingerprint:
            raise RunMismatchError(
                f"La configuration, le prompt, le corpus ou le découpage ont changé depuis le "
                f"début de {run_dir.name}. Supprime {run_dir} ou change le nom de la config."
            )
    elif (run_dir / PREDICTIONS_FILE).exists():
        raise RunMismatchError(f"{run_dir} contient des prédictions sans {RUN_FILE} : supprime-le")
    else:
        meta = {
            "name": config.name,
            "split": split,
            "created_at": now,
            "config": config.model_dump(),
            "generation_fingerprint": fingerprint,
            "prompt_sha": PROMPT_SHA,
            "corpus_sha": corpus_sha,
            "split_sha": split_sha,
            "git": [],
        }

    # Chaque session de génération note le commit utilisé
    meta["git"].append({**git_state(), "at": now})
    meta["questions_expected"] = expected
    write_json(run_file, meta)
    return run_dir


def load_run_meta(run_dir: Path) -> dict[str, Any]:
    run_file = run_dir / RUN_FILE
    if not run_file.exists():
        raise FileNotFoundError(f"{run_file} introuvable : lance d'abord la génération")
    meta: dict[str, Any] = json.loads(run_file.read_text(encoding="utf-8"))
    return meta
