"""Snapshot des données servies par l'API : archive versionnée et vérifiée par empreinte.

Utilisation :
    python -m ledger_agent.snapshot build            # crée l'archive et data-snapshot.json
    python -m ledger_agent.snapshot extract ARCHIVE --dest DIR --sha256 EMPREINTE
"""

import argparse
import hashlib
import json
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path

from ledger_agent.config import REPO_ROOT, get_settings

# Seulement ce qu'il faut pour répondre : ni les PDF bruts, ni le cache, ni les modèles
INCLUDED = (
    "processed/facts.duckdb",
    "processed/chunks.parquet",
    "processed/facts_manifest.json",
    "processed/documents_manifest.json",
    "indexes/qdrant",
    "indexes/bm25",
)
LOCK_FILE = REPO_ROOT / "data-snapshot.json"


class SnapshotError(RuntimeError):
    """Archive absente, incomplète ou dont l'empreinte ne correspond pas."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def build(data_dir: Path, out_dir: Path, tag: str) -> dict[str, str]:
    missing = [p for p in INCLUDED if not (data_dir / p).exists()]
    if missing:
        raise SnapshotError(f"Données manquantes, relance l'ingestion : {missing}")
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / f"snapshot-{tag}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for relative in INCLUDED:
            # Le verrou de Qdrant n'a pas de sens hors de la machine qui l'a posé
            tar.add(
                data_dir / relative,
                arcname=relative,
                filter=lambda info: None if info.name.endswith(".lock") else info,
            )
    return {"tag": f"data-{tag}", "asset": archive.name, "sha256": sha256_of(archive)}


def extract(archive: Path, dest: Path, expected_sha256: str) -> None:
    if not archive.exists():
        raise SnapshotError(f"Archive introuvable : {archive}")
    actual = sha256_of(archive)
    if actual != expected_sha256:
        raise SnapshotError(f"Empreinte inattendue : {actual} au lieu de {expected_sha256}")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        # Le filtre « data » refuse les chemins absolus, les « .. » et les liens dangereux
        tar.extractall(dest, filter="data")
    missing = [p for p in INCLUDED if not (dest / p).exists()]
    if missing:
        raise SnapshotError(f"Archive incomplète : {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m ledger_agent.snapshot")
    sub = parser.add_subparsers(dest="command", required=True)
    build_cmd = sub.add_parser("build", help="Crée l'archive des données et data-snapshot.json")
    build_cmd.add_argument("--tag", default=datetime.now(UTC).strftime("%Y-%m"))
    build_cmd.add_argument("--out", type=Path, default=get_settings().data_dir / "snapshots")
    extract_cmd = sub.add_parser("extract", help="Vérifie l'empreinte puis extrait l'archive")
    extract_cmd.add_argument("archive", type=Path)
    extract_cmd.add_argument("--dest", type=Path, required=True)
    extract_cmd.add_argument("--sha256", required=True)
    args = parser.parse_args()

    try:
        if args.command == "build":
            lock = build(get_settings().data_dir, args.out, args.tag)
            LOCK_FILE.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({**lock, "path": str(args.out / lock["asset"])}, indent=2))
        else:
            extract(args.archive, args.dest, args.sha256)
            print(f"Données extraites dans {args.dest}")
    except SnapshotError as exc:
        sys.exit(f"Erreur : {exc}")


if __name__ == "__main__":
    main()
