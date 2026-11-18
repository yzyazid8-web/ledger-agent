"""Manifestes d'ingestion : ce qui a été ingéré, depuis quelle source, avec quel hash."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(path: Path, kind: str, entries: list[dict[str, Any]], **extra: Any) -> None:
    manifest = {
        "kind": kind,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "entries": entries,
        **extra,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
