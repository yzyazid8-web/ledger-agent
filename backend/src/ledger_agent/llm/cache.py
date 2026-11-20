"""Cache disque des appels LLM (SQLite)."""

import hashlib
import json
import sqlite3
import threading
from dataclasses import asdict, replace
from pathlib import Path

from ledger_agent.llm.types import CompletionParams, LLMClient, LLMResponse, Message


def cache_key(model: str, messages: list[Message], params: CompletionParams) -> str:
    raw = json.dumps(
        {"model": model, "messages": messages, "params": asdict(params)},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class CachedLLM:
    def __init__(self, inner: LLMClient, path: Path) -> None:
        self._inner = inner
        path.parent.mkdir(parents=True, exist_ok=True)
        # L'API traite chaque requête dans un fil différent : la connexion est partagée
        # entre fils, et le verrou empêche deux accès simultanés
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS completions (key TEXT PRIMARY KEY, value TEXT)"
        )

    @property
    def model(self) -> str:
        return self._inner.model

    def close(self) -> None:
        self._db.close()

    def complete(self, messages: list[Message], params: CompletionParams) -> LLMResponse:
        key = cache_key(self.model, messages, params)
        with self._lock:
            row = self._db.execute("SELECT value FROM completions WHERE key = ?", (key,)).fetchone()
        if row is not None:
            return replace(LLMResponse(**json.loads(row[0])), cached=True)

        # L'appel au fournisseur se fait hors verrou : il peut durer plusieurs secondes
        response = self._inner.complete(messages, params)
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO completions VALUES (?, ?)",
                (key, json.dumps(asdict(response), ensure_ascii=False)),
            )
            self._db.commit()
        return response
