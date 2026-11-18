"""Reclassement des candidats par un cross-encoder."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

DEFAULT_RERANKER = "Xenova/ms-marco-MiniLM-L-12-v2"


class Reranker(Protocol):
    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER, cache_dir: Path | None = None) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name, cache_dir=str(cache_dir) if cache_dir else None)

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, list(passages), batch_size=16)]
