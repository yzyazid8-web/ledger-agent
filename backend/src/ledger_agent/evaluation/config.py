"""Configuration d'une évaluation, lue depuis evaluation/configs/<nom>.toml."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from ledger_agent.config import REPO_ROOT
from ledger_agent.llm.types import CompletionParams

EVAL_DIR = REPO_ROOT / "evaluation"


@dataclass(frozen=True, slots=True)
class LLMConfig:
    provider: str
    model: str
    max_tokens: int = 1024
    temperature: float = 0.0
    reasoning_effort: str | None = None
    tokens_per_minute: int | None = None

    def params(self) -> CompletionParams:
        return CompletionParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
        )


@dataclass(frozen=True, slots=True)
class JudgeConfig:
    provider: str
    model: str
    requests_per_minute: float = 8.0


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    top_k: int = 5
    # "dense" : vecteurs seuls ; "hybrid" : vecteurs + BM25 fusionnés par RRF
    mode: str = "dense"
    # "none" : aucun filtre ; "rules" : entreprise et exercice repérés dans la question
    filters: str = "none"
    lexical_weight: float = 1.0
    candidates: int = 30
    # Nom d'un cross-encoder fastembed, ou None pour ne pas reclasser
    reranker: str | None = None


@dataclass(frozen=True, slots=True)
class RunConfig:
    name: str
    pipeline: str
    retrieval: RetrievalConfig
    llm: LLMConfig
    judge: JudgeConfig


def load_run_config(path_or_name: str | Path) -> RunConfig:
    """Accepte un chemin de fichier ou un simple nom (« naive » → configs/naive.toml)."""
    path = Path(path_or_name)
    if not path.suffix:
        path = EVAL_DIR / "configs" / f"{path_or_name}.toml"
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return RunConfig(
        name=raw["name"],
        pipeline=raw["pipeline"],
        retrieval=RetrievalConfig(**raw.get("retrieval", {})),
        llm=LLMConfig(**raw["llm"]),
        judge=JudgeConfig(**raw["judge"]),
    )
