"""Configuration d'une évaluation, lue et validée depuis evaluation/configs/<nom>.toml."""

import hashlib
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ledger_agent.config import REPO_ROOT
from ledger_agent.llm.types import CompletionParams

EVAL_DIR = REPO_ROOT / "evaluation"


class _Strict(BaseModel):
    # Une clé mal orthographiée dans le TOML est une erreur, pas une option ignorée
    model_config = ConfigDict(extra="forbid", frozen=True)


class LLMConfig(_Strict):
    provider: Literal["groq"]
    model: str
    max_tokens: int = Field(default=1024, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    tokens_per_minute: int | None = Field(default=None, gt=0)

    def params(self) -> CompletionParams:
        return CompletionParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
        )


class JudgeConfig(_Strict):
    provider: Literal["gemini"]
    model: str
    requests_per_minute: float = Field(default=8.0, gt=0)


class RetrievalConfig(_Strict):
    top_k: int = Field(default=5, gt=0)
    # "dense" : vecteurs seuls ; "hybrid" : vecteurs + BM25 fusionnés par RRF
    mode: Literal["dense", "hybrid"] = "dense"
    # "none" : aucun filtre ; "rules" : entreprise et exercice repérés dans la question
    filters: Literal["none", "rules"] = "none"
    lexical_weight: float = Field(default=1.0, ge=0.0)
    candidates: int = Field(default=30, gt=0)
    # Nom d'un cross-encoder fastembed, ou None pour ne pas reclasser
    reranker: str | None = None


class RunConfig(_Strict):
    name: str = Field(pattern=r"^[a-z0-9_]+$")
    pipeline: Literal["naive_rag"]
    retrieval: RetrievalConfig = RetrievalConfig()
    llm: LLMConfig
    judge: JudgeConfig

    def generation_fingerprint(self, *parts: str) -> str:
        """Empreinte de tout ce qui influence les réponses (le juge n'en fait pas partie)."""
        raw = self.model_dump_json(include={"pipeline", "retrieval", "llm"}) + "".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_run_config(path_or_name: str | Path) -> RunConfig:
    """Accepte un chemin de fichier ou un simple nom (« naive » → configs/naive.toml)."""
    path = Path(path_or_name)
    if not path.suffix:
        path = EVAL_DIR / "configs" / f"{path_or_name}.toml"
    return RunConfig.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
