"""Configuration d'une exécution : recherche, modèle, agent, juge (fichiers evaluation/configs).

Partagée par l'API et par l'évaluation : le service servi aux utilisateurs ne dépend ainsi pas
du code du banc d'évaluation.
"""

import hashlib
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ledger_agent.config import REPO_ROOT
from ledger_agent.llm.types import CompletionParams

# Les fichiers de configuration sont versionnés avec le banc d'évaluation
CONFIGS_DIR = REPO_ROOT / "evaluation" / "configs"


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


class AgentSettings(_Strict):
    # "react" : le LLM choisit ses outils et lit leurs résultats ;
    # "workflow" : graphe d'étapes fixes, le LLM ne décide qu'à l'intérieur de chaque étape
    mode: Literal["react", "workflow"] = "react"
    # Petit modèle pour planifier et juger la suffisance (mode workflow seulement)
    planner_llm: LLMConfig | None = None
    use_xbrl: bool = True
    use_grader: bool = True
    use_verifier: bool = True
    max_steps: int = Field(default=8, ge=2, le=20)
    max_search_rounds: int = Field(default=2, ge=1, le=4)
    passages_per_query: int = Field(default=3, ge=1, le=10)
    max_passages: int = Field(default=8, ge=1, le=20)
    max_synthesis_attempts: int = Field(default=2, ge=1, le=3)

    @model_validator(mode="after")
    def _workflow_needs_planner(self) -> "AgentSettings":
        if self.mode == "workflow" and self.planner_llm is None:
            raise ValueError("le mode « workflow » exige une section [agent.planner_llm]")
        return self


class RunConfig(_Strict):
    name: str = Field(pattern=r"^[a-z0-9_]+$")
    pipeline: Literal["naive_rag", "agent"]
    retrieval: RetrievalConfig = RetrievalConfig()
    # Pour l'agent, `llm` est le modèle qui rédige la réponse finale
    llm: LLMConfig
    judge: JudgeConfig
    agent: AgentSettings | None = None

    @model_validator(mode="after")
    def _agent_needs_settings(self) -> "RunConfig":
        if self.pipeline == "agent" and self.agent is None:
            raise ValueError("une configuration « agent » exige une section [agent]")
        return self

    def generation_fingerprint(self, *parts: str) -> str:
        """Empreinte de tout ce qui influence les réponses (le juge n'en fait pas partie)."""
        # Une section optionnelle absente ne doit pas changer l'empreinte des configs existantes
        fields = {"pipeline", "retrieval", "llm"} | ({"agent"} if self.agent else set())
        raw = self.model_dump_json(include=fields) + "".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_run_config(path_or_name: str | Path) -> RunConfig:
    """Accepte un chemin de fichier ou un simple nom (« naive » → configs/naive.toml)."""
    path = Path(path_or_name)
    if not path.suffix:
        path = CONFIGS_DIR / f"{path_or_name}.toml"
    return RunConfig.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
