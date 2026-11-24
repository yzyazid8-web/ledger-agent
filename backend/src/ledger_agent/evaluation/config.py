"""Configuration d'une évaluation : réexporte `ledger_agent.run_config` et ajoute EVAL_DIR."""

from ledger_agent.config import REPO_ROOT
from ledger_agent.run_config import (
    CONFIGS_DIR,
    AgentSettings,
    JudgeConfig,
    LLMConfig,
    RetrievalConfig,
    RunConfig,
    load_run_config,
)

EVAL_DIR = REPO_ROOT / "evaluation"

__all__ = [
    "CONFIGS_DIR",
    "EVAL_DIR",
    "AgentSettings",
    "JudgeConfig",
    "LLMConfig",
    "RetrievalConfig",
    "RunConfig",
    "load_run_config",
]
