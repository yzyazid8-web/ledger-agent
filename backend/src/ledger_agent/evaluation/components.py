"""Composants propres à l'évaluation : juge, référence RAG classique, empreinte des prompts.

Les composants partagés avec l'API (modèle, recherche, agent) sont dans `ledger_agent.assembly`.
"""

import logging
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager

from ledger_agent.agent.prompts import agent_code_sha, prompts_sha
from ledger_agent.assembly import build_agent, build_llm, build_retriever
from ledger_agent.baselines.naive_rag import PROMPT_SHA, NaiveRAG
from ledger_agent.config import Settings
from ledger_agent.evaluation.config import JudgeConfig, RunConfig
from ledger_agent.evaluation.judge import LLMJudge
from ledger_agent.llm.cache import CachedLLM
from ledger_agent.llm.gemini import GeminiClient
from ledger_agent.pipeline import Pipeline

# Réexportés : le reste du code d'évaluation les importe d'ici
__all__ = [
    "build_agent",
    "build_judge",
    "build_llm",
    "build_pipeline",
    "build_retriever",
    "pipeline_prompt_sha",
]

logger = logging.getLogger(__name__)


def build_judge(config: JudgeConfig, settings: Settings, stack: ExitStack) -> LLMJudge | None:
    if settings.gemini_api_key is None or not settings.gemini_api_key.get_secret_value():
        logger.warning("GEMINI_API_KEY vide : les réponses rédigées resteront non jugées")
        return None
    client = GeminiClient(
        settings.gemini_api_key.get_secret_value(),
        config.model,
        requests_per_minute=config.requests_per_minute,
    )
    stack.callback(client.close)
    cached = CachedLLM(client, settings.data_dir / "cache" / "judge.sqlite")
    stack.callback(cached.close)
    return LLMJudge(cached)


def pipeline_prompt_sha(config: RunConfig) -> str:
    # Pour l'agent, l'empreinte couvre aussi son code : modifier le graphe ou les outils
    # interdit de reprendre une évaluation commencée avec l'ancienne version
    return f"{prompts_sha()}-{agent_code_sha()}" if config.pipeline == "agent" else PROMPT_SHA


@contextmanager
def build_pipeline(config: RunConfig, settings: Settings) -> Iterator[Pipeline]:
    with ExitStack() as stack:
        retriever = build_retriever(config.retrieval, settings, stack)
        if config.pipeline == "agent":
            yield build_agent(config, settings, retriever, stack)
            return
        llm = build_llm(config.llm, settings, stack)
        yield NaiveRAG(retriever, llm, top_k=config.retrieval.top_k, params=config.llm.params())
