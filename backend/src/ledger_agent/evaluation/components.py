"""Construction des composants d'une évaluation à partir de sa configuration."""

import logging
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager

from qdrant_client import QdrantClient

from ledger_agent.agent.graph import FinancialAgent
from ledger_agent.agent.nodes import AgentConfig, AgentRuntime
from ledger_agent.agent.prompts import prompts_sha
from ledger_agent.baselines.naive_rag import PROMPT_SHA, NaiveRAG
from ledger_agent.config import Settings
from ledger_agent.evaluation.config import JudgeConfig, LLMConfig, RetrievalConfig, RunConfig
from ledger_agent.evaluation.judge import LLMJudge
from ledger_agent.ingestion.indexer import FastEmbedEmbedder
from ledger_agent.llm.cache import CachedLLM
from ledger_agent.llm.gemini import GeminiClient
from ledger_agent.llm.openai_compatible import OpenAICompatibleClient
from ledger_agent.llm.types import LLMClient
from ledger_agent.pipeline import Pipeline
from ledger_agent.retrieval.dense import DenseRetriever
from ledger_agent.retrieval.hybrid import AutoFilterRetriever, HybridRetriever
from ledger_agent.retrieval.lexical import LexicalRetriever
from ledger_agent.retrieval.reranker import CrossEncoderReranker
from ledger_agent.retrieval.store import ChunkStore
from ledger_agent.retrieval.types import Retriever
from ledger_agent.tools.financial_facts import FactStore
from ledger_agent.tools.search_filings import SearchFilingsTool

logger = logging.getLogger(__name__)


def build_llm(config: LLMConfig, settings: Settings, stack: ExitStack) -> LLMClient:
    key = settings.groq_api_key.get_secret_value() if settings.groq_api_key else ""
    client = OpenAICompatibleClient(key, config.model, tokens_per_minute=config.tokens_per_minute)
    stack.callback(client.close)
    cached = CachedLLM(client, settings.data_dir / "cache" / "llm.sqlite")
    stack.callback(cached.close)
    return cached


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


def build_retriever(config: RetrievalConfig, settings: Settings, stack: ExitStack) -> Retriever:
    store = ChunkStore.from_parquet(settings.data_dir / "processed" / "chunks.parquet")
    qdrant = QdrantClient(path=str(settings.data_dir / "indexes" / "qdrant"))
    stack.callback(qdrant.close)
    models_dir = settings.data_dir / "models"
    retriever: Retriever = DenseRetriever(qdrant, FastEmbedEmbedder(cache_dir=models_dir), store)

    lexical = (
        LexicalRetriever.load(settings.data_dir / "indexes" / "bm25", store)
        if config.mode == "hybrid"
        else None
    )
    reranker = CrossEncoderReranker(config.reranker, models_dir) if config.reranker else None
    if lexical is not None or reranker is not None:
        retriever = HybridRetriever(
            retriever,
            lexical,
            reranker,
            candidates=config.candidates,
            lexical_weight=config.lexical_weight,
        )
    if config.filters == "rules":
        retriever = AutoFilterRetriever(retriever)
    return retriever


def pipeline_prompt_sha(config: RunConfig) -> str:
    return prompts_sha() if config.pipeline == "agent" else PROMPT_SHA


def build_agent(
    config: RunConfig, settings: Settings, retriever: Retriever, stack: ExitStack
) -> FinancialAgent:
    agent = config.agent
    if agent is None:
        raise ValueError("section [agent] manquante")
    facts = None
    if agent.use_xbrl:
        facts = FactStore.open(settings.data_dir / "processed" / "facts.duckdb")
        stack.callback(facts.close)
    runtime = AgentRuntime(
        planner_llm=build_llm(agent.planner_llm, settings, stack),
        writer_llm=build_llm(config.llm, settings, stack),
        search=SearchFilingsTool(retriever),
        facts=facts,
        config=AgentConfig(
            use_xbrl=agent.use_xbrl,
            use_grader=agent.use_grader,
            use_verifier=agent.use_verifier,
            max_search_rounds=agent.max_search_rounds,
            passages_per_query=agent.passages_per_query,
            max_passages=agent.max_passages,
            max_synthesis_attempts=agent.max_synthesis_attempts,
        ),
        planner_params=agent.planner_llm.params(),
        writer_params=config.llm.params(),
    )
    return FinancialAgent(runtime)


@contextmanager
def build_pipeline(config: RunConfig, settings: Settings) -> Iterator[Pipeline]:
    with ExitStack() as stack:
        retriever = build_retriever(config.retrieval, settings, stack)
        if config.pipeline == "agent":
            yield build_agent(config, settings, retriever, stack)
            return
        llm = build_llm(config.llm, settings, stack)
        yield NaiveRAG(retriever, llm, top_k=config.retrieval.top_k, params=config.llm.params())
