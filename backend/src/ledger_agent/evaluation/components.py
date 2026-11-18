"""Construction des composants d'une évaluation à partir de sa configuration."""

import logging
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager

from qdrant_client import QdrantClient

from ledger_agent.baselines.naive_rag import NaiveRAG
from ledger_agent.config import Settings
from ledger_agent.evaluation.config import JudgeConfig, LLMConfig, RetrievalConfig, RunConfig
from ledger_agent.evaluation.judge import LLMJudge
from ledger_agent.ingestion.indexer import FastEmbedEmbedder
from ledger_agent.llm.cache import CachedLLM
from ledger_agent.llm.gemini import GeminiClient
from ledger_agent.llm.openai_compatible import OpenAICompatibleClient
from ledger_agent.llm.types import LLMClient
from ledger_agent.retrieval.dense import DenseRetriever
from ledger_agent.retrieval.hybrid import AutoFilterRetriever, HybridRetriever
from ledger_agent.retrieval.lexical import LexicalRetriever
from ledger_agent.retrieval.reranker import CrossEncoderReranker
from ledger_agent.retrieval.store import ChunkStore
from ledger_agent.retrieval.types import Retriever

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


@contextmanager
def build_pipeline(config: RunConfig, settings: Settings) -> Iterator[NaiveRAG]:
    with ExitStack() as stack:
        retriever = build_retriever(config.retrieval, settings, stack)
        llm = build_llm(config.llm, settings, stack)
        yield NaiveRAG(retriever, llm, top_k=config.retrieval.top_k, params=config.llm.params())
