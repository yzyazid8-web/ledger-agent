"""Construction des composants à partir d'une configuration : modèle, recherche, agent.

Utilisé par l'API (service en production) comme par l'évaluation, sans dépendre de cette dernière.
"""

from contextlib import ExitStack

from qdrant_client import QdrantClient

from ledger_agent.agent.graph import FinancialAgent
from ledger_agent.agent.nodes import AgentConfig, AgentRuntime
from ledger_agent.agent.react import ReactAgent, ReactConfig
from ledger_agent.config import Settings
from ledger_agent.ingestion.indexer import FastEmbedEmbedder
from ledger_agent.llm.cache import CachedLLM
from ledger_agent.llm.openai_compatible import OpenAICompatibleClient
from ledger_agent.llm.types import LLMClient
from ledger_agent.observability import TracedLLM, Tracer
from ledger_agent.retrieval.dense import DenseRetriever
from ledger_agent.retrieval.hybrid import AutoFilterRetriever, HybridRetriever
from ledger_agent.retrieval.lexical import LexicalRetriever
from ledger_agent.retrieval.reranker import CrossEncoderReranker
from ledger_agent.retrieval.store import ChunkStore
from ledger_agent.retrieval.types import Retriever
from ledger_agent.run_config import LLMConfig, RetrievalConfig, RunConfig
from ledger_agent.tools.financial_facts import FactStore
from ledger_agent.tools.search_filings import SearchFilingsTool


def build_llm(
    config: LLMConfig, settings: Settings, stack: ExitStack, tracer: Tracer | None = None
) -> LLMClient:
    key = settings.groq_api_key.get_secret_value() if settings.groq_api_key else ""
    client = OpenAICompatibleClient(
        key,
        config.model,
        tokens_per_minute=config.tokens_per_minute,
        force_ipv4=settings.force_ipv4,
    )
    stack.callback(client.close)
    cached = CachedLLM(client, settings.data_dir / "cache" / "llm.sqlite")
    stack.callback(cached.close)
    # Le suivi enrobe le cache : les réponses servies par le cache sont aussi enregistrées
    return TracedLLM(cached, tracer) if tracer is not None else cached


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


def build_agent(
    config: RunConfig,
    settings: Settings,
    retriever: Retriever,
    stack: ExitStack,
    tracer: Tracer | None = None,
) -> FinancialAgent | ReactAgent:
    # `tracer` : suivi en production des appels au modèle ; l'évaluation n'en passe jamais
    agent = config.agent
    if agent is None:
        raise ValueError("section [agent] manquante")
    facts = None
    if agent.use_xbrl:
        facts = FactStore.open(settings.data_dir / "processed" / "facts.duckdb")
        stack.callback(facts.close)
    if agent.mode == "react":
        return ReactAgent(
            llm=build_llm(config.llm, settings, stack, tracer),
            search=SearchFilingsTool(retriever),
            facts=facts,
            config=ReactConfig(
                use_xbrl=agent.use_xbrl,
                use_verifier=agent.use_verifier,
                max_steps=agent.max_steps,
                passages_per_search=agent.passages_per_query,
                max_answer_attempts=agent.max_synthesis_attempts,
            ),
            params=config.llm.params(),
        )
    if agent.planner_llm is None:
        raise ValueError("le mode « workflow » exige [agent.planner_llm]")
    runtime = AgentRuntime(
        planner_llm=build_llm(agent.planner_llm, settings, stack, tracer),
        writer_llm=build_llm(config.llm, settings, stack, tracer),
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
