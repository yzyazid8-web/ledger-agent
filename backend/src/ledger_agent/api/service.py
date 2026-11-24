"""Service qui détient l'agent et ses ressources (index, base de faits, clients LLM)."""

import logging
import threading
from collections.abc import Iterator
from contextlib import ExitStack
from typing import Any, Protocol

from ledger_agent.assembly import build_agent, build_retriever
from ledger_agent.config import Settings
from ledger_agent.ingestion.universe import load_universe
from ledger_agent.llm.types import LLMError, LLMQuotaError
from ledger_agent.pipeline import StreamingPipeline
from ledger_agent.run_config import load_run_config

logger = logging.getLogger(__name__)


class AgentService(Protocol):
    @property
    def ready(self) -> bool: ...

    @property
    def config_name(self) -> str: ...

    def companies(self) -> list[dict[str, str]]: ...

    def stream(self, question: str) -> Iterator[dict[str, Any]]: ...

    def close(self) -> None: ...


class LocalAgentService:
    """Agent réel, chargé une fois au démarrage de l'API."""

    def __init__(self, settings: Settings) -> None:
        self._config = load_run_config(settings.api_config)
        if self._config.pipeline != "agent":
            raise ValueError(f"{settings.api_config} n'est pas une configuration d'agent")
        self._stack = ExitStack()
        retriever = build_retriever(self._config.retrieval, settings, self._stack)
        # Le service ne dépend d'aucune architecture d'agent précise, seulement de ce protocole
        self._agent: StreamingPipeline = build_agent(self._config, settings, retriever, self._stack)
        # Qdrant en mode local et DuckDB ne supportent pas les accès concurrents :
        # les questions sont traitées une à une (suffisant pour une démonstration)
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return True

    @property
    def config_name(self) -> str:
        return self._config.name

    def companies(self) -> list[dict[str, str]]:
        return [{"ticker": c.ticker, "name": c.name} for c in load_universe()]

    def stream(self, question: str) -> Iterator[dict[str, Any]]:
        with self._lock:
            try:
                yield from self._agent.stream(question)
            except LLMQuotaError as exc:
                logger.warning("Quota épuisé : %s", exc)
                yield {
                    "type": "error",
                    "code": "quota",
                    "message": "Quota gratuit du modèle atteint. Réessaie dans quelques minutes.",
                    "retry_after_s": exc.retry_after_s,
                }
            except LLMError as exc:
                logger.warning("Erreur du fournisseur LLM : %s", exc)
                yield {"type": "error", "code": "llm", "message": "Le modèle n'a pas répondu."}
            except Exception:
                # Sans cela, le client recevrait un flux vide, sans aucune explication
                logger.exception("Erreur inattendue pendant la réponse")
                yield {"type": "error", "code": "internal", "message": "Erreur interne de l'agent."}

    def close(self) -> None:
        self._stack.close()
