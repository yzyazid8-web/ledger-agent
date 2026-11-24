"""Observabilité en production (Langfuse) : chaque question posée à l'API devient une trace."""

import hashlib
import logging
import secrets
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

from ledger_agent.config import Settings
from ledger_agent.llm.types import CompletionParams, LLMClient, LLMResponse, Message

logger = logging.getLogger(__name__)

# Les prompts complets sont lourds : seuls les derniers messages sont envoyés avec chaque appel
MAX_MESSAGES_SENT = 6


class GenerationRecord(Protocol):
    def done(self, response: LLMResponse) -> None: ...

    def failed(self, error: Exception) -> None: ...


@dataclass
class QuestionTrace:
    """Trace d'une question en cours : callbacks à transmettre à l'agent, puis bilan final."""

    callbacks: list[Any] = field(default_factory=list)
    _span: Any = None
    _client: Any = None

    def finish(self, output: Any, scores: dict[str, float | str]) -> None:
        if self._span is None:
            return
        try:
            self._span.update(output=output)
            for name, value in scores.items():
                if isinstance(value, str):
                    self._client.score_current_trace(name=name, value=value)
                else:
                    self._client.score_current_trace(name=name, value=float(value))
        except Exception:
            logger.warning("Langfuse : bilan de la question non enregistré", exc_info=True)


class Tracer(Protocol):
    def question(
        self, question: str, session_id: str | None, metadata: dict[str, str]
    ) -> AbstractContextManager[QuestionTrace]: ...

    def generation(
        self, model: str, messages: list[Message], params: CompletionParams
    ) -> AbstractContextManager[GenerationRecord]: ...

    def flush(self) -> None: ...


class _NoRecord:
    def done(self, response: LLMResponse) -> None:
        return None

    def failed(self, error: Exception) -> None:
        return None


class NullTracer:
    """Aucun suivi : comportement par défaut, en évaluation comme dans les tests."""

    def question(
        self, question: str, session_id: str | None, metadata: dict[str, str]
    ) -> AbstractContextManager[QuestionTrace]:
        return nullcontext(QuestionTrace())

    def generation(
        self, model: str, messages: list[Message], params: CompletionParams
    ) -> AbstractContextManager[GenerationRecord]:
        return nullcontext(_NoRecord())

    def flush(self) -> None:
        return None


class _GenerationRecord:
    def __init__(self, observation: Any) -> None:
        self._observation = observation

    def done(self, response: LLMResponse) -> None:
        try:
            self._observation.update(
                output={"text": response.text, "tool_calls": response.tool_calls},
                usage_details={
                    "input": response.prompt_tokens,
                    "output": response.completion_tokens,
                },
                metadata={"cached": response.cached, **response.extra},
            )
        except Exception:
            logger.warning("Langfuse : appel au modèle non enregistré", exc_info=True)

    def failed(self, error: Exception) -> None:
        try:
            self._observation.update(level="ERROR", status_message=str(error)[:500])
        except Exception:
            logger.warning("Langfuse : erreur du modèle non enregistrée", exc_info=True)


@contextmanager
def _guarded(manager_factory: Any, fallback: Any) -> Iterator[Any]:
    """Ouvre un contexte Langfuse ; s'il échoue, le code continue avec un substitut inerte."""
    try:
        manager = manager_factory()
        value = manager.__enter__()
    except Exception:
        logger.warning("Langfuse indisponible : suivi désactivé pour cet appel", exc_info=True)
        yield fallback
        return
    try:
        yield value
    except BaseException as exc:
        manager.__exit__(type(exc), exc, exc.__traceback__)
        raise
    else:
        manager.__exit__(None, None, None)


class LangfuseTracer:
    """Envoie les traces à Langfuse, en arrière-plan et par lots (SDK OpenTelemetry)."""

    def __init__(self, settings: Settings) -> None:
        from langfuse import Langfuse  # importé seulement si le suivi est activé

        secret = settings.langfuse_secret_key
        self._public_key = settings.langfuse_public_key
        self._client = Langfuse(
            public_key=self._public_key,
            secret_key=secret.get_secret_value() if secret else None,
            base_url=settings.langfuse_host,
            sample_rate=settings.langfuse_sample_rate,
            environment="production",
        )

    @contextmanager
    def question(
        self, question: str, session_id: str | None, metadata: dict[str, str]
    ) -> Iterator[QuestionTrace]:
        from langfuse import propagate_attributes
        from langfuse.langchain import CallbackHandler

        def open_trace() -> Any:
            @contextmanager
            def both() -> Iterator[Any]:
                with (
                    self._client.start_as_current_observation(
                        name="question", as_type="agent", input=question
                    ) as span,
                    propagate_attributes(
                        trace_name="question", session_id=session_id, metadata=metadata
                    ),
                ):
                    yield span

            return both()

        with _guarded(open_trace, None) as span:
            if span is None:
                yield QuestionTrace()
                return
            try:
                # Rattaché à l'observation courante : les étapes du graphe s'y imbriquent
                callbacks: list[Any] = [CallbackHandler(public_key=self._public_key)]
            except Exception:
                logger.warning("Langfuse : étapes de l'agent non suivies", exc_info=True)
                callbacks = []
            yield QuestionTrace(callbacks=callbacks, _span=span, _client=self._client)

    @contextmanager
    def generation(
        self, model: str, messages: list[Message], params: CompletionParams
    ) -> Iterator[GenerationRecord]:
        def open_generation() -> Any:
            return self._client.start_as_current_observation(
                name="llm",
                as_type="generation",
                model=model,
                input=messages[-MAX_MESSAGES_SENT:],
                model_parameters={
                    "max_tokens": params.max_tokens,
                    "temperature": params.temperature,
                    "tool_choice": params.tool_choice if params.tools else "none",
                },
            )

        with _guarded(open_generation, None) as observation:
            yield _NoRecord() if observation is None else _GenerationRecord(observation)

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception:
            logger.warning("Langfuse : envoi final des traces impossible", exc_info=True)


def build_tracer(settings: Settings) -> Tracer:
    """Langfuse si les deux clés sont définies, sinon aucun suivi."""
    secret = settings.langfuse_secret_key
    if not settings.langfuse_public_key or not (secret and secret.get_secret_value()):
        return NullTracer()
    try:
        return LangfuseTracer(settings)
    except Exception:
        logger.warning("Langfuse : initialisation impossible, suivi désactivé", exc_info=True)
        return NullTracer()


class TracedLLM:
    """Client LLM qui déclare chacun de ses appels au tracer.

    Le projet utilise son propre client HTTP, pas les classes LLM de LangChain : le
    CallbackHandler de Langfuse ne voit donc pas ces appels, d'où cet enrobage.
    """

    def __init__(self, inner: LLMClient, tracer: Tracer) -> None:
        self._inner = inner
        self._tracer = tracer

    @property
    def model(self) -> str:
        return self._inner.model

    def complete(self, messages: list[Message], params: CompletionParams) -> LLMResponse:
        with self._tracer.generation(self._inner.model, messages, params) as record:
            try:
                response = self._inner.complete(messages, params)
            except Exception as exc:
                record.failed(exc)
                raise
            record.done(response)
            return response


# Sel aléatoire propre à ce processus : sans lui, une adresse IP se retrouverait par force brute
_SESSION_SALT = secrets.token_hex(16)


def anonymous_session(visitor: str | None, today: date | None = None) -> str | None:
    """Identifiant de session anonyme : regroupe les questions d'un visiteur sur une journée.

    L'adresse du visiteur n'est jamais envoyée : seul un hachage salé, qui change chaque jour
    et à chaque redémarrage de l'API, permet de regrouper ses questions.
    """
    if not visitor:
        return None
    day = (today or date.today()).isoformat()
    raw = f"{_SESSION_SALT}:{day}:{visitor}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def question_scores(events: list[dict[str, Any]]) -> tuple[Any, dict[str, float | str]]:
    """Réponse finale (ou erreur) et notes de la question, à partir des événements diffusés."""
    answer = next((e for e in reversed(events) if e.get("type") == "answer"), None)
    error = next((e for e in reversed(events) if e.get("type") == "error"), None)
    if answer is None:
        code = str(error.get("code", "inconnue")) if error else "sans_reponse"
        return {"erreur": code}, {"erreur": code}
    steps = [e for e in events if e.get("type") == "step"]
    scores: dict[str, float | str] = {
        "abstention": 1.0 if answer.get("abstained") else 0.0,
        "tours": float(sum(1 for e in steps if e.get("node") == "agent")),
        "tokens": float(answer.get("tokens", 0)),
        # Réponses renvoyées à l'agent par le vérificateur avant d'être acceptées ou abandonnées
        "refus_verificateur": float(
            sum(1 for e in steps if e.get("node") == "verify" and e.get("ok") is False)
        ),
    }
    output = {"final_answer": answer.get("final_answer"), "abstained": answer.get("abstained")}
    return output, scores
