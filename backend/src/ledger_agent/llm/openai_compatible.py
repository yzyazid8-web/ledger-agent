"""Client pour les API compatibles OpenAI (Groq), sans SDK : un simple appel HTTP."""

import json
import logging
import time
from collections.abc import Callable
from typing import Any

import httpx

from ledger_agent.common.rate_limit import Reservation, TokenRateLimiter
from ledger_agent.llm.http_errors import daily_quota_wait, parse_seconds, sanitize
from ledger_agent.llm.types import (
    CompletionParams,
    LLMError,
    LLMQuotaError,
    LLMResponse,
    LLMToolCallError,
    Message,
    ToolCall,
)

logger = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
MAX_BACKOFF_SECONDS = 60.0
# Approximation classique pour l'anglais : 1 token ≈ 4 caractères
CHARS_PER_TOKEN = 4


MALFORMED_OUTPUT = ("tool_use_failed", "output_parse_failed")


def _tool_error_message(response: httpx.Response) -> str:
    # Groq valide les arguments contre le schéma de l'outil et renvoie la raison du refus
    try:
        message = str(response.json()["error"]["message"])
    except (ValueError, KeyError, TypeError):
        message = response.text
    return sanitize(message)[:500]


def estimate_tokens(
    messages: list[Message], max_tokens: int, tools: tuple[dict[str, Any], ...] = ()
) -> int:
    # Les appels d'outils et leurs définitions comptent aussi dans le quota
    prompt_chars = sum(
        len(m.get("content") or "") + len(json.dumps(m.get("tool_calls", []))) for m in messages
    ) + len(json.dumps(list(tools)))
    # On réserve une sortie typique plutôt que max_tokens, qui est rarement atteint
    return prompt_chars // CHARS_PER_TOKEN + min(max_tokens, 600)


class OpenAICompatibleClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = GROQ_BASE_URL,
        tokens_per_minute: int | None = 8000,
        max_retries: int = 6,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        force_ipv4: bool = False,
    ) -> None:
        if not api_key:
            raise ValueError(f"Clé d'API manquante pour le modèle {model}")
        if transport is None and force_ipv4:
            # Se lier à une adresse locale IPv4 oblige la connexion à passer par IPv4
            transport = httpx.HTTPTransport(local_address="0.0.0.0")
        self._model = model
        self._max_retries = max_retries
        self._sleep = sleep
        self._limiter = (
            TokenRateLimiter(tokens_per_minute, sleep=sleep) if tokens_per_minute else None
        )
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            transport=transport,
        )

    @property
    def model(self) -> str:
        return self._model

    def close(self) -> None:
        self._client.close()

    def _body(self, messages: list[Message], params: CompletionParams) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
        }
        if params.reasoning_effort:
            body["reasoning_effort"] = params.reasoning_effort
        if params.json_mode:
            body["response_format"] = {"type": "json_object"}
        if params.tools:
            body["tools"] = list(params.tools)
            body["tool_choice"] = params.tool_choice
        return body

    def _raise_if_quota_exhausted(self, response: httpx.Response) -> None:
        # Quota quotidien épuisé : attendre des minutes ne sert à rien, on s'arrête net
        if response.status_code != 429:
            return
        retry_after = response.headers.get("retry-after")
        wait = daily_quota_wait(response.text, retry_after)
        if wait is None:
            header_wait = parse_seconds(retry_after)
            wait = header_wait if header_wait and header_wait > MAX_BACKOFF_SECONDS else None
        if wait is not None:
            raise LLMQuotaError(
                f"Quota épuisé pour {self._model}, réessayer dans {wait / 60:.0f} min",
                retry_after_s=wait,
            )

    def complete(self, messages: list[Message], params: CompletionParams) -> LLMResponse:
        body = self._body(messages, params)
        for attempt in range(self._max_retries + 1):
            is_last_attempt = attempt == self._max_retries
            reservation = (
                self._limiter.acquire(estimate_tokens(messages, params.max_tokens, params.tools))
                if self._limiter
                else None
            )
            started = time.perf_counter()
            try:
                response = self._client.post("/chat/completions", json=body)
            except httpx.TransportError as exc:
                self._release(reservation, 0)  # la requête n'a pas atteint le fournisseur
                if is_last_attempt:
                    raise LLMError(f"Erreur réseau : {exc}") from exc
                self._backoff(attempt, None, type(exc).__name__)
                continue
            latency = time.perf_counter() - started

            if response.is_error:
                # Un appel refusé ne génère rien : seule l'estimation du prompt reste comptée.
                # Sinon la réservation complète bloquerait l'appel suivant près d'une minute
                self._release(reservation, estimate_tokens(messages, 0, params.tools))
            self._raise_if_quota_exhausted(response)
            if response.status_code in RETRYABLE_STATUS and not is_last_attempt:
                self._backoff(
                    attempt, response.headers.get("retry-after"), str(response.status_code)
                )
                continue
            # Sortie mal formée du modèle (appel d'outil invalide ou texte illisible) : il peut
            # se corriger au tour suivant, ce n'est pas une panne du fournisseur
            if response.status_code == 400 and any(c in response.text for c in MALFORMED_OUTPUT):
                raise LLMToolCallError(_tool_error_message(response))
            if response.is_error:
                raise LLMError(f"HTTP {response.status_code} : {sanitize(response.text)}")

            try:
                payload = response.json()
                choice = payload["choices"][0]
                text = (choice["message"].get("content") or "").strip()
                tool_calls: list[ToolCall] = [
                    {
                        "id": str(call["id"]),
                        "type": "function",
                        "function": {
                            "name": str(call["function"]["name"]),
                            "arguments": str(call["function"].get("arguments") or "{}"),
                        },
                    }
                    for call in choice["message"].get("tool_calls") or []
                ]
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                raise LLMError(f"Réponse inattendue : {sanitize(response.text)}") from exc
            usage = payload.get("usage") or {}
            result = LLMResponse(
                text=text,
                model=self._model,
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                latency_s=latency,
                extra={"finish_reason": str(choice.get("finish_reason", ""))},
                tool_calls=tool_calls,
            )
            if self._limiter and reservation is not None:
                self._limiter.settle(reservation, result.total_tokens)
            return result

        raise LLMError("Abandon après toutes les relances")  # pragma: no cover

    def _release(self, reservation: Reservation | None, tokens: int) -> None:
        if self._limiter and reservation is not None:
            self._limiter.settle(reservation, tokens)

    def _backoff(self, attempt: int, retry_after: str | None, reason: str) -> None:
        delay = min(2.0 ** (attempt + 1), MAX_BACKOFF_SECONDS)
        if (seconds := parse_seconds(retry_after)) is not None:
            delay = min(seconds + 0.5, MAX_BACKOFF_SECONDS)
        logger.warning("LLM : essai %d après %s, attente %.1f s", attempt + 1, reason, delay)
        self._sleep(delay)
