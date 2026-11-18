"""Client pour les API compatibles OpenAI (Groq), sans SDK : un simple appel HTTP."""

import json
import logging
import time
from collections.abc import Callable
from typing import Any

import httpx

from ledger_agent.llm.rate_limit import TokenRateLimiter
from ledger_agent.llm.types import CompletionParams, LLMError, LLMQuotaError, LLMResponse, Message

logger = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
MAX_BACKOFF_SECONDS = 60.0
# Approximation classique pour l'anglais : 1 token ≈ 4 caractères
CHARS_PER_TOKEN = 4


def _retry_after_seconds(header: str | None) -> float | None:
    try:
        return float(header) if header else None
    except ValueError:
        return None


def estimate_tokens(messages: list[Message], max_tokens: int) -> int:
    prompt_chars = sum(len(m["content"]) for m in messages)
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
    ) -> None:
        if not api_key:
            raise ValueError(f"Missing API key for model {model}")
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
        return body

    def complete(self, messages: list[Message], params: CompletionParams) -> LLMResponse:
        body = self._body(messages, params)
        for attempt in range(self._max_retries + 1):
            is_last_attempt = attempt == self._max_retries
            reservation = (
                self._limiter.acquire(estimate_tokens(messages, params.max_tokens))
                if self._limiter
                else None
            )
            started = time.perf_counter()
            try:
                response = self._client.post("/chat/completions", json=body)
            except httpx.TransportError as exc:
                if is_last_attempt:
                    raise LLMError(f"Network error: {exc}") from exc
                self._backoff(attempt, None, type(exc).__name__)
                continue
            latency = time.perf_counter() - started

            # Quota quotidien épuisé : attendre des minutes ne sert à rien, on s'arrête net
            wait = _retry_after_seconds(response.headers.get("retry-after"))
            if response.status_code == 429 and wait is not None and wait > MAX_BACKOFF_SECONDS:
                raise LLMQuotaError(
                    f"Quota exhausted for {self._model}, retry in {wait / 60:.0f} min",
                    retry_after_s=wait,
                )
            if response.status_code in RETRYABLE_STATUS and not is_last_attempt:
                self._backoff(
                    attempt, response.headers.get("retry-after"), str(response.status_code)
                )
                continue
            if response.is_error:
                raise LLMError(f"HTTP {response.status_code}: {response.text[:300]}")

            payload = response.json()
            usage = payload.get("usage") or {}
            result = LLMResponse(
                text=(payload["choices"][0]["message"].get("content") or "").strip(),
                model=self._model,
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                latency_s=latency,
            )
            if self._limiter and reservation is not None:
                self._limiter.settle(reservation, result.total_tokens)
            return result

        raise LLMError("Giving up after retries")  # pragma: no cover

    def _backoff(self, attempt: int, retry_after: str | None, reason: str) -> None:
        delay = min(2.0 ** (attempt + 1), MAX_BACKOFF_SECONDS)
        if retry_after:
            try:
                delay = min(float(retry_after) + 0.5, MAX_BACKOFF_SECONDS)
            except ValueError:
                logger.debug("Unparsable retry-after header: %s", json.dumps(retry_after))
        logger.warning("LLM retry %d after %s, waiting %.1fs", attempt + 1, reason, delay)
        self._sleep(delay)
