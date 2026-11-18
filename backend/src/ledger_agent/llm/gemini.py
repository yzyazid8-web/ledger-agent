"""Client Gemini (API REST generativelanguage), utilisé comme juge d'évaluation."""

import logging
import time
from collections.abc import Callable
from typing import Any

import httpx

from ledger_agent.common.rate_limit import RateLimiter
from ledger_agent.llm.http_errors import daily_quota_wait, parse_seconds, sanitize
from ledger_agent.llm.types import CompletionParams, LLMError, LLMQuotaError, LLMResponse, Message

logger = logging.getLogger(__name__)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_JUDGE_MODEL = "gemini-3.5-flash-lite"
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
MAX_BACKOFF_SECONDS = 60.0


def parse_response(payload: dict[str, Any], model: str, latency: float) -> LLMResponse:
    """Extrait le texte, ou lève LLMError si la réponse est bloquée ou tronquée."""
    candidates = payload.get("candidates") or []
    if not candidates:
        # Réponse bloquée par un filtre de sécurité, ou vide
        reason = payload.get("promptFeedback", {}).get("blockReason", "aucun candidat")
        raise LLMError(f"Réponse Gemini bloquée : {reason}")
    candidate = candidates[0]
    finish = candidate.get("finishReason", "")
    parts = candidate.get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts).strip()
    if finish not in ("", "STOP") or not text:
        raise LLMError(f"Réponse Gemini incomplète : finishReason={finish or 'vide'}")
    usage = payload.get("usageMetadata", {})
    return LLMResponse(
        text=text,
        model=model,
        prompt_tokens=int(usage.get("promptTokenCount", 0)),
        completion_tokens=int(usage.get("candidatesTokenCount", 0)),
        latency_s=latency,
        extra={"finish_reason": finish},
    )


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_JUDGE_MODEL,
        *,
        requests_per_minute: float = 8.0,
        max_retries: int = 6,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY manquante")
        self._model = model
        self._max_retries = max_retries
        self._sleep = sleep
        self._limiter = RateLimiter(requests_per_minute / 60.0, sleep=sleep)
        self._client = httpx.Client(
            base_url=GEMINI_BASE_URL,
            # La clé passe dans un en-tête plutôt que dans l'URL, pour ne pas finir dans les logs
            headers={"x-goog-api-key": api_key},
            timeout=timeout,
            transport=transport,
        )

    @property
    def model(self) -> str:
        return self._model

    def close(self) -> None:
        self._client.close()

    def _body(self, messages: list[Message], params: CompletionParams) -> dict[str, Any]:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [
            {
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": [{"text": m["content"]}],
            }
            for m in messages
            if m["role"] != "system"
        ]
        config: dict[str, Any] = {
            "temperature": params.temperature,
            "maxOutputTokens": params.max_tokens,
        }
        if params.json_mode:
            config["responseMimeType"] = "application/json"
        body: dict[str, Any] = {"contents": contents, "generationConfig": config}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return body

    def complete(self, messages: list[Message], params: CompletionParams) -> LLMResponse:
        body = self._body(messages, params)
        url = f"/models/{self._model}:generateContent"
        for attempt in range(self._max_retries + 1):
            is_last_attempt = attempt == self._max_retries
            self._limiter.wait()
            started = time.perf_counter()
            try:
                response = self._client.post(url, json=body)
            except httpx.TransportError as exc:
                if is_last_attempt:
                    raise LLMError(f"Erreur réseau : {exc}") from exc
                self._sleep(2.0 ** (attempt + 1))
                continue

            retry_after = response.headers.get("retry-after")
            if response.status_code == 429 and (
                wait := daily_quota_wait(response.text, retry_after)
            ):
                raise LLMQuotaError(f"Quota Gemini épuisé ({self._model})", retry_after_s=wait)
            if response.status_code in RETRYABLE_STATUS and not is_last_attempt:
                delay = min(2.0 ** (attempt + 2), MAX_BACKOFF_SECONDS)
                if (seconds := parse_seconds(retry_after)) is not None:
                    delay = min(seconds + 0.5, MAX_BACKOFF_SECONDS)
                logger.warning(
                    "Gemini : essai %d après %d, attente %.0f s",
                    attempt + 1,
                    response.status_code,
                    delay,
                )
                self._sleep(delay)
                continue
            if response.is_error:
                raise LLMError(f"HTTP {response.status_code} : {sanitize(response.text)}")

            try:
                payload = response.json()
            except ValueError as exc:
                raise LLMError(f"Réponse non JSON : {sanitize(response.text)}") from exc
            return parse_response(payload, self._model, time.perf_counter() - started)
        raise LLMError("Abandon après toutes les relances")  # pragma: no cover
