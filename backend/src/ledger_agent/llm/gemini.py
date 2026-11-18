"""Client Gemini (API REST generativelanguage), utilisé comme juge d'évaluation."""

import logging
import time
from collections.abc import Callable
from typing import Any

import httpx

from ledger_agent.ingestion.edgar_client import RateLimiter
from ledger_agent.llm.types import CompletionParams, LLMError, LLMResponse, Message

logger = logging.getLogger(__name__)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_JUDGE_MODEL = "gemini-3.5-flash-lite"
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


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
            raise ValueError("Missing GEMINI_API_KEY")
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
            self._limiter.wait()
            started = time.perf_counter()
            try:
                response = self._client.post(url, json=body)
            except httpx.TransportError as exc:
                if attempt == self._max_retries:
                    raise LLMError(f"Network error: {exc}") from exc
                self._sleep(2.0 ** (attempt + 1))
                continue
            if response.status_code in RETRYABLE_STATUS and attempt < self._max_retries:
                delay = min(2.0 ** (attempt + 2), 60.0)
                logger.warning(
                    "Gemini retry %d after %d, waiting %.0fs",
                    attempt + 1,
                    response.status_code,
                    delay,
                )
                self._sleep(delay)
                continue
            if response.is_error:
                raise LLMError(f"HTTP {response.status_code}: {response.text[:300]}")

            payload = response.json()
            parts = payload["candidates"][0].get("content", {}).get("parts", [])
            usage = payload.get("usageMetadata", {})
            return LLMResponse(
                text="".join(p.get("text", "") for p in parts).strip(),
                model=self._model,
                prompt_tokens=int(usage.get("promptTokenCount", 0)),
                completion_tokens=int(usage.get("candidatesTokenCount", 0)),
                latency_s=time.perf_counter() - started,
            )
        raise LLMError("Giving up after retries")  # pragma: no cover
