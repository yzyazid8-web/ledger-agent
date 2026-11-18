"""Client HTTP respectueux de SEC EDGAR : User-Agent identifié, limite de débit, relances."""

import contextlib
import json
import logging
import threading
import time
from collections.abc import Callable
from types import TracebackType
from typing import Any, Self

import httpx

from ledger_agent.config import Settings, get_settings

logger = logging.getLogger(__name__)

SEC_DATA_URL = "https://data.sec.gov"
SEC_WWW_URL = "https://www.sec.gov"

# Politique d'accès de la SEC : 10 requêtes par seconde au maximum
DEFAULT_MAX_REQUESTS_PER_SECOND = 8.0
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
MAX_BACKOFF_SECONDS = 30.0


class EdgarError(RuntimeError):
    """Levée quand EDGAR est injoignable ou refuse la requête."""


class RateLimiter:
    """Espace les appels pour ne jamais dépasser `max_per_second` appels par seconde."""

    def __init__(
        self,
        max_per_second: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_per_second <= 0:
            raise ValueError("max_per_second must be positive")
        self._interval = 1.0 / max_per_second
        self._clock = clock
        self._sleep = sleep
        self._next_slot = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            if now < self._next_slot:
                self._sleep(self._next_slot - now)
                now = self._next_slot
            self._next_slot = now + self._interval


class EdgarClient:
    def __init__(
        self,
        user_agent: str,
        *,
        max_requests_per_second: float = DEFAULT_MAX_REQUESTS_PER_SECOND,
        max_retries: int = 5,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if "@" not in user_agent:
            raise ValueError(
                "SEC_USER_AGENT must contain a contact email, e.g. 'my-app me@example.com'"
            )
        self._max_retries = max_retries
        self._sleep = sleep
        self._limiter = RateLimiter(max_requests_per_second, sleep=sleep)
        self._client = httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=timeout,
            transport=transport,
            follow_redirects=True,
        )

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> Self:
        return cls((settings or get_settings()).sec_user_agent)

    # Gestionnaire de contexte

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # Requêtes génériques

    def get_bytes(self, url: str) -> bytes:
        for attempt in range(self._max_retries + 1):
            is_last_attempt = attempt == self._max_retries
            self._limiter.wait()
            try:
                response = self._client.get(url)
            except httpx.TransportError as exc:
                if is_last_attempt:
                    raise EdgarError(f"Network error on {url}: {exc}") from exc
                self._backoff(attempt, None, reason=type(exc).__name__)
                continue

            if response.status_code in RETRYABLE_STATUS and not is_last_attempt:
                self._backoff(
                    attempt, response.headers.get("Retry-After"), reason=str(response.status_code)
                )
                continue
            if response.status_code == 403:
                raise EdgarError(f"403 Forbidden on {url}: check that SEC_USER_AGENT is set")
            if response.is_error:
                raise EdgarError(f"HTTP {response.status_code} on {url}")
            return response.content

        raise EdgarError(f"Giving up on {url}")  # pragma: no cover

    def get_json(self, url: str) -> dict[str, Any]:
        payload = json.loads(self.get_bytes(url))
        if not isinstance(payload, dict):
            raise EdgarError(f"Expected a JSON object from {url}")
        return payload

    # Points d'accès EDGAR

    @staticmethod
    def company_facts_url(cik: int) -> str:
        return f"{SEC_DATA_URL}/api/xbrl/companyfacts/CIK{cik:010d}.json"

    @staticmethod
    def submissions_url(cik: int) -> str:
        return f"{SEC_DATA_URL}/submissions/CIK{cik:010d}.json"

    def company_facts(self, cik: int) -> dict[str, Any]:
        return self.get_json(self.company_facts_url(cik))

    def submissions(self, cik: int) -> dict[str, Any]:
        return self.get_json(self.submissions_url(cik))

    # Fonctions internes

    def _backoff(self, attempt: int, retry_after: str | None, *, reason: str) -> None:
        # Attente exponentielle, sauf si le serveur impose un délai via Retry-After
        delay = min(2.0**attempt, MAX_BACKOFF_SECONDS)
        if retry_after is not None:
            with contextlib.suppress(ValueError):
                delay = min(float(retry_after), MAX_BACKOFF_SECONDS)
        logger.warning("EDGAR retry %d after %s, waiting %.1fs", attempt + 1, reason, delay)
        self._sleep(delay)
