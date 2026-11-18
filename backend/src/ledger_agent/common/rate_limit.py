"""Limites de débit : en requêtes par seconde, ou en tokens sur une minute glissante."""

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

WINDOW_SECONDS = 60.0


class RateLimiter:
    """Espace les appels pour ne jamais dépasser `max_per_second` appels par seconde."""

    def __init__(
        self,
        max_per_second: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_per_second <= 0:
            raise ValueError("max_per_second doit être strictement positif")
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


@dataclass(slots=True)
class Reservation:
    timestamp: float
    tokens: int


class TokenRateLimiter:
    """Quota de tokens par minute : attend avant d'envoyer plutôt que de subir des 429."""

    def __init__(
        self,
        tokens_per_minute: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if tokens_per_minute <= 0:
            raise ValueError("tokens_per_minute doit être strictement positif")
        self._limit = tokens_per_minute
        self._clock = clock
        self._sleep = sleep
        self._window: deque[Reservation] = deque()

    def _purge(self, now: float) -> None:
        while self._window and now - self._window[0].timestamp >= WINDOW_SECONDS:
            self._window.popleft()

    def used(self) -> int:
        self._purge(self._clock())
        return sum(r.tokens for r in self._window)

    def acquire(self, estimated_tokens: int) -> Reservation:
        """Attend qu'il y ait de la place pour `estimated_tokens`, puis les réserve."""
        tokens = min(estimated_tokens, self._limit)
        while True:
            now = self._clock()
            self._purge(now)
            if not self._window or self.used() + tokens <= self._limit:
                reservation = Reservation(now, tokens)
                self._window.append(reservation)
                return reservation
            # On attend que la plus ancienne réservation sorte de la fenêtre
            self._sleep(WINDOW_SECONDS - (now - self._window[0].timestamp) + 0.05)

    @staticmethod
    def settle(reservation: Reservation, actual_tokens: int) -> None:
        """Remplace l'estimation par la consommation réelle renvoyée par l'API."""
        reservation.tokens = actual_tokens
