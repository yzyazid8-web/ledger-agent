"""Limite de débit en tokens par minute, sur une fenêtre glissante de 60 secondes."""

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

WINDOW_SECONDS = 60.0


@dataclass(slots=True)
class Reservation:
    timestamp: float
    tokens: int


class TokenRateLimiter:
    def __init__(
        self,
        tokens_per_minute: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if tokens_per_minute <= 0:
            raise ValueError("tokens_per_minute must be positive")
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
