"""Limite de questions par visiteur : protège le quota gratuit du modèle sur le site public."""

import threading
import time
from collections import deque
from collections.abc import Callable

from fastapi import Request

WINDOW_SECONDS = 3600.0
# Au-delà, les visiteurs les plus anciens sont oubliés : la mémoire reste bornée
MAX_TRACKED_VISITORS = 10_000


def visitor_key(request: Request) -> str:
    """Adresse du visiteur, telle que transmise par le relais de l'interface."""
    forwarded = request.headers.get("x-forwarded-for", "")
    # L'entrée HTTP d'Azure ajoute l'adresse réelle en fin d'en-tête ; le début peut être
    # rempli à volonté par le client, il ne sert donc pas à l'identifier
    parts = [p.strip() for p in forwarded.split(",") if p.strip()]
    if parts:
        return parts[-1]
    return request.client.host if request.client else "inconnu"


class VisitorLimiter:
    """Au plus `per_hour` questions par visiteur sur une heure glissante ; 0 désactive."""

    def __init__(self, per_hour: int, clock: Callable[[], float] = time.monotonic) -> None:
        self._per_hour = per_hour
        self._clock = clock
        self._seen: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    @property
    def per_hour(self) -> int:
        return self._per_hour

    def allow(self, key: str) -> bool:
        if self._per_hour <= 0:
            return True
        now = self._clock()
        with self._lock:
            times = self._seen.setdefault(key, deque())
            while times and now - times[0] >= WINDOW_SECONDS:
                times.popleft()
            if len(times) >= self._per_hour:
                return False
            times.append(now)
            if len(self._seen) > MAX_TRACKED_VISITORS:
                self._forget_idle(now)
            return True

    def _forget_idle(self, now: float) -> None:
        idle = [k for k, t in self._seen.items() if not t or now - t[-1] >= WINDOW_SECONDS]
        for key in idle:
            del self._seen[key]
        # Toujours trop de visiteurs actifs : on oublie les plus anciens
        while len(self._seen) > MAX_TRACKED_VISITORS:
            del self._seen[next(iter(self._seen))]
