"""Lecture des réponses d'erreur des fournisseurs : quotas et messages nettoyés."""

import re

# Identifiants de compte que les fournisseurs glissent dans leurs messages d'erreur
_ACCOUNT_IDS = re.compile(
    r"\b(org|organization|project)[_ `'\"]*[A-Za-z0-9_-]{8,}`?", re.IGNORECASE
)
_DAILY_QUOTA = re.compile(r"per day|\bTPD\b|\bRPD\b|daily", re.IGNORECASE)
_TRY_AGAIN = re.compile(r"try again in\s*(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:([\d.]+)s)?", re.IGNORECASE)


def sanitize(text: str, limit: int = 300) -> str:
    """Tronque un message d'erreur et masque les identifiants de compte."""
    return _ACCOUNT_IDS.sub(r"\1 <masqué>", text)[:limit]


def parse_seconds(header: str | None) -> float | None:
    try:
        return float(header) if header else None
    except ValueError:
        return None


def daily_quota_wait(body: str, retry_after: str | None) -> float | None:
    """Délai d'attente si le corps signale un quota quotidien, sinon None."""
    if not _DAILY_QUOTA.search(body):
        return None
    if (seconds := parse_seconds(retry_after)) is not None:
        return seconds
    match = _TRY_AGAIN.search(body)
    if match and any(match.groups()):
        hours, minutes, seconds_text = match.groups()
        return int(hours or 0) * 3600 + int(minutes or 0) * 60 + float(seconds_text or 0)
    # Quota quotidien sans délai indiqué : on suppose une heure
    return 3600.0
