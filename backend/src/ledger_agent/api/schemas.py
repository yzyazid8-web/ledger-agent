"""Schémas des requêtes et réponses de l'API."""

from typing import Any

from pydantic import BaseModel, Field


class Turn(BaseModel):
    """Échange précédent de la conversation : la question posée et la réponse courte obtenue."""

    question: str = Field(max_length=1000)
    answer: str | None = Field(default=None, max_length=2000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    # Derniers échanges, envoyés par le navigateur : l'API ne garde aucune conversation en mémoire
    history: list[Turn] = Field(default_factory=list, max_length=6)


class Company(BaseModel):
    ticker: str
    name: str


class Health(BaseModel):
    status: str
    ready: bool
    config: str


class AnswerResponse(BaseModel):
    """Réponse complète, pour les clients qui n'utilisent pas le flux SSE."""

    answer: dict[str, Any]
    steps: list[dict[str, Any]]
