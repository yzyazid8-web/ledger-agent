"""Schémas des requêtes et réponses de l'API."""

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


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
