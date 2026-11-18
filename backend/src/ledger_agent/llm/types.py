"""Types partagés par tous les fournisseurs de LLM."""

from dataclasses import dataclass, field
from typing import Literal, Protocol, TypedDict


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class CompletionParams:
    max_tokens: int = 1024
    temperature: float = 0.0
    # Utilisé seulement par les modèles de raisonnement (gpt-oss) : "low", "medium", "high"
    reasoning_effort: str | None = None
    json_mode: bool = False


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cached: bool = False
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient(Protocol):
    @property
    def model(self) -> str: ...

    def complete(self, messages: list[Message], params: CompletionParams) -> LLMResponse: ...


class LLMError(RuntimeError):
    """Levée quand le fournisseur échoue après toutes les relances."""


class LLMQuotaError(LLMError):
    """Quota (souvent quotidien) épuisé : inutile de réessayer avant `retry_after_s` secondes."""

    def __init__(self, message: str, retry_after_s: float) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s
