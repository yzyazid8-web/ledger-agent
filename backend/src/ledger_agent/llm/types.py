"""Types partagés par tous les fournisseurs de LLM."""

from dataclasses import dataclass, field
from typing import Any, Literal, NotRequired, Protocol, TypedDict


class ToolCallFunction(TypedDict):
    name: str
    # Arguments sérialisés en JSON, tels que renvoyés par le modèle
    arguments: str


class ToolCall(TypedDict):
    id: str
    type: Literal["function"]
    function: ToolCallFunction


class Message(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None
    # Message de l'assistant qui demande des appels d'outils
    tool_calls: NotRequired[list[ToolCall]]
    # Message « tool » qui rapporte le résultat d'un appel
    tool_call_id: NotRequired[str]


@dataclass(frozen=True, slots=True)
class CompletionParams:
    max_tokens: int = 1024
    temperature: float = 0.0
    # Utilisé seulement par les modèles de raisonnement (gpt-oss) : "low", "medium", "high"
    reasoning_effort: str | None = None
    json_mode: bool = False
    # Définitions d'outils au format OpenAI (type « function ») ; vide = aucun outil proposé
    tools: tuple[dict[str, Any], ...] = ()
    # "required" oblige le modèle à appeler un outil plutôt qu'à répondre en texte libre
    tool_choice: Literal["auto", "required"] = "auto"


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cached: bool = False
    extra: dict[str, str] = field(default_factory=dict)
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient(Protocol):
    @property
    def model(self) -> str: ...

    def complete(self, messages: list[Message], params: CompletionParams) -> LLMResponse: ...


class LLMError(RuntimeError):
    """Levée quand le fournisseur échoue après toutes les relances."""


class LLMToolCallError(LLMError):
    """Appel d'outil mal formé, refusé par le fournisseur : le modèle peut se corriger."""


class LLMQuotaError(LLMError):
    """Quota (souvent quotidien) épuisé : inutile de réessayer avant `retry_after_s` secondes."""

    def __init__(self, message: str, retry_after_s: float) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s
