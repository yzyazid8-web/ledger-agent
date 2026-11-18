"""Format de sortie commun à tous les systèmes évalués (RAG naïf, agent)."""

from dataclasses import dataclass, field
from typing import Any, Protocol

from ledger_agent.llm.types import LLMResponse
from ledger_agent.retrieval.types import SearchOutcome


@dataclass(frozen=True, slots=True)
class RetrievedPage:
    chunk_id: str
    doc_name: str
    page_index: int
    score: float


@dataclass(frozen=True, slots=True)
class PipelineAnswer:
    text: str
    abstained: bool
    retrieved: list[RetrievedPage]
    cited: list[tuple[str, int]]
    llm_calls: list[LLMResponse]
    retrieval_s: float
    search: SearchOutcome | None = None
    # Étapes de l'agent, pour l'interface et le débogage
    trace: list[dict[str, Any]] = field(default_factory=list)

    @property
    def prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.llm_calls)

    @property
    def completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.llm_calls)

    @property
    def all_cached(self) -> bool:
        return bool(self.llm_calls) and all(c.cached for c in self.llm_calls)

    @property
    def llm_s(self) -> float:
        return sum(c.latency_s for c in self.llm_calls)


class Pipeline(Protocol):
    def run(self, question: str) -> PipelineAnswer: ...
