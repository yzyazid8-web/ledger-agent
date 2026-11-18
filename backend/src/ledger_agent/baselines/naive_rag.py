"""RAG naïf : une recherche top-k, puis une seule génération."""

import hashlib
import re
import time
from dataclasses import dataclass

from ledger_agent.answer_format import ABSTAIN_MARKER, is_abstention
from ledger_agent.llm.types import CompletionParams, LLMClient, LLMResponse, Message
from ledger_agent.pipeline import PipelineAnswer, RetrievedPage
from ledger_agent.retrieval.types import (
    ReportingRetriever,
    RetrievedChunk,
    Retriever,
    SearchOutcome,
)

# Ne pas modifier sans raison : le texte du prompt fait partie de la clé du cache LLM
SYSTEM_PROMPT = f"""You are a financial analyst answering questions about company filings.
Answer ONLY from the numbered context passages. Do not use outside knowledge.
Cite the passages you use with their number in plain square brackets, e.g. [2].
Do not use any other citation format.
Show the key figures and any calculation briefly.
End with a last line of the form "Final answer: <answer>".
If the passages do not contain what is needed, reply exactly: {ABSTAIN_MARKER}"""

PROMPT_SHA = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:12]

# gpt-oss cite parfois dans son format natif 【3†L9-L12】 malgré la consigne
_CITATION = re.compile(r"\[(\d{1,2})\]|" + chr(0x3010) + r"(\d{1,2})" + chr(0x2020))


@dataclass(frozen=True, slots=True)
class RAGAnswer:
    text: str
    retrieved: list[RetrievedChunk]
    cited: list[RetrievedChunk]
    abstained: bool
    llm: LLMResponse
    search: SearchOutcome
    retrieval_s: float


def format_context(passages: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{i}] {p.chunk.header}\n{p.chunk.text}" for i, p in enumerate(passages, 1))


def build_messages(question: str, passages: list[RetrievedChunk]) -> list[Message]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context passages:\n\n{format_context(passages)}\n\nQuestion: {question}",
        },
    ]


def parse_citations(text: str, passages: list[RetrievedChunk]) -> list[RetrievedChunk]:
    numbers = dict.fromkeys(int(a or b) for a, b in _CITATION.findall(text))
    return [passages[n - 1] for n in numbers if 1 <= n <= len(passages)]


class NaiveRAG:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        top_k: int = 5,
        params: CompletionParams | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._top_k = top_k
        self._params = params or CompletionParams()

    def _search(self, question: str) -> SearchOutcome:
        if isinstance(self._retriever, ReportingRetriever):
            return self._retriever.search_with_outcome(question, self._top_k)
        results = self._retriever.search(question, self._top_k)
        return SearchOutcome(results=results, requested=None, applied=None)

    def answer(self, question: str) -> RAGAnswer:
        started = time.perf_counter()
        outcome = self._search(question)
        retrieval_s = time.perf_counter() - started
        passages = outcome.results
        response = self._llm.complete(build_messages(question, passages), self._params)
        return RAGAnswer(
            text=response.text,
            retrieved=passages,
            cited=parse_citations(response.text, passages),
            abstained=is_abstention(response.text),
            llm=response,
            search=outcome,
            retrieval_s=retrieval_s,
        )

    def run(self, question: str) -> PipelineAnswer:
        answer = self.answer(question)
        return PipelineAnswer(
            text=answer.text,
            abstained=answer.abstained,
            retrieved=[
                RetrievedPage(r.chunk.chunk_id, r.chunk.doc_name, r.chunk.page_index, r.score)
                for r in answer.retrieved
            ],
            cited=[(c.chunk.doc_name, c.chunk.page_index) for c in answer.cited],
            llm_calls=[answer.llm],
            retrieval_s=answer.retrieval_s,
            search=answer.search,
        )
