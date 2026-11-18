from ledger_agent.baselines.naive_rag import ABSTAIN_MARKER, NaiveRAG
from ledger_agent.ingestion.chunking import Chunk
from ledger_agent.llm.types import CompletionParams, LLMResponse, Message
from ledger_agent.retrieval.types import RetrievedChunk, SearchFilters


def make_chunk(page_index: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"id-{page_index}",
        doc_name="3M_2018_10K",
        company="3M",
        ticker="MMM",
        doc_type="10-K",
        fiscal_year=2018,
        fiscal_quarter=None,
        page_index=page_index,
        chunk_index=0,
        header=f"3M | 10-K | FY2018 | page {page_index + 1}",
        text=text,
    )


class FakeRetriever:
    def search(
        self, query: str, k: int, filters: SearchFilters | None = None
    ) -> list[RetrievedChunk]:
        chunks = [make_chunk(59, "Purchases of PP&E | (1,577)"), make_chunk(10, "Risk factors")]
        return [RetrievedChunk(c, 1.0 - i / 10, i) for i, c in enumerate(chunks[:k])]


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[list[Message]] = []

    @property
    def model(self) -> str:
        return "fake"

    def complete(self, messages: list[Message], params: CompletionParams) -> LLMResponse:
        self.prompts.append(messages)
        return LLMResponse(text=self.reply, model="fake")


def test_prompt_contains_numbered_context_and_citations_are_resolved() -> None:
    llm = FakeLLM("Capex was 1,577 [1].\nFinal answer: $1,577 million")
    answer = NaiveRAG(FakeRetriever(), llm, top_k=2).answer("3M capex 2018?")

    user_prompt = llm.prompts[0][1]["content"]
    assert "[1] 3M | 10-K | FY2018 | page 60" in user_prompt
    assert "[2] 3M | 10-K | FY2018 | page 11" in user_prompt
    assert [c.chunk.page_index for c in answer.cited] == [59]
    assert not answer.abstained


def test_native_gpt_oss_citations_are_resolved() -> None:
    reply = "Capex " + chr(0x3010) + "2" + chr(0x2020) + "L9-L12" + chr(0x3011) + " and [1]."
    answer = NaiveRAG(FakeRetriever(), FakeLLM(reply), top_k=2).answer("q")

    assert [c.chunk.page_index for c in answer.cited] == [10, 59]


def test_abstention_marker_is_detected() -> None:
    answer = NaiveRAG(FakeRetriever(), FakeLLM(ABSTAIN_MARKER)).answer("Unknown?")

    assert answer.abstained
    assert answer.cited == []


def test_out_of_range_citations_are_ignored() -> None:
    answer = NaiveRAG(FakeRetriever(), FakeLLM("See [7] and [2]."), top_k=2).answer("q")

    assert [c.chunk.page_index for c in answer.cited] == [10]
