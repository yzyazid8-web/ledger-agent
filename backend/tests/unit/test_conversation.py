"""Questions de suivi : reformulation en question autonome, prudente en cas de doute."""

from typing import Any

from ledger_agent.api.conversation import MAX_TURNS, QuestionRewriter
from ledger_agent.llm.types import CompletionParams, LLMError, LLMResponse, Message

HISTORY = [{"question": "What was 3M's capital expenditure in FY2018?", "answer": "$1,577 million"}]


class FakeLLM:
    def __init__(self, reply: str | Exception) -> None:
        self.reply = reply
        self.prompts: list[list[Message]] = []

    @property
    def model(self) -> str:
        return "fake"

    def complete(self, messages: list[Message], params: CompletionParams) -> LLMResponse:
        self.prompts.append(messages)
        if isinstance(self.reply, Exception):
            raise self.reply
        return LLMResponse(text=self.reply, model="fake")


def test_a_follow_up_becomes_a_standalone_question() -> None:
    llm = FakeLLM("What was 3M's capital expenditure in FY2019?")

    rewritten = QuestionRewriter(llm).rewrite("And in 2019?", HISTORY)

    assert rewritten == "What was 3M's capital expenditure in FY2019?"
    prompt = llm.prompts[0][-1]["content"] or ""
    assert "FY2018" in prompt and "$1,577 million" in prompt and "And in 2019?" in prompt


def test_without_history_the_model_is_not_called() -> None:
    llm = FakeLLM("never used")

    assert QuestionRewriter(llm).rewrite("What was 3M's capex in FY2018?", []) == (
        "What was 3M's capex in FY2018?"
    )
    assert llm.prompts == []


def test_a_failure_or_an_unusable_reply_keeps_the_original_question() -> None:
    for reply in [LLMError("quota"), "", "line one\nline two", "x" * 500]:
        assert QuestionRewriter(FakeLLM(reply)).rewrite("And in 2019?", HISTORY) == "And in 2019?"


def test_prefixes_and_quotes_added_by_the_model_are_removed() -> None:
    llm = FakeLLM('Standalone question: "What was 3M\'s capital expenditure in FY2019?"')

    assert QuestionRewriter(llm).rewrite("And in 2019?", HISTORY) == (
        "What was 3M's capital expenditure in FY2019?"
    )


def test_only_the_last_turns_are_sent() -> None:
    history: list[dict[str, Any]] = [
        {"question": f"old question {i}", "answer": f"answer {i}"} for i in range(6)
    ]
    llm = FakeLLM("What was 3M's capital expenditure in FY2019?")

    QuestionRewriter(llm).rewrite("And in 2019?", history)

    prompt = llm.prompts[0][-1]["content"] or ""
    assert "old question 5" in prompt and "old question 0" not in prompt
    assert prompt.count("User:") == MAX_TURNS
