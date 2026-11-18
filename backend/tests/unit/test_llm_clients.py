import json
from pathlib import Path

import httpx
import pytest

from ledger_agent.llm.cache import CachedLLM
from ledger_agent.llm.gemini import GeminiClient
from ledger_agent.llm.openai_compatible import OpenAICompatibleClient
from ledger_agent.llm.types import CompletionParams, LLMError, LLMResponse, Message

MESSAGES: list[Message] = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "2+2?"},
]


def chat_response(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        },
    )


def test_openai_compatible_parses_answer_and_usage() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return chat_response(" 4 ")

    client = OpenAICompatibleClient(
        "key", "openai/gpt-oss-120b", tokens_per_minute=None, transport=httpx.MockTransport(handler)
    )
    response = client.complete(MESSAGES, CompletionParams(reasoning_effort="low"))

    assert (response.text, response.prompt_tokens, response.completion_tokens) == ("4", 12, 3)
    assert seen[0]["reasoning_effort"] == "low"
    assert seen[0]["model"] == "openai/gpt-oss-120b"


def test_openai_compatible_retries_on_429() -> None:
    responses = iter([httpx.Response(429, headers={"retry-after": "3"}), chat_response("4")])
    sleeps: list[float] = []
    client = OpenAICompatibleClient(
        "key",
        "m",
        tokens_per_minute=None,
        transport=httpx.MockTransport(lambda _: next(responses)),
        sleep=sleeps.append,
    )

    assert client.complete(MESSAGES, CompletionParams()).text == "4"
    assert sleeps == [3.5]


def test_openai_compatible_raises_on_client_error() -> None:
    client = OpenAICompatibleClient(
        "key",
        "m",
        tokens_per_minute=None,
        transport=httpx.MockTransport(lambda _: httpx.Response(400, text="bad request")),
    )

    with pytest.raises(LLMError, match="400"):
        client.complete(MESSAGES, CompletionParams())


def test_gemini_moves_system_prompt_and_reads_text() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        assert request.headers["x-goog-api-key"] == "key"
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": '{"verdict": "correct"}'}]}}],
                "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 5},
            },
        )

    client = GeminiClient("key", requests_per_minute=6000, transport=httpx.MockTransport(handler))
    response = client.complete(MESSAGES, CompletionParams(json_mode=True))

    assert response.text == '{"verdict": "correct"}'
    body = seen[0]
    assert body["systemInstruction"] == {"parts": [{"text": "Be brief."}]}
    assert body["contents"] == [{"role": "user", "parts": [{"text": "2+2?"}]}]
    assert body["generationConfig"]["responseMimeType"] == "application/json"  # type: ignore[index]


class CountingLLM:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def model(self) -> str:
        return "fake"

    def complete(self, messages: list[Message], params: CompletionParams) -> LLMResponse:
        self.calls += 1
        return LLMResponse(text=f"answer {self.calls}", model="fake", prompt_tokens=10)


def test_cache_avoids_paying_twice(tmp_path: Path) -> None:
    inner = CountingLLM()
    cached = CachedLLM(inner, tmp_path / "cache.sqlite")

    first = cached.complete(MESSAGES, CompletionParams())
    second = cached.complete(MESSAGES, CompletionParams())
    other = cached.complete(MESSAGES, CompletionParams(temperature=0.5))
    cached.close()

    assert inner.calls == 2
    assert (first.text, first.cached) == ("answer 1", False)
    assert (second.text, second.cached) == ("answer 1", True)
    assert other.text == "answer 2"
