import json
from pathlib import Path

import httpx
import pytest

from ledger_agent.llm.cache import CachedLLM
from ledger_agent.llm.gemini import GeminiClient, parse_response
from ledger_agent.llm.http_errors import daily_quota_wait, sanitize
from ledger_agent.llm.openai_compatible import OpenAICompatibleClient
from ledger_agent.llm.types import (
    CompletionParams,
    LLMError,
    LLMQuotaError,
    LLMResponse,
    LLMToolCallError,
    Message,
)

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


def test_daily_quota_stops_immediately_without_retrying() -> None:
    sleeps: list[float] = []
    client = OpenAICompatibleClient(
        "key",
        "m",
        tokens_per_minute=None,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(429, headers={"retry-after": "822"}, text="TPD")
        ),
        sleep=sleeps.append,
    )

    with pytest.raises(LLMQuotaError, match="14 min") as error:
        client.complete(MESSAGES, CompletionParams())

    assert error.value.retry_after_s == 822
    assert sleeps == []


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


GROQ_TPD_BODY = (
    '{"error":{"message":"Rate limit reached for model `openai/gpt-oss-120b` in organization '
    "`org_01ksyp3jn6erbajmettfn0ex6h` service tier `on_demand` on tokens per day (TPD): "
    'Limit 200000, Used 199063, Requested 2840. Please try again in 13m42.09s."}}'
)


def test_daily_quota_detected_from_body_without_retry_after() -> None:
    sleeps: list[float] = []
    client = OpenAICompatibleClient(
        "key",
        "m",
        tokens_per_minute=None,
        transport=httpx.MockTransport(lambda _: httpx.Response(429, text=GROQ_TPD_BODY)),
        sleep=sleeps.append,
    )

    with pytest.raises(LLMQuotaError) as error:
        client.complete(MESSAGES, CompletionParams())

    assert error.value.retry_after_s == pytest.approx(13 * 60 + 42.09)
    assert sleeps == []


def test_minute_rate_limit_is_not_a_daily_quota() -> None:
    assert daily_quota_wait("Rate limit reached on tokens per minute (TPM)", "2") is None


def test_error_messages_hide_account_identifiers() -> None:
    cleaned = sanitize(GROQ_TPD_BODY)

    assert "01ksyp3jn6erbajmettfn0ex6h" not in cleaned
    assert "masqué" in cleaned


@pytest.mark.parametrize(
    "payload",
    [
        {"promptFeedback": {"blockReason": "SAFETY"}},
        {"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": [{"text": "{"}]}}]},
        {"candidates": [{"finishReason": "STOP", "content": {"parts": []}}]},
    ],
)
def test_blocked_or_truncated_gemini_answers_raise_llm_error(payload: dict[str, object]) -> None:
    with pytest.raises(LLMError):
        parse_response(payload, "gemini", 0.1)


def test_force_ipv4_binds_the_transport_to_an_ipv4_address() -> None:
    client = OpenAICompatibleClient("key", "m", tokens_per_minute=None, force_ipv4=True)

    pool = client._client._transport._pool  # type: ignore[attr-defined]
    assert pool._local_address == "0.0.0.0"


def test_cache_can_be_used_from_another_thread(tmp_path: Path) -> None:
    # Régression : l'API traite chaque requête dans un fil différent du fil de création
    import threading

    cached = CachedLLM(CountingLLM(), tmp_path / "cache.sqlite")
    results: list[str] = []

    worker = threading.Thread(
        target=lambda: results.append(cached.complete(MESSAGES, CompletionParams()).text)
    )
    worker.start()
    worker.join()
    cached.close()

    assert results == ["answer 1"]


def test_a_rejected_tool_call_does_not_block_the_next_call_for_a_minute() -> None:
    # Cas observé en ligne : après un appel d'outil refusé (HTTP 400), l'appel suivant
    # attendait 58 s, car la réservation complète du limiteur n'était jamais libérée
    rejected = httpx.Response(
        400, json={"error": {"code": "tool_use_failed", "message": "missing properties"}}
    )
    # Deux refus : la nouvelle tentative silencieuse échoue aussi, l'erreur remonte à l'agent
    responses = iter([rejected, rejected, chat_response("ok")])
    sleeps: list[float] = []
    client = OpenAICompatibleClient(
        "key",
        "m",
        tokens_per_minute=3000,
        transport=httpx.MockTransport(lambda _: next(responses)),
        sleep=sleeps.append,
    )
    params = CompletionParams(max_tokens=2500)

    with pytest.raises(LLMToolCallError, match="missing properties"):
        client.complete(MESSAGES, params)
    client.complete(MESSAGES, params)

    assert sleeps == []


def test_tool_calls_are_parsed_from_the_response() -> None:
    call = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}
    response = httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": None, "tool_calls": [call]}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        },
    )
    client = OpenAICompatibleClient(
        "key", "m", tokens_per_minute=None, transport=httpx.MockTransport(lambda _: response)
    )

    result = client.complete(MESSAGES, CompletionParams())

    assert result.text == ""
    assert result.tool_calls[0]["function"]["arguments"] == '{"a": 1}'


def test_unparseable_model_output_is_reported_as_a_correctable_error() -> None:
    rejected = httpx.Response(
        400, json={"error": {"code": "output_parse_failed", "message": "Parsing failed"}}
    )
    client = OpenAICompatibleClient(
        "key", "m", tokens_per_minute=None, transport=httpx.MockTransport(lambda _: rejected)
    )

    with pytest.raises(LLMToolCallError, match="Parsing failed"):
        client.complete(MESSAGES, CompletionParams())


def test_a_one_off_malformed_output_is_retried_silently() -> None:
    # Cas observé avec gpt-oss : faux outil « commentary » une fois, puis une réponse correcte
    rejected = httpx.Response(
        400,
        json={
            "error": {
                "code": "tool_use_failed",
                "message": "attempted to call tool 'commentary' which was not in request.tools",
            }
        },
    )
    responses = iter([rejected, chat_response("ok")])
    calls: list[int] = []

    def handler(_: httpx.Request) -> httpx.Response:
        calls.append(1)
        return next(responses)

    client = OpenAICompatibleClient(
        "key", "m", tokens_per_minute=None, transport=httpx.MockTransport(handler)
    )

    result = client.complete(MESSAGES, CompletionParams())

    assert result.text == "ok"
    assert len(calls) == 2
