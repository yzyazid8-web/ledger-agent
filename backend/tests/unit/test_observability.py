"""Suivi en production (Langfuse) : inactif par défaut, sans effet sur les réponses, robuste."""

from contextlib import ExitStack, contextmanager
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import SecretStr
from test_react import GOOD_RUN, QUESTION, make_agent

from ledger_agent import observability
from ledger_agent.assembly import build_llm
from ledger_agent.config import Settings
from ledger_agent.llm.cache import CachedLLM
from ledger_agent.llm.types import CompletionParams, LLMError, LLMResponse, Message
from ledger_agent.observability import (
    LangfuseTracer,
    NullTracer,
    QuestionTrace,
    TracedLLM,
    anonymous_session,
    build_tracer,
    question_scores,
)
from ledger_agent.run_config import LLMConfig


class RecordingTracer:
    """Faux tracer : garde en mémoire ce qui aurait été envoyé à Langfuse."""

    def __init__(self) -> None:
        self.generations: list[dict[str, Any]] = []

    @contextmanager
    def question(self, question: str, session_id: str | None, metadata: dict[str, str]) -> Any:
        yield QuestionTrace()

    @contextmanager
    def generation(self, model: str, messages: list[Message], params: CompletionParams) -> Any:
        record: dict[str, Any] = {"model": model, "tools": len(params.tools)}
        self.generations.append(record)

        class Record:
            def done(self, response: LLMResponse) -> None:
                record["tokens"] = response.prompt_tokens + response.completion_tokens

            def failed(self, error: Exception) -> None:
                record["error"] = str(error)

        yield Record()

    def flush(self) -> None:
        return None


# Activation


def test_without_keys_nothing_is_traced(monkeypatch: pytest.MonkeyPatch) -> None:
    def must_not_be_created(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Langfuse ne doit pas être initialisé sans clés")

    monkeypatch.setattr(observability, "LangfuseTracer", must_not_be_created)

    assert isinstance(build_tracer(Settings(langfuse_public_key=None)), NullTracer)
    assert isinstance(
        build_tracer(Settings(langfuse_public_key="pk", langfuse_secret_key=SecretStr(""))),
        NullTracer,
    )


def test_a_failing_langfuse_setup_falls_back_to_no_tracing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("serveur injoignable")

    monkeypatch.setattr(observability, "LangfuseTracer", broken)
    settings = Settings(langfuse_public_key="pk", langfuse_secret_key=SecretStr("sk"))

    assert isinstance(build_tracer(settings), NullTracer)


def test_evaluation_builds_untraced_clients_and_the_api_traced_ones(tmp_path: Path) -> None:
    settings = Settings(groq_api_key=SecretStr("gsk"), data_dir=tmp_path)
    config = LLMConfig(provider="groq", model="openai/gpt-oss-120b")
    with ExitStack() as stack:
        plain = build_llm(config, settings, stack)  # appel de l'évaluation : pas de tracer
        traced = build_llm(config, settings, stack, RecordingTracer())

    assert isinstance(plain, CachedLLM)
    assert isinstance(traced, TracedLLM)


# Enregistrement


def test_each_model_call_of_the_agent_is_recorded_and_answers_are_unchanged() -> None:
    plain_agent, _ = make_agent(list(GOOD_RUN))
    traced_agent, fake = make_agent(list(GOOD_RUN))
    tracer = RecordingTracer()
    traced_agent._llm = TracedLLM(fake, tracer)

    plain = plain_agent.run(QUESTION)
    traced = traced_agent.run(QUESTION)

    assert traced.text == plain.text
    assert len(tracer.generations) == len(traced.llm_calls) == 4
    assert all(g["model"] == "fake" and "tokens" in g for g in tracer.generations)


def test_a_failed_model_call_is_recorded_then_raised() -> None:
    class Failing:
        model = "fake"

        def complete(self, messages: list[Message], params: CompletionParams) -> LLMResponse:
            raise LLMError("HTTP 500")

    tracer = RecordingTracer()

    with pytest.raises(LLMError):
        TracedLLM(Failing(), tracer).complete([], CompletionParams())
    assert tracer.generations[0]["error"] == "HTTP 500"


def test_langgraph_callbacks_reach_the_agent_graph() -> None:
    class Counter(BaseCallbackHandler):
        def __init__(self) -> None:
            self.starts = 0

        def on_chain_start(self, *args: Any, **kwargs: Any) -> None:
            self.starts += 1

    agent, _ = make_agent(list(GOOD_RUN))
    counter = Counter()

    events = list(agent.stream(QUESTION, callbacks=[counter]))

    assert events[-1]["type"] == "answer"
    assert counter.starts > 0


# Robustesse : une panne de Langfuse ne doit jamais casser une réponse


class BrokenClient:
    def start_as_current_observation(self, **kwargs: Any) -> Any:
        raise RuntimeError("Langfuse en panne")

    def flush(self) -> None:
        raise RuntimeError("Langfuse en panne")


def broken_tracer() -> LangfuseTracer:
    tracer = object.__new__(LangfuseTracer)  # sans connexion réelle
    tracer._client = BrokenClient()
    tracer._public_key = "pk"
    return tracer


def test_a_langfuse_outage_does_not_break_model_calls_or_questions() -> None:
    tracer = broken_tracer()
    agent, fake = make_agent(list(GOOD_RUN))
    agent._llm = TracedLLM(fake, tracer)

    with tracer.question(QUESTION, None, {}) as trace:
        events = list(agent.stream(QUESTION, callbacks=trace.callbacks or None))
        trace.finish({"ok": True}, {"tours": 4.0})
    tracer.flush()

    assert events[-1]["type"] == "answer"
    assert trace.callbacks == []


# Confidentialité et bilan


def test_sessions_are_anonymous_stable_for_a_day_and_renewed_daily() -> None:
    monday, tuesday = date(2026, 11, 23), date(2026, 11, 24)

    first = anonymous_session("203.0.113.7", monday)

    assert first == anonymous_session("203.0.113.7", monday)
    assert first != anonymous_session("203.0.113.7", tuesday)
    assert first != anonymous_session("198.51.100.1", monday)
    assert "203.0.113.7" not in (first or "")
    assert anonymous_session(None) is None


def test_question_scores_summarise_the_answer_or_the_error() -> None:
    steps = [
        {"type": "step", "node": "agent"},
        {"type": "step", "node": "tool"},
        {"type": "step", "node": "verify", "ok": False},
        {"type": "step", "node": "agent"},
        {"type": "step", "node": "verify", "ok": True},
    ]
    answer = {"type": "answer", "final_answer": "14.86%", "abstained": False, "tokens": 5300}

    output, scores = question_scores([*steps, answer])
    _, failure = question_scores([{"type": "error", "code": "quota", "message": "…"}])

    assert output == {"final_answer": "14.86%", "abstained": False}
    assert scores == {
        "abstention": 0.0,
        "tours": 2.0,
        "tokens": 5300.0,
        "refus_verificateur": 1.0,
    }
    assert failure == {"erreur": "quota"}
