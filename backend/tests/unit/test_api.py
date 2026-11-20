"""API avec un faux service : flux SSE, réponse JSON, erreurs et validation."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ledger_agent.api import main as api_main
from ledger_agent.api.main import create_app, sse

STEPS = [
    {"type": "step", "node": "plan", "ms": 10},
    {"type": "step", "node": "gather", "ms": 20},
    {"type": "step", "node": "synthesize", "ms": 30},
]
ANSWER = {"type": "answer", "final_answer": "$1,577 million", "abstained": False, "evidence": []}


class FakeService:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.events = events
        self.closed = False
        self.questions: list[str] = []

    @property
    def ready(self) -> bool:
        return True

    @property
    def config_name(self) -> str:
        return "agent"

    def companies(self) -> list[dict[str, str]]:
        return [{"ticker": "MMM", "name": "3M"}]

    def stream(self, question: str) -> Iterator[dict[str, Any]]:
        self.questions.append(question)
        yield from self.events

    def close(self) -> None:
        self.closed = True


def client_for(events: list[dict[str, Any]]) -> tuple[TestClient, FakeService]:
    service = FakeService(events)
    return TestClient(create_app(lambda: service)), service


def parse_sse(raw: str) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for block in raw.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_health_and_companies() -> None:
    client, service = client_for([])
    with client:
        assert client.get("/health").json() == {"status": "ok", "ready": True, "config": "agent"}
        assert client.get("/companies").json() == [{"ticker": "MMM", "name": "3M"}]
    # Les ressources de l'agent sont libérées à l'arrêt de l'API
    assert service.closed


def test_chat_streams_each_step_then_the_answer() -> None:
    client, service = client_for([*STEPS, ANSWER])
    with client, client.stream("POST", "/chat", json={"question": "3M capex FY2018?"}) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        events = parse_sse(response.read().decode("utf-8"))

    assert [name for name, _ in events] == ["step", "step", "step", "answer"]
    assert [data.get("node") for _, data in events[:3]] == ["plan", "gather", "synthesize"]
    assert events[-1][1]["final_answer"] == "$1,577 million"
    assert service.questions == ["3M capex FY2018?"]


def test_ask_returns_the_answer_and_its_steps() -> None:
    client, _ = client_for([*STEPS, ANSWER])
    with client:
        body = client.post("/ask", json={"question": "3M capex FY2018?"}).json()

    assert body["answer"]["final_answer"] == "$1,577 million"
    assert len(body["steps"]) == 3


@pytest.mark.parametrize(("code", "status"), [("quota", 429), ("llm", 502)])
def test_ask_maps_agent_errors_to_http_statuses(code: str, status: int) -> None:
    client, _ = client_for([{"type": "error", "code": code, "message": "indisponible"}])
    with client:
        response = client.post("/ask", json={"question": "3M capex FY2018?"})

    assert response.status_code == status
    assert response.json()["detail"] == "indisponible"


def test_question_is_validated() -> None:
    client, _ = client_for([])
    with client:
        assert client.post("/ask", json={"question": "?"}).status_code == 422
        assert client.post("/chat", json={}).status_code == 422


def test_sse_format_handles_non_ascii_text() -> None:
    assert sse({"type": "step", "node": "plan", "note": "élargi"}) == (
        'event: step\ndata: {"type": "step", "node": "plan", "note": "élargi"}\n\n'
    )


def test_benchmark_serves_the_ablation_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_main, "REPORTS_DIR", tmp_path)
    (tmp_path / "ablation-dev.json").write_text('{"split": "dev", "rows": []}', encoding="utf-8")
    client, _ = client_for([])
    with client:
        assert client.get("/benchmark").json() == {"split": "dev", "rows": []}
        assert client.get("/benchmark?split=test").status_code == 404
        assert client.get("/benchmark?split=all").status_code == 422
