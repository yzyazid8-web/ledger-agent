"""Chaîne d'évaluation complète (génération, reprise, notation, comparaison) sans réseau."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from ledger_agent.config import Settings
from ledger_agent.evaluation import compare, predict, report, runs, scoring
from ledger_agent.evaluation.config import RunConfig
from ledger_agent.evaluation.dataset import EvidencePage, Question
from ledger_agent.evaluation.judge import Verdict
from ledger_agent.llm.types import LLMError, LLMQuotaError, LLMResponse
from ledger_agent.pipeline import PipelineAnswer
from ledger_agent.retrieval.types import SearchOutcome

QUESTIONS = [
    Question(
        f"q{i}",
        f"What is metric {i}?",
        f"${i}00.00",
        "3M",
        "3M_2018_10K",
        "metrics-generated",
        (EvidencePage("3M_2018_10K", i),),
    )
    for i in range(1, 5)
]


def make_config(name: str) -> RunConfig:
    return RunConfig.model_validate(
        {
            "name": name,
            "pipeline": "naive_rag",
            "llm": {"provider": "groq", "model": "m"},
            "judge": {"provider": "gemini", "model": "j"},
        }
    )


class FakePipeline:
    """Répond juste, sauf scénario : erreur ponctuelle, puis quota épuisé."""

    def __init__(self, fail_ids: set[str], quota_ids: set[str], wrong_ids: set[str]) -> None:
        self.fail_ids, self.quota_ids, self.wrong_ids = fail_ids, quota_ids, wrong_ids
        self.calls: list[str] = []

    def run(self, question: str) -> PipelineAnswer:
        qid = "q" + question.split()[-1].rstrip("?")
        self.calls.append(qid)
        if qid in self.quota_ids:
            raise LLMQuotaError("quota", retry_after_s=600)
        if qid in self.fail_ids:
            raise LLMError("HTTP 500")
        value = 999 if qid in self.wrong_ids else int(qid[1:]) * 100
        return PipelineAnswer(
            text=f"Final answer: ${value} million",
            abstained=False,
            retrieved=[],
            cited=[],
            llm_calls=[LLMResponse(text="", model="m", prompt_tokens=10, completion_tokens=5)],
            retrieval_s=0.1,
            search=SearchOutcome(results=[], requested=None, applied=None),
        )


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")  # type: ignore[call-arg]
    state: dict[str, Any] = {"pipeline": None}

    @contextmanager
    def fake_build_pipeline(config: RunConfig, _: Settings) -> Iterator[FakePipeline]:
        yield state["pipeline"]

    class FakeJudge:
        def grade(self, question: str, reference: str, candidate: str) -> Verdict:
            return Verdict("correct", "ok")

    monkeypatch.setattr(runs, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(compare, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(report, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(predict, "get_settings", lambda: settings)
    monkeypatch.setattr(predict, "load_split_questions", lambda split: QUESTIONS)
    monkeypatch.setattr(predict, "build_pipeline", fake_build_pipeline)
    monkeypatch.setattr(scoring, "build_judge", lambda *args: FakeJudge())
    yield state


def test_errors_are_isolated_quota_stops_cleanly_and_resume_completes(
    sandbox: dict[str, Any],
) -> None:
    # Session 1 : q2 échoue, puis le quota tombe sur q3 : on s'arrête sans planter
    sandbox["pipeline"] = FakePipeline(fail_ids={"q2"}, quota_ids={"q3"}, wrong_ids=set())
    run_dir = predict.run_predictions(make_config("alpha"), "dev")
    first = scoring.score_run(run_dir)

    assert sandbox["pipeline"].calls == ["q1", "q2", "q3"]
    assert (first["questions"], first["errors"], first["complete"]) == (2, 1, False)
    assert "incomplète" in (run_dir / "summary.md").read_text(encoding="utf-8")

    # Session 2 : q1 n'est pas refaite, q2 est retentée, q3 et q4 sont traitées
    sandbox["pipeline"] = FakePipeline(fail_ids=set(), quota_ids=set(), wrong_ids=set())
    predict.run_predictions(make_config("alpha"), "dev")
    final = scoring.score_run(run_dir)

    assert sandbox["pipeline"].calls == ["q2", "q3", "q4"]
    assert (final["questions"], final["errors"], final["complete"]) == (4, 0, True)
    assert final["accuracy"] == 1.0
    assert final["judge"] is not None
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert len(meta["git"]) == 2


def test_compare_requires_same_questions_then_reports_mcnemar(sandbox: dict[str, Any]) -> None:
    sandbox["pipeline"] = FakePipeline(set(), set(), wrong_ids={"q1", "q2"})
    scoring.score_run(predict.run_predictions(make_config("alpha"), "dev"))

    sandbox["pipeline"] = FakePipeline(set(), set(), wrong_ids=set())
    beta_dir = predict.run_predictions(make_config("beta"), "dev", limit=2)
    scoring.score_run(beta_dir)
    with pytest.raises(compare.ComparisonError, match="mêmes questions"):
        compare.compare_runs("alpha", "beta", "dev")

    predict.run_predictions(make_config("beta"), "dev")
    scoring.score_run(beta_dir)
    report = compare.compare_runs("alpha", "beta", "dev")

    assert (report["accuracy_a"], report["accuracy_b"]) == (0.5, 1.0)
    assert (report["won_by_b"], report["won_by_a"]) == (2, 0)
    assert report["significant"] is False
    assert compare.report_path("alpha", "beta", "dev").exists()


def test_ablation_report_compares_only_complete_runs(sandbox: dict[str, Any]) -> None:
    sandbox["pipeline"] = FakePipeline(set(), set(), wrong_ids={"q1", "q2"})
    scoring.score_run(predict.run_predictions(make_config("naive"), "dev"))
    sandbox["pipeline"] = FakePipeline(set(), set(), wrong_ids=set())
    scoring.score_run(predict.run_predictions(make_config("hybrid"), "dev"))
    # L'agent n'a traité que 2 questions sur 4 : ses comparaisons restent en attente
    scoring.score_run(predict.run_predictions(make_config("agent"), "dev", limit=2))

    result = report.build_report("dev")

    assert [r["config"] for r in result["rows"]] == ["naive", "hybrid", "agent"]
    by_pair = {(c["a"], c["b"]): c for c in result["comparisons"]}
    assert by_pair[("naive", "hybrid")]["won_by_b"] == 2
    assert by_pair[("hybrid", "agent")]["status"].startswith("en attente")
    markdown = report.report_markdown(result)
    assert "| `agent` | 2/4 ⚠️ |" in markdown
