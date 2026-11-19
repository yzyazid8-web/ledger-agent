import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from ledger_agent.config import Settings
from ledger_agent.evaluation import runs
from ledger_agent.evaluation.config import RunConfig, load_run_config
from ledger_agent.evaluation.dataset import (
    EvidencePage,
    Question,
    SplitError,
    load_or_create_split,
    make_split,
    validate_split,
)
from ledger_agent.evaluation.judge import Verdict, parse_verdict
from ledger_agent.evaluation.scoring import score_record, summarize
from ledger_agent.evaluation.stats import (
    bootstrap_ci,
    cohen_kappa,
    compare_paired,
    mcnemar_exact_p,
)

TYPES = ["metrics-generated", "novel-generated", "domain-relevant"]


def make_questions(n_docs: int = 60, per_doc: int = 3) -> list[Question]:
    return [
        Question(
            id=f"q{d:02d}-{i}",
            question="?",
            answer="1",
            company=f"C{d % 20}",
            doc_name=f"DOC{d:02d}",
            question_type=TYPES[(d + i) % 3],
            evidence=(EvidencePage(f"DOC{d:02d}", 1),),
        )
        for d in range(n_docs)
        for i in range(per_doc)
    ]


# Découpage


def test_split_groups_whole_documents_and_is_balanced() -> None:
    questions = make_questions()
    split = make_split(questions, dev_size=50)
    doc_of = {q.id: q.doc_name for q in questions}
    types = {q.id: q.question_type for q in questions}

    assert {doc_of[i] for i in split["dev"]}.isdisjoint({doc_of[i] for i in split["test"]})
    assert 45 <= len(split["dev"]) <= 50
    assert max(Counter(types[i] for i in split["dev"]).values()) <= 50 / 3 + 2
    assert make_split(questions, dev_size=50) == split


def test_split_validation_rejects_old_format_and_document_overlap() -> None:
    questions = make_questions(n_docs=2, per_doc=2)
    with pytest.raises(SplitError, match="ancien"):
        validate_split({"dev": ["q00-0"], "test": []}, questions)

    overlapping = {
        "strategy": "doc_name",
        "dev": ["q00-0"],
        "test": ["q00-1", "q01-0", "q01-1"],
    }
    with pytest.raises(SplitError, match="même document"):
        validate_split(overlapping, questions)


def test_split_file_is_frozen_once_written(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    first = load_or_create_split(path, make_questions())

    assert load_or_create_split(path, make_questions()) == first


# Statistiques


def test_bootstrap_ci_contains_the_mean() -> None:
    low, high = bootstrap_ci([True] * 21 + [False] * 29)

    assert low < 0.42 < high
    assert high - low > 0.2  # n = 50 : l'incertitude est large


def test_mcnemar_matches_known_values() -> None:
    # 5 contre 2 désaccords : p ≈ 0,45, comme calculé par l'audit
    assert mcnemar_exact_p(2, 5) == pytest.approx(0.453, abs=0.001)
    assert mcnemar_exact_p(0, 10) < 0.01
    assert mcnemar_exact_p(0, 0) == 1.0


def test_paired_comparison_counts_disagreements() -> None:
    result = compare_paired([True, True, False, False], [True, False, True, True])

    assert (result.only_a, result.only_b) == (1, 2)
    assert result.difference == pytest.approx(0.25)


def test_cohen_kappa() -> None:
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert cohen_kappa([True, True], [True, True]) is None


# Configuration


def test_config_rejects_typos(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(
        'name = "bad"\npipeline = "naive_rag"\n[retrieval]\ntop_kk = 5\n'
        '[llm]\nprovider = "groq"\nmodel = "m"\n[judge]\nprovider = "gemini"\nmodel = "j"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="top_kk"):
        load_run_config(path)


def make_config(top_k: int = 5) -> RunConfig:
    return RunConfig.model_validate(
        {
            "name": "unit",
            "pipeline": "naive_rag",
            "retrieval": {"top_k": top_k},
            "llm": {"provider": "groq", "model": "m"},
            "judge": {"provider": "gemini", "model": "j"},
        }
    )


def test_resume_is_refused_when_the_config_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runs, "EVAL_DIR", tmp_path)
    settings = Settings(_env_file=None, data_dir=tmp_path)  # type: ignore[call-arg]

    run_dir = runs.prepare_run(
        make_config(top_k=5), "dev", ["q1"], settings, expected=1, prompt_sha="p"
    )
    # Même configuration : la reprise est acceptée et le commit de la session est noté
    runs.prepare_run(make_config(top_k=5), "dev", ["q1"], settings, expected=1, prompt_sha="p")
    assert len(json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["git"]) == 2

    with pytest.raises(runs.RunMismatchError, match="changé"):
        runs.prepare_run(make_config(top_k=10), "dev", ["q1"], settings, expected=1, prompt_sha="p")


# Notation


class FakeJudge:
    def __init__(self, correct: bool) -> None:
        self.correct = correct

    def grade(self, question: str, reference: str, candidate: str) -> Verdict:
        return Verdict("correct" if self.correct else "incorrect", "raison")


def make_record(gold: str, prediction: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": extra.get("id", "q1"),
        "question_type": "metrics-generated",
        "question": "What is the FY2018 capex for 3M?",
        "gold_answer": gold,
        "gold_pages": [["3M_2018_10K", 59]],
        "prediction": prediction,
        "retrieved": [{"doc_name": "3M_2018_10K", "page_index": 59}],
        "cited": [["3M_2018_10K", 59]],
        "search": {"relaxed": False},
        "llm": {"prompt_tokens": 100, "completion_tokens": 20},
        "timing": {"total_s": 1.5},
    }


def test_numeric_answer_is_scored_by_rules_and_also_judged() -> None:
    score = score_record(make_record("$1577.00", "Final answer: $1,577 million"), FakeJudge(False))

    assert (score["kind"], score["correct"], score["judge_correct"]) == ("numeric", True, False)
    assert score["latency_s"] == 1.5


def test_errors_and_unjudged_answers_count_as_failures() -> None:
    scores = [
        score_record(make_record("$1577.00", "Final answer: $1,577 million"), None),
        score_record(make_record("Data Center", "Final answer: Data Center"), None),
        score_record({**make_record("$1", ""), "error": "HTTP 500"}, None),
    ]
    summary = summarize(scores, expected=4)

    assert summary["accuracy"] == pytest.approx(1 / 3, abs=1e-3)
    assert (summary["errors"], summary["unjudged"], summary["complete"]) == (1, 1, False)


def test_judge_agreement_is_reported() -> None:
    scores = [
        score_record(
            make_record("$1577.00", "Final answer: $1,577 million", id="a"), FakeJudge(True)
        ),
        score_record(make_record("$1577.00", "Final answer: $9 million", id="b"), FakeJudge(False)),
    ]

    agreement = summarize(scores, expected=2)["judge_agreement"]

    assert agreement == {"n": 2, "agreement": 1.0, "kappa": 1.0}


def test_parse_verdict_handles_markdown_and_garbage() -> None:
    assert (
        parse_verdict('```json\n{"verdict": "Correct", "reason": "ok"}\n```').verdict == "correct"
    )
    assert parse_verdict("not json").verdict == "invalid"
    assert parse_verdict('{"verdict": "maybe"}').verdict == "invalid"


def test_adding_an_optional_section_keeps_existing_fingerprints() -> None:
    # Régression : ajouter la section [agent] au schéma avait invalidé les exécutions en cours
    import hashlib

    config = make_config()
    legacy = config.model_dump_json(include={"pipeline", "retrieval", "llm"}) + "p"

    assert config.generation_fingerprint("p") == hashlib.sha256(legacy.encode()).hexdigest()[:16]
