from collections import Counter
from pathlib import Path

from ledger_agent.evaluation.dataset import (
    EvidencePage,
    Question,
    load_or_create_split,
    make_split,
)
from ledger_agent.evaluation.judge import parse_verdict
from ledger_agent.evaluation.runner import score_record, summarize


def make_questions() -> list[Question]:
    types = ["metrics-generated", "novel-generated", "domain-relevant"]
    return [
        Question(
            id=f"q{i:03d}",
            question="?",
            answer="1",
            company="3M",
            doc_name="3M_2018_10K",
            question_type=types[i % 3],
            evidence=(EvidencePage("3M_2018_10K", 1),),
        )
        for i in range(150)
    ]


def test_split_is_stratified_disjoint_and_reproducible() -> None:
    questions = make_questions()
    types = {q.id: q.question_type for q in questions}

    split = make_split(questions)

    assert len(split["dev"]) == 50
    assert len(split["test"]) == 100
    assert set(split["dev"]).isdisjoint(split["test"])
    assert sorted(Counter(types[i] for i in split["dev"]).values()) == [16, 17, 17]
    assert make_split(questions) == split


def test_split_file_is_frozen_once_written(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    first = load_or_create_split(path, make_questions())

    second = load_or_create_split(path, make_questions()[:10])

    assert second == first


def test_parse_verdict_handles_markdown_and_garbage() -> None:
    assert (
        parse_verdict('```json\n{"verdict": "Correct", "reason": "ok"}\n```').verdict == "correct"
    )
    assert parse_verdict("not json").verdict == "invalid"
    assert parse_verdict('{"verdict": "maybe"}').verdict == "invalid"


def make_record(gold: str, prediction: str) -> dict[str, object]:
    return {
        "id": "q1",
        "question_type": "metrics-generated",
        "question": "What is the FY2018 capex for 3M?",
        "gold_answer": gold,
        "gold_pages": [["3M_2018_10K", 59]],
        "prediction": prediction,
        "retrieved": [{"doc_name": "3M_2018_10K", "page_index": 59}],
        "cited": [["3M_2018_10K", 59]],
        "llm": {"prompt_tokens": 100, "completion_tokens": 20, "latency_s": 1.0},
    }


def test_score_record_numeric_without_judge() -> None:
    score = score_record(make_record("$1577.00", "Final answer: $1,577 million"), judge=None)

    assert score["kind"] == "numeric"
    assert score["correct"] is True
    assert score["page_hit"] is True
    assert score["citation_precision"] == 1.0


def test_text_answers_stay_unjudged_without_judge() -> None:
    scores = [
        score_record(make_record("Data Center", "Final answer: Data Center segment"), None),
        score_record(make_record("$1577.00", "INSUFFICIENT_CONTEXT"), None),
    ]
    summary = summarize(scores)

    assert scores[0]["correct"] is None
    assert scores[1]["abstained"] is True
    assert summary["judged"] == 1
    assert summary["accuracy"] == 0.0
    assert summary["abstention_rate"] == 0.5
