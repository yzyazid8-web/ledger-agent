import json
from pathlib import Path

from ledger_agent.ingestion.financebench import (
    DOCUMENTS_FILE,
    QUESTIONS_FILE,
    FinanceBenchDocument,
    select_documents,
)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def make_raw_dir(tmp_path: Path) -> Path:
    write_jsonl(
        tmp_path / DOCUMENTS_FILE,
        [
            {"doc_name": "3M_2018_10K", "company": "3M", "doc_type": "10k", "doc_period": 2018},
            {"doc_name": "AMD_2023Q2_10Q", "company": "AMD", "doc_type": "10q", "doc_period": 2023},
            {"doc_name": "3M_2015_10K", "company": "3M", "doc_type": "10k", "doc_period": 2015},
        ],
    )
    write_jsonl(
        tmp_path / QUESTIONS_FILE,
        [{"doc_name": "3M_2018_10K"}, {"doc_name": "AMD_2023Q2_10Q"}, {"doc_name": "3M_2018_10K"}],
    )
    return tmp_path


def test_scope_questions_keeps_only_cited_documents(tmp_path: Path) -> None:
    documents = select_documents(make_raw_dir(tmp_path), "questions")

    assert [d.doc_name for d in documents] == ["3M_2018_10K", "AMD_2023Q2_10Q"]


def test_scope_all_keeps_everything(tmp_path: Path) -> None:
    assert len(select_documents(make_raw_dir(tmp_path), "all")) == 3


def test_labels_for_annual_and_quarterly_documents() -> None:
    annual = FinanceBenchDocument("3M_2018_10K", "3M", "10k", 2018)
    quarterly = FinanceBenchDocument("AMD_2023Q2_10Q", "AMD", "10q", 2023)

    assert (annual.doc_type_label, annual.period_label) == ("10-K", "FY2018")
    assert (quarterly.doc_type_label, quarterly.period_label) == ("10-Q", "FY2023 Q2")
