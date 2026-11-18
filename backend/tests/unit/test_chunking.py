import itertools

import pytest

from ledger_agent.ingestion.chunking import (
    chunk_document,
    chunks_to_table,
    split_text,
    table_to_chunks,
)
from ledger_agent.ingestion.financebench import FinanceBenchDocument
from ledger_agent.ingestion.parsing import PageText

DOCUMENT = FinanceBenchDocument("3M_2018_10K", "3M", "10k", 2018)


def test_short_text_is_a_single_piece() -> None:
    assert split_text("hello", max_chars=100, overlap_chars=10) == ["hello"]


def test_long_text_respects_size_and_overlaps() -> None:
    text = "\n".join(f"line {i:03d} " + "x" * 40 for i in range(60))

    pieces = split_text(text, max_chars=500, overlap_chars=120)

    assert len(pieces) > 1
    assert all(len(p) <= 500 for p in pieces)
    # La dernière ligne d'un morceau est reprise au début du suivant
    for first, second in itertools.pairwise(pieces):
        assert first.splitlines()[-1] in second.splitlines()


def test_very_long_line_is_cut() -> None:
    pieces = split_text("y" * 1200, max_chars=500, overlap_chars=100)

    assert all(len(p) <= 500 for p in pieces)
    assert "".join(pieces).count("y") >= 1200


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValueError):
        split_text("abc", max_chars=10, overlap_chars=10)


def test_chunk_document_metadata_and_stable_ids() -> None:
    pages = [
        PageText(0, "3"),  # page quasi vide, ignorée
        PageText(59, "Consolidated Statement of Cash Flows\nPurchases of PP&E | (1,577)"),
    ]

    first = chunk_document(DOCUMENT, "MMM", pages)
    second = chunk_document(DOCUMENT, "MMM", pages)

    assert len(first) == 1
    chunk = first[0]
    assert chunk.page_index == 59
    assert chunk.header == "3M | 10-K | FY2018 | page 60"
    assert chunk.embedding_text.startswith("3M | 10-K | FY2018 | page 60\n")
    assert chunk.chunk_id == second[0].chunk_id


def test_parquet_roundtrip() -> None:
    chunks = chunk_document(DOCUMENT, "MMM", [PageText(3, "Some long enough text " * 5)])

    assert table_to_chunks(chunks_to_table(chunks)) == chunks
