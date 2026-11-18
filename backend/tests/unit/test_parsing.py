from pathlib import Path

import pymupdf

from ledger_agent.ingestion.parsing import clean_page_text, extract_pages


def test_column_gaps_become_separators() -> None:
    raw = "   Purchases of PP&E        (1,577)      (1,373)   \n\n\n\nNext line"

    assert clean_page_text(raw) == "Purchases of PP&E | (1,577) | (1,373)\n\nNext line"


def test_extract_pages_keeps_zero_based_index(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    with pymupdf.open() as document:
        for text in ("First page", "Second page"):
            page = document.new_page()
            page.insert_text((72, 72), text)
        document.save(pdf_path)

    pages = extract_pages(pdf_path)

    assert [(p.page_index, p.text) for p in pages] == [(0, "First page"), (1, "Second page")]
