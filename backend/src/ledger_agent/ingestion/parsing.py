"""Extraction du texte des PDF, page par page."""

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

# Un grand espace au milieu d'une ligne sépare presque toujours deux colonnes de tableau
_COLUMN_GAP = re.compile(r" {3,}")
_BLANK_LINES = re.compile(r"\n{3,}")


@dataclass(frozen=True, slots=True)
class PageText:
    # Index à partir de 0, même convention que `evidence_page_num` dans FinanceBench
    page_index: int
    text: str


def clean_page_text(raw: str) -> str:
    lines = [_COLUMN_GAP.sub(" | ", line.strip()) for line in raw.splitlines()]
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def extract_pages(pdf_path: Path) -> list[PageText]:
    with pymupdf.open(pdf_path) as document:  # type: ignore[no-untyped-call]
        return [
            PageText(page_index=index, text=clean_page_text(page.get_text(sort=True)))
            for index, page in enumerate(document)
        ]
