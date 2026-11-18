"""Jeu de données FinanceBench : métadonnées, questions et PDF sources."""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx

logger = logging.getLogger(__name__)

FINANCEBENCH_RAW_URL = "https://raw.githubusercontent.com/patronus-ai/financebench/main"
QUESTIONS_FILE = "financebench_open_source.jsonl"
DOCUMENTS_FILE = "financebench_document_information.jsonl"

DOC_TYPE_LABELS = {
    "10k": "10-K",
    "10q": "10-Q",
    "8k": "8-K",
    "earnings": "Earnings release",
    "10k_annualreport": "Annual report",
}
_QUARTER_IN_NAME = re.compile(r"\d{4}(Q[1-4])")

# "questions" : seulement les documents cités par les 150 questions ; "all" : tout le corpus
Scope = Literal["questions", "all"]


@dataclass(frozen=True, slots=True)
class FinanceBenchDocument:
    doc_name: str
    company: str
    doc_type: str
    doc_period: int
    sector: str | None = None

    @property
    def doc_type_label(self) -> str:
        return DOC_TYPE_LABELS.get(self.doc_type.lower(), self.doc_type)

    @property
    def fiscal_quarter(self) -> str | None:
        # Le trimestre n'existe que dans le nom des 10-Q, par exemple AMD_2023Q2_10Q
        match = _QUARTER_IN_NAME.search(self.doc_name)
        return match.group(1) if match else None

    @property
    def period_label(self) -> str:
        quarter = self.fiscal_quarter
        return f"FY{self.doc_period} {quarter}" if quarter else f"FY{self.doc_period}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def download_metadata(raw_dir: Path, client: httpx.Client, refresh: bool = False) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in (QUESTIONS_FILE, DOCUMENTS_FILE):
        target = raw_dir / name
        if refresh or not target.exists():
            logger.info("Downloading %s", name)
            response = client.get(f"{FINANCEBENCH_RAW_URL}/data/{name}")
            response.raise_for_status()
            target.write_bytes(response.content)


def load_questions(raw_dir: Path) -> list[dict[str, Any]]:
    return read_jsonl(raw_dir / QUESTIONS_FILE)


def load_documents(raw_dir: Path) -> list[FinanceBenchDocument]:
    return [
        FinanceBenchDocument(
            doc_name=row["doc_name"],
            company=row["company"],
            doc_type=row["doc_type"],
            doc_period=int(row["doc_period"]),
            sector=row.get("gics_sector"),
        )
        for row in read_jsonl(raw_dir / DOCUMENTS_FILE)
    ]


def select_documents(raw_dir: Path, scope: Scope) -> list[FinanceBenchDocument]:
    documents = sorted(load_documents(raw_dir), key=lambda d: d.doc_name)
    if scope == "all":
        return documents
    used = {q["doc_name"] for q in load_questions(raw_dir)}
    return [d for d in documents if d.doc_name in used]


def pdf_url(doc_name: str) -> str:
    return f"{FINANCEBENCH_RAW_URL}/pdfs/{doc_name}.pdf"


def download_pdf(
    document: FinanceBenchDocument, pdf_dir: Path, client: httpx.Client, refresh: bool = False
) -> Path:
    """Télécharge le PDF s'il n'est pas déjà en cache et renvoie son chemin local."""
    pdf_dir.mkdir(parents=True, exist_ok=True)
    target = pdf_dir / f"{document.doc_name}.pdf"
    if refresh or not target.exists():
        logger.info("Downloading %s.pdf", document.doc_name)
        response = client.get(pdf_url(document.doc_name))
        response.raise_for_status()
        # Écriture atomique : un téléchargement interrompu ne laisse pas de PDF tronqué
        partial = target.with_suffix(".pdf.part")
        partial.write_bytes(response.content)
        partial.replace(target)
    return target
