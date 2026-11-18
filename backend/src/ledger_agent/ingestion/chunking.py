"""Découpage des pages en chunks, avec un en-tete de contexte et des identifiants stables."""

import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import pyarrow as pa

from ledger_agent.ingestion.financebench import FinanceBenchDocument
from ledger_agent.ingestion.parsing import PageText

# Espace de noms fixe : le même chunk garde le même identifiant d'une ingestion à l'autre
CHUNK_NAMESPACE = uuid.UUID("6f1c2a52-3f0e-4c1b-9a43-2d8e5b7c9e10")

DEFAULT_MAX_CHARS = 1800
DEFAULT_OVERLAP_CHARS = 250
# En dessous, la page est vide ou ne contient qu'un numéro de page
MIN_PAGE_CHARS = 40


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    doc_name: str
    company: str
    ticker: str | None
    doc_type: str
    fiscal_year: int
    fiscal_quarter: str | None
    page_index: int
    chunk_index: int
    header: str
    text: str

    @property
    def embedding_text(self) -> str:
        # L'en-tête donne au chunk le contexte que la page seule n'a pas (entreprise, année)
        return f"{self.header}\n{self.text}"


CHUNK_SCHEMA = pa.schema(
    [
        ("chunk_id", pa.string()),
        ("doc_name", pa.string()),
        ("company", pa.string()),
        ("ticker", pa.string()),
        ("doc_type", pa.string()),
        ("fiscal_year", pa.int32()),
        ("fiscal_quarter", pa.string()),
        ("page_index", pa.int32()),
        ("chunk_index", pa.int32()),
        ("header", pa.string()),
        ("text", pa.string()),
    ]
)


def make_chunk_id(doc_name: str, page_index: int, chunk_index: int) -> str:
    return str(uuid.uuid5(CHUNK_NAMESPACE, f"{doc_name}:{page_index}:{chunk_index}"))


def make_header(document: FinanceBenchDocument, page_index: int) -> str:
    return (
        f"{document.company} | {document.doc_type_label} | {document.period_label}"
        f" | page {page_index + 1}"
    )


def split_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Découpe sur les fins de ligne, avec un recouvrement des dernières lignes."""
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars doit être inférieur à max_chars")
    if len(text) <= max_chars:
        return [text]

    # Une ligne plus longue que l'espace disponible est coupée en morceaux
    room = max_chars - overlap_chars
    lines: list[str] = []
    for line in text.splitlines():
        lines.extend(line[i : i + room] for i in range(0, max(len(line), 1), room))

    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for line in lines:
        if current and size + len(line) + 1 > max_chars:
            pieces.append("\n".join(current))
            tail: list[str] = []
            tail_size = 0
            for previous in reversed(current):
                if tail_size + len(previous) + 1 > overlap_chars:
                    break
                tail.insert(0, previous)
                tail_size += len(previous) + 1
            current, size = tail, tail_size
        current.append(line)
        size += len(line) + 1
    if current:
        pieces.append("\n".join(current))
    return pieces


def chunk_document(
    document: FinanceBenchDocument,
    ticker: str | None,
    pages: Sequence[PageText],
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        if len(page.text) < MIN_PAGE_CHARS:
            continue
        header = make_header(document, page.page_index)
        for chunk_index, piece in enumerate(split_text(page.text, max_chars, overlap_chars)):
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(document.doc_name, page.page_index, chunk_index),
                    doc_name=document.doc_name,
                    company=document.company,
                    ticker=ticker,
                    doc_type=document.doc_type_label,
                    fiscal_year=document.doc_period,
                    fiscal_quarter=document.fiscal_quarter,
                    page_index=page.page_index,
                    chunk_index=chunk_index,
                    header=header,
                    text=piece,
                )
            )
    return chunks


def chunks_to_table(chunks: Sequence[Chunk]) -> pa.Table:
    return pa.Table.from_pylist([asdict(c) for c in chunks], schema=CHUNK_SCHEMA)


def table_to_chunks(table: pa.Table) -> list[Chunk]:
    return [Chunk(**row) for row in table.to_pylist()]
