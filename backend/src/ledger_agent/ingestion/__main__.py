"""Ligne de commande d'ingestion.

Utilisation :
    python -m ledger_agent.ingestion facts [--tickers MMM AMD] [--refresh]
    python -m ledger_agent.ingestion documents [--scope questions|all] [--refresh] [--rebuild]
"""

import argparse
import json
import logging

import duckdb
import httpx
import pyarrow.parquet as pq
from qdrant_client import QdrantClient

from ledger_agent.config import get_settings
from ledger_agent.ingestion.chunking import Chunk, chunk_document, chunks_to_table
from ledger_agent.ingestion.edgar_client import EdgarClient
from ledger_agent.ingestion.financebench import (
    Scope,
    download_metadata,
    download_pdf,
    select_documents,
)
from ledger_agent.ingestion.indexer import (
    FastEmbedEmbedder,
    build_bm25_index,
    ensure_collection,
    sync_dense_index,
)
from ledger_agent.ingestion.manifest import sha256_file, write_manifest
from ledger_agent.ingestion.parsing import extract_pages
from ledger_agent.ingestion.universe import find_company, select_companies
from ledger_agent.ingestion.xbrl import (
    build_derived_tables,
    init_schema,
    parse_company_facts,
    upsert_company,
)

logger = logging.getLogger("ledger_agent.ingestion")


def ingest_facts(tickers: list[str] | None, refresh: bool) -> None:
    settings = get_settings()
    raw_dir = settings.data_dir / "raw" / "sec" / "companyfacts"
    db_path = settings.data_dir / "processed" / "facts.duckdb"
    raw_dir.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    companies = select_companies(tickers)
    entries = []

    with EdgarClient.from_settings(settings) as client, duckdb.connect(str(db_path)) as con:
        init_schema(con)
        for company in companies:
            # Le JSON brut est mis en cache : il n'est retéléchargé qu'avec --refresh
            raw_path = raw_dir / f"CIK{company.cik10}.json"
            if refresh or not raw_path.exists():
                logger.info("Téléchargement des companyfacts de %s", company.ticker)
                raw_path.write_bytes(client.get_bytes(client.company_facts_url(company.cik)))

            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            facts, concepts = parse_company_facts(payload, company.ticker)
            n_rows = upsert_company(con, company.cik, facts, concepts)
            logger.info("%-5s %8d faits bruts", company.ticker, n_rows)
            entries.append(
                {
                    "ticker": company.ticker,
                    "cik": company.cik,
                    "source": client.company_facts_url(company.cik),
                    "file": raw_path.name,
                    "sha256": sha256_file(raw_path),
                    "raw_facts": n_rows,
                }
            )

        build_derived_tables(con)
        totals = {
            table: con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]  # type: ignore[index]
            for table in ("raw_facts", "facts", "filings", "concepts")
        }

    write_manifest(db_path.parent / "facts_manifest.json", "xbrl_facts", entries, totals=totals)
    logger.info("Base de faits prête dans %s : %s", db_path, totals)


def ingest_documents(scope: Scope, refresh: bool, rebuild: bool) -> None:
    settings = get_settings()
    raw_dir = settings.data_dir / "raw" / "financebench"
    pdf_dir = raw_dir / "pdfs"
    processed_dir = settings.data_dir / "processed"
    index_dir = settings.data_dir / "indexes"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Téléchargement, extraction et découpage
    all_chunks: list[Chunk] = []
    entries = []
    with httpx.Client(timeout=120.0, follow_redirects=True) as http:
        download_metadata(raw_dir, http, refresh)
        documents = select_documents(raw_dir, scope)
        logger.info("%d documents dans le périmètre '%s'", len(documents), scope)

        for document in documents:
            pdf_path = download_pdf(document, pdf_dir, http, refresh)
            pages = extract_pages(pdf_path)
            company = find_company(document.company)
            chunks = chunk_document(document, company.ticker if company else None, pages)
            all_chunks.extend(chunks)
            entries.append(
                {
                    "doc_name": document.doc_name,
                    "ticker": company.ticker if company else None,
                    "file": pdf_path.name,
                    "sha256": sha256_file(pdf_path),
                    "pages": len(pages),
                    "chunks": len(chunks),
                }
            )
            logger.info("%-28s %4d pages %5d chunks", document.doc_name, len(pages), len(chunks))

    pq.write_table(chunks_to_table(all_chunks), processed_dir / "chunks.parquet")

    # Index vectoriel (incrémental) puis index BM25 (reconstruit)
    embedder = FastEmbedEmbedder(cache_dir=settings.data_dir / "models")
    qdrant = QdrantClient(path=str(index_dir / "qdrant"))
    try:
        ensure_collection(qdrant, embedder.dimension, recreate=rebuild)
        added, removed = sync_dense_index(qdrant, all_chunks, embedder)
    finally:
        qdrant.close()
    build_bm25_index(all_chunks, index_dir / "bm25")

    totals = {
        "documents": len(entries),
        "pages": sum(e["pages"] for e in entries),  # type: ignore[misc]
        "chunks": len(all_chunks),
        "embedded_now": added,
        "removed_now": removed,
    }
    write_manifest(
        processed_dir / "documents_manifest.json",
        "financebench_documents",
        entries,
        scope=scope,
        totals=totals,
    )
    logger.info("Index documentaires prêts dans %s : %s", index_dir, totals)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m ledger_agent.ingestion")
    sub = parser.add_subparsers(dest="command", required=True)

    facts = sub.add_parser("facts", help="Télécharge les chiffres XBRL dans DuckDB")
    facts.add_argument("--tickers", nargs="*", help="Limite l'ingestion à ces tickers")
    facts.add_argument("--refresh", action="store_true", help="Retélécharge les JSON en cache")

    documents = sub.add_parser("documents", help="Indexe les PDF FinanceBench")
    documents.add_argument(
        "--scope",
        choices=["questions", "all"],
        default="questions",
        help="questions : documents cités par le benchmark ; all : corpus complet",
    )
    documents.add_argument("--refresh", action="store_true", help="Retélécharge les fichiers")
    documents.add_argument("--rebuild", action="store_true", help="Recrée l'index vectoriel")

    args = parser.parse_args()
    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "facts":
        ingest_facts(args.tickers, args.refresh)
    elif args.command == "documents":
        ingest_documents(args.scope, args.refresh, args.rebuild)


if __name__ == "__main__":
    main()
