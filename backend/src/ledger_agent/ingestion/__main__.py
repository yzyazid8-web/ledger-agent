"""Ligne de commande d'ingestion.

Utilisation :
    python -m ledger_agent.ingestion facts [--tickers MMM AMD] [--refresh]
"""

import argparse
import json
import logging

import duckdb

from ledger_agent.config import get_settings
from ledger_agent.ingestion.edgar_client import EdgarClient
from ledger_agent.ingestion.manifest import sha256_file, write_manifest
from ledger_agent.ingestion.universe import select_companies
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
                logger.info("Downloading companyfacts for %s", company.ticker)
                raw_path.write_bytes(client.get_bytes(client.company_facts_url(company.cik)))

            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            facts, concepts = parse_company_facts(payload, company.ticker)
            n_rows = upsert_company(con, company.cik, facts, concepts)
            logger.info("%-5s %8d raw facts", company.ticker, n_rows)
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
    logger.info("Fact store ready at %s: %s", db_path, totals)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m ledger_agent.ingestion")
    sub = parser.add_subparsers(dest="command", required=True)

    facts = sub.add_parser("facts", help="Télécharge les chiffres XBRL dans DuckDB")
    facts.add_argument("--tickers", nargs="*", help="Limite l'ingestion à ces tickers")
    facts.add_argument("--refresh", action="store_true", help="Retélécharge les JSON en cache")

    args = parser.parse_args()
    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "facts":
        ingest_facts(args.tickers, args.refresh)


if __name__ == "__main__":
    main()
