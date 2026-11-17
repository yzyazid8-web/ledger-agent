"""Chiffres XBRL (API companyfacts de la SEC) vers une base de faits DuckDB."""

from typing import Any

import duckdb
import pyarrow as pa

RAW_FACT_SCHEMA = pa.schema(
    [
        ("cik", pa.int64()),
        ("ticker", pa.string()),
        ("taxonomy", pa.string()),
        ("concept", pa.string()),
        ("unit", pa.string()),
        ("value", pa.float64()),
        ("period_start", pa.string()),
        ("period_end", pa.string()),
        ("fy", pa.int32()),
        ("fp", pa.string()),
        ("form", pa.string()),
        ("filed", pa.string()),
        ("accn", pa.string()),
        ("frame", pa.string()),
    ]
)

CONCEPT_SCHEMA = pa.schema(
    [
        ("taxonomy", pa.string()),
        ("concept", pa.string()),
        ("label", pa.string()),
        ("description", pa.string()),
    ]
)

PERIODIC_FORMS = ("10-K", "10-K/A", "10-KT", "10-Q", "10-Q/A", "20-F", "40-F")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS raw_facts (
    cik BIGINT NOT NULL,
    ticker VARCHAR NOT NULL,
    taxonomy VARCHAR NOT NULL,
    concept VARCHAR NOT NULL,
    unit VARCHAR NOT NULL,
    value DOUBLE NOT NULL,
    period_start DATE,
    period_end DATE NOT NULL,
    fy INTEGER,
    fp VARCHAR,
    form VARCHAR,
    filed DATE,
    accn VARCHAR NOT NULL,
    frame VARCHAR
);
CREATE TABLE IF NOT EXISTS concepts (
    taxonomy VARCHAR NOT NULL,
    concept VARCHAR NOT NULL,
    label VARCHAR,
    description VARCHAR,
    PRIMARY KEY (taxonomy, concept)
);
"""


def parse_company_facts(payload: dict[str, Any], ticker: str) -> tuple[pa.Table, pa.Table]:
    """Aplatit un JSON companyfacts en deux tables Arrow : (faits, concepts)."""
    cik = int(payload["cik"])
    facts: dict[str, list[Any]] = {name: [] for name in RAW_FACT_SCHEMA.names}
    concepts: dict[str, list[Any]] = {name: [] for name in CONCEPT_SCHEMA.names}

    for taxonomy, taxonomy_concepts in payload.get("facts", {}).items():
        for concept, body in taxonomy_concepts.items():
            concepts["taxonomy"].append(taxonomy)
            concepts["concept"].append(concept)
            concepts["label"].append(body.get("label"))
            concepts["description"].append(body.get("description"))

            for unit, entries in body.get("units", {}).items():
                for entry in entries:
                    if entry.get("val") is None or entry.get("end") is None:
                        continue
                    facts["cik"].append(cik)
                    facts["ticker"].append(ticker)
                    facts["taxonomy"].append(taxonomy)
                    facts["concept"].append(concept)
                    facts["unit"].append(unit)
                    facts["value"].append(float(entry["val"]))
                    facts["period_start"].append(entry.get("start"))
                    facts["period_end"].append(entry["end"])
                    facts["fy"].append(entry.get("fy"))
                    facts["fp"].append(entry.get("fp"))
                    facts["form"].append(entry.get("form"))
                    facts["filed"].append(entry.get("filed"))
                    facts["accn"].append(entry["accn"])
                    facts["frame"].append(entry.get("frame"))

    return (
        pa.table(facts, schema=RAW_FACT_SCHEMA),
        pa.table(concepts, schema=CONCEPT_SCHEMA),
    )


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(SCHEMA_SQL)


def upsert_company(
    con: duckdb.DuckDBPyConnection, cik: int, facts: pa.Table, concepts: pa.Table
) -> int:
    """Remplace tous les faits bruts d'une entreprise (idempotent). Renvoie le nombre de lignes."""
    con.execute("DELETE FROM raw_facts WHERE cik = ?", [cik])
    con.register("incoming_facts", facts)
    con.register("incoming_concepts", concepts)
    try:
        con.execute(
            """
            INSERT INTO raw_facts
            SELECT cik, ticker, taxonomy, concept, unit, value,
                   TRY_CAST(period_start AS DATE), CAST(period_end AS DATE),
                   fy, fp, form, TRY_CAST(filed AS DATE), accn, frame
            FROM incoming_facts
            """
        )
        con.execute(
            """
            INSERT INTO concepts
            SELECT DISTINCT ON (taxonomy, concept) taxonomy, concept, label, description
            FROM incoming_concepts
            ON CONFLICT DO NOTHING
            """
        )
    finally:
        con.unregister("incoming_facts")
        con.unregister("incoming_concepts")
    return int(facts.num_rows)


def build_derived_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Construit `filings` et `facts` à partir de `raw_facts`, avec les exercices recalculés."""
    forms = ", ".join(f"'{form}'" for form in PERIODIC_FORMS)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE filings AS
        SELECT
            cik,
            any_value(ticker) AS ticker,
            accn,
            any_value(form) AS form,
            any_value(fy) AS fy,
            any_value(fp) AS fp,
            min(filed) AS filed,
            -- dernière période de durée du dépôt ; les faits instantanés sont exclus car
            -- ceux de la page de garde (actions en circulation) sont datés après la clôture
            max(period_end) AS report_end
        FROM raw_facts
        WHERE period_start IS NOT NULL
          AND fy IS NOT NULL AND fp IS NOT NULL
          AND form IN ({forms})
        GROUP BY cik, accn
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE facts AS
        WITH typed AS (
            SELECT
                *,
                date_diff('day', period_start, period_end) AS duration_days,
                CASE
                    WHEN period_start IS NULL THEN 'instant'
                    WHEN date_diff('day', period_start, period_end) BETWEEN 350 AND 380
                        THEN 'annual'
                    WHEN date_diff('day', period_start, period_end) BETWEEN 80 AND 100
                        THEN 'quarter'
                    ELSE 'other'
                END AS period_type
            FROM raw_facts
        ),
        labels AS (
            -- le dépôt d'origine (le plus ancien) dont la période courante se termine à cette date
            SELECT
                cik,
                report_end AS period_end,
                arg_min(fy, filed) AS fiscal_year,
                arg_min(fp, filed) AS fiscal_period
            FROM filings
            GROUP BY cik, report_end
        )
        SELECT
            t.cik,
            t.ticker,
            t.taxonomy,
            t.concept,
            t.unit,
            t.value,
            t.period_start,
            t.period_end,
            t.duration_days,
            t.period_type,
            l.fiscal_year,
            CASE
                WHEN l.fiscal_period = 'FY' AND t.period_type = 'quarter' THEN 'Q4'
                ELSE l.fiscal_period
            END AS fiscal_period,
            t.form,
            t.filed,
            t.accn,
            t.frame,
            coalesce(f.report_end = t.period_end, false) AS is_current_period,
            row_number() OVER (
                PARTITION BY t.cik, t.taxonomy, t.concept, t.unit, t.period_start, t.period_end
                ORDER BY t.filed DESC, t.accn DESC
            ) = 1 AS is_latest
        FROM typed t
        LEFT JOIN labels l ON l.cik = t.cik AND l.period_end = t.period_end
        LEFT JOIN filings f ON f.cik = t.cik AND f.accn = t.accn
        """
    )
