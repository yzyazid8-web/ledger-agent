from typing import Any

import duckdb
import pytest

from ledger_agent.ingestion.xbrl import (
    build_derived_tables,
    init_schema,
    parse_company_facts,
    upsert_company,
)

ACCN_FY2018 = "0000000001-19-000001"
ACCN_FY2019 = "0000000001-20-000001"


def capex(start: str, end: str, val: int, accn: str, fy: int, filed: str) -> dict[str, Any]:
    return {
        "start": start,
        "end": end,
        "val": val,
        "accn": accn,
        "fy": fy,
        "fp": "FY",
        "form": "10-K",
        "filed": filed,
    }


# Reproduit le vrai JSON de 3M : le 10-K FY2019 reprend FY2018 comme comparatif
PAYLOAD: dict[str, Any] = {
    "cik": 66740,
    "entityName": "3M CO",
    "facts": {
        "us-gaap": {
            "PaymentsToAcquirePropertyPlantAndEquipment": {
                "label": "Payments to Acquire Property, Plant, and Equipment",
                "description": "Cash outflow for capex.",
                "units": {
                    "USD": [
                        capex("2017-01-01", "2017-12-31", 1373, ACCN_FY2018, 2018, "2019-02-07"),
                        capex("2018-01-01", "2018-12-31", 1577, ACCN_FY2018, 2018, "2019-02-07"),
                        capex("2018-01-01", "2018-12-31", 1577, ACCN_FY2019, 2019, "2020-02-06"),
                        capex("2019-01-01", "2019-12-31", 1699, ACCN_FY2019, 2019, "2020-02-06"),
                        capex("2019-10-01", "2019-12-31", 450, ACCN_FY2019, 2019, "2020-02-06"),
                    ]
                },
            },
            "Assets": {
                "label": "Assets",
                "units": {
                    "USD": [
                        {
                            "end": "2018-12-31",
                            "val": 36500,
                            "accn": ACCN_FY2018,
                            "fy": 2018,
                            "fp": "FY",
                            "form": "10-K",
                            "filed": "2019-02-07",
                        }
                    ]
                },
            },
        },
        "dei": {
            "EntityCommonStockSharesOutstanding": {
                "label": "Entity Common Stock, Shares Outstanding",
                "units": {
                    "shares": [
                        {
                            "end": "2019-01-31",
                            "val": 575000000,
                            "accn": ACCN_FY2018,
                            "fy": 2018,
                            "fp": "FY",
                            "form": "10-K",
                            "filed": "2019-02-07",
                        }
                    ]
                },
            }
        },
    },
}


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(":memory:")
    init_schema(connection)
    facts, concepts = parse_company_facts(PAYLOAD, "MMM")
    upsert_company(connection, 66740, facts, concepts)
    build_derived_tables(connection)
    return connection


def test_parse_flattens_all_entries() -> None:
    facts, concepts = parse_company_facts(PAYLOAD, "MMM")

    assert facts.num_rows == 7
    assert concepts.num_rows == 3


def test_comparative_fact_gets_its_own_fiscal_year(con: duckdb.DuckDBPyConnection) -> None:
    rows = con.execute(
        """
        SELECT accn, fiscal_year, fiscal_period, is_current_period, is_latest
        FROM facts
        WHERE concept = 'PaymentsToAcquirePropertyPlantAndEquipment'
          AND period_end = DATE '2018-12-31'
        ORDER BY accn
        """
    ).fetchall()

    # Les deux lignes décrivent FY2018, même celle déposée dans le 10-K FY2019
    assert rows == [
        (ACCN_FY2018, 2018, "FY", True, False),
        (ACCN_FY2019, 2018, "FY", False, True),
    ]


def test_period_types_and_q4(con: duckdb.DuckDBPyConnection) -> None:
    rows = con.execute(
        """
        SELECT period_type, fiscal_period
        FROM facts
        WHERE concept = 'PaymentsToAcquirePropertyPlantAndEquipment'
          AND period_end = DATE '2019-12-31'
        ORDER BY period_type
        """
    ).fetchall()

    assert rows == [("annual", "FY"), ("quarter", "Q4")]


def test_balance_sheet_instant_is_labelled(con: duckdb.DuckDBPyConnection) -> None:
    row = con.execute(
        "SELECT period_type, fiscal_year FROM facts WHERE concept = 'Assets'"
    ).fetchone()

    assert row == ("instant", 2018)


def test_cover_page_fact_does_not_shift_report_end(con: duckdb.DuckDBPyConnection) -> None:
    report_end = con.execute(
        "SELECT report_end FROM filings WHERE accn = ?", [ACCN_FY2018]
    ).fetchone()

    assert str(report_end[0]) == "2018-12-31"  # type: ignore[index]


def test_upsert_is_idempotent(con: duckdb.DuckDBPyConnection) -> None:
    facts, concepts = parse_company_facts(PAYLOAD, "MMM")
    upsert_company(con, 66740, facts, concepts)

    count = con.execute("SELECT count(*) FROM raw_facts").fetchone()

    assert count == (7,)
