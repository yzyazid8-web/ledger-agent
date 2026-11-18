from typing import Any

import duckdb
import pytest

from ledger_agent.ingestion.xbrl import (
    build_derived_tables,
    init_schema,
    parse_company_facts,
    upsert_company,
)
from ledger_agent.tools.financial_facts import FactStore

FY18 = "0000000001-19-000001"
FY19 = "0000000001-20-000001"


def annual(start: str, end: str, val: float, accn: str, fy: int, filed: str) -> dict[str, Any]:
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


PAYLOAD: dict[str, Any] = {
    "cik": 66740,
    "facts": {
        "us-gaap": {
            "PaymentsToAcquirePropertyPlantAndEquipment": {
                "label": "Payments to Acquire Property, Plant, and Equipment",
                "units": {
                    "USD": [
                        annual("2018-01-01", "2018-12-31", 1577e6, FY18, 2018, "2019-02-07"),
                        # Le 10-K suivant retraite le chiffre 2018
                        annual("2018-01-01", "2018-12-31", 1580e6, FY19, 2019, "2020-02-06"),
                        annual("2019-01-01", "2019-12-31", 1699e6, FY19, 2019, "2020-02-06"),
                    ]
                },
            },
            "RevenueFromContractWithCustomerExcludingAssessedTax": {
                "label": "Revenue from Contract with Customer, Excluding Assessed Tax",
                "units": {
                    "USD": [annual("2019-01-01", "2019-12-31", 32e9, FY19, 2019, "2020-02-06")]
                },
            },
        }
    },
}


@pytest.fixture
def store() -> FactStore:
    con = duckdb.connect(":memory:")
    init_schema(con)
    facts, concepts = parse_company_facts(PAYLOAD, "MMM")
    upsert_company(con, 66740, facts, concepts)
    build_derived_tables(con)
    return FactStore(con)


def test_get_fact_returns_original_value_and_flags_restatement(store: FactStore) -> None:
    fact = store.get_fact("MMM", "PaymentsToAcquirePropertyPlantAndEquipment", 2018)

    assert fact is not None
    assert fact.value == 1577e6
    assert fact.latest_value == 1580e6
    assert fact.accn == FY18
    assert fact.source_id == "xbrl:MMM:PaymentsToAcquirePropertyPlantAndEquipment:2018FY"


def test_missing_fact_returns_none(store: FactStore) -> None:
    assert store.get_fact("MMM", "Revenues", 2019) is None
    assert store.get_fact("MMM", "PaymentsToAcquirePropertyPlantAndEquipment", 2030) is None


def test_synonyms_map_labels_to_the_concept_actually_used(store: FactStore) -> None:
    revenue = store.search_concepts("total revenue", "MMM")
    capex = store.search_concepts("What was the capex?", "MMM")

    # « Revenues » est le synonyme préféré, mais l'entreprise ne le déclare pas
    assert revenue[0].concept == "RevenueFromContractWithCustomerExcludingAssessedTax"
    assert capex[0].concept == "PaymentsToAcquirePropertyPlantAndEquipment"


def test_unknown_ticker_has_no_concepts(store: FactStore) -> None:
    assert store.search_concepts("capex", "NOPE") == []
