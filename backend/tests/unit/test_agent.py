"""Agent complet avec de faux LLM : planification, outils, calculs, vérification, abstention."""

import json
from typing import Any

import duckdb
import pytest

from ledger_agent.agent.evidence import Evidence
from ledger_agent.agent.graph import FinancialAgent
from ledger_agent.agent.nodes import (
    AgentConfig,
    AgentRuntime,
    fill_placeholders,
    json_call,
    run_calculations,
)
from ledger_agent.agent.schemas import Plan, Synthesis
from ledger_agent.agent.verify import verify_numbers
from ledger_agent.ingestion.chunking import Chunk
from ledger_agent.ingestion.xbrl import (
    build_derived_tables,
    init_schema,
    parse_company_facts,
    upsert_company,
)
from ledger_agent.llm.types import CompletionParams, LLMResponse, Message
from ledger_agent.retrieval.types import RetrievedChunk, SearchFilters
from ledger_agent.tools.calculator import calculate
from ledger_agent.tools.financial_facts import FactStore
from ledger_agent.tools.search_filings import SearchFilingsTool

QUESTION = "What is the FY2018 capital expenditure amount (in USD millions) for 3M?"


class ScriptedLLM:
    """Faux LLM : renvoie la réponse associée au premier mot-clé trouvé dans le prompt système."""

    def __init__(self, model: str, replies: dict[str, list[str]]) -> None:
        self._model = model
        self.replies = replies
        self.prompts: list[list[Message]] = []

    @property
    def model(self) -> str:
        return self._model

    def complete(self, messages: list[Message], params: CompletionParams) -> LLMResponse:
        self.prompts.append(messages)
        system = messages[0]["content"]
        for key, queue in self.replies.items():
            if key in system:
                text = queue.pop(0) if len(queue) > 1 else queue[0]
                return LLMResponse(text=text, model=self._model, prompt_tokens=100)
        raise AssertionError(f"Prompt inattendu : {system[:60]}")


def chunk(page_index: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"c{page_index}",
        doc_name="3M_2018_10K",
        company="3M",
        ticker="MMM",
        doc_type="10-K",
        fiscal_year=2018,
        fiscal_quarter=None,
        page_index=page_index,
        chunk_index=0,
        header=f"3M | 10-K | FY2018 | page {page_index + 1}",
        text=text,
    )


class StaticRetriever:
    def __init__(self) -> None:
        self.filters: list[SearchFilters | None] = []

    def search(
        self, query: str, k: int, filters: SearchFilters | None = None
    ) -> list[RetrievedChunk]:
        self.filters.append(filters)
        items = [
            chunk(59, "Purchases of property, plant and equipment (PP&E) | (1,577) | (1,373)"),
            chunk(10, "Risk factors: litigation"),
            chunk(20, "Segment information"),
        ]
        return [RetrievedChunk(c, 1.0, i) for i, c in enumerate(items[:k])]


def fact_store() -> FactStore:
    payload: dict[str, Any] = {
        "cik": 66740,
        "facts": {
            "us-gaap": {
                "PaymentsToAcquirePropertyPlantAndEquipment": {
                    "label": "Payments to Acquire Property, Plant, and Equipment",
                    "units": {
                        "USD": [
                            {
                                "start": "2018-01-01",
                                "end": "2018-12-31",
                                "val": 1577000000,
                                "accn": "a1",
                                "fy": 2018,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2019-02-07",
                            }
                        ]
                    },
                }
            }
        },
    }
    con = duckdb.connect(":memory:")
    init_schema(con)
    facts, concepts = parse_company_facts(payload, "MMM")
    upsert_company(con, 66740, facts, concepts)
    build_derived_tables(con)
    return FactStore(con)


PLAN = json.dumps(
    {
        "ticker": "MMM",
        "fiscal_year": 2018,
        "answer_type": "numeric",
        "metrics": [{"label": "capital expenditure", "fiscal_year": 2018}],
        "search_queries": ["capital expenditures purchases of property plant equipment"],
    }
)
GOOD_SYNTHESIS = json.dumps(
    {
        "sufficient": True,
        "calculations": [
            {
                "name": "capex_musd",
                "expression": "capex / 1000000",
                "inputs": {"capex": {"evidence": "E1", "value": 1577000000}},
            }
        ],
        "answer": "3M spent ${capex_musd} million on PP&E in FY2018 [E1][E2].",
        "final_answer": "${capex_musd} million",
        "cited": ["E1", "E2"],
    }
)
# Le modèle invente un chiffre absent des preuves : la vérification doit le refuser
INVENTED = json.dumps(
    {"sufficient": True, "answer": "Capex was $1,999 million.", "final_answer": "$1,999 million"}
)


def make_agent(
    synthesis_replies: list[str], config: AgentConfig | None = None
) -> tuple[FinancialAgent, ScriptedLLM, ScriptedLLM, StaticRetriever]:
    planner = ScriptedLLM(
        "small",
        {
            "You plan": [PLAN],
            "You check whether": ['{"sufficient": true, "missing": "", "new_queries": []}'],
        },
    )
    writer = ScriptedLLM("big", {"You are a financial analyst": synthesis_replies})
    retriever = StaticRetriever()
    runtime = AgentRuntime(
        planner_llm=planner,
        writer_llm=writer,
        search=SearchFilingsTool(retriever),
        facts=fact_store(),
        config=config or AgentConfig(passages_per_query=2),
        planner_params=CompletionParams(),
        writer_params=CompletionParams(),
    )
    return FinancialAgent(runtime), planner, writer, retriever


def test_full_run_traces_every_number_to_a_source() -> None:
    agent, _, writer, retriever = make_agent([GOOD_SYNTHESIS])

    answer = agent.run(QUESTION)

    assert "Final answer: $1,577 million" in answer.text
    assert not answer.abstained
    # Le fait XBRL arrive en E1, les passages ensuite, filtrés sur l'entreprise et l'exercice
    assert retriever.filters[0] == SearchFilters(ticker="MMM", fiscal_year=2018)
    assert answer.cited == [("3M_2018_10K", 59)]
    assert [t["node"] for t in answer.trace] == [
        "plan",
        "gather",
        "grade",
        "synthesize",
        "verify",
        "total",
    ]
    assert "E1" in writer.prompts[0][1]["content"]
    assert len(answer.llm_calls) == 3


def test_invented_number_triggers_a_retry_then_succeeds() -> None:
    agent, _, writer, _ = make_agent([INVENTED, GOOD_SYNTHESIS])

    answer = agent.run(QUESTION)

    assert "$1,577 million" in answer.text
    # Le second essai reçoit la raison du refus
    assert "untraceable number $1,999 million" in writer.prompts[1][1]["content"]


def test_persistent_invention_ends_in_abstention() -> None:
    agent, _, _, _ = make_agent([INVENTED])

    answer = agent.run(QUESTION)

    assert answer.abstained
    assert answer.text.startswith("INSUFFICIENT_CONTEXT")


def test_without_verifier_the_invented_number_goes_through() -> None:
    agent, _, _, _ = make_agent([INVENTED], AgentConfig(use_verifier=False))

    answer = agent.run(QUESTION)

    assert "$1,999 million" in answer.text
    assert [t["node"] for t in answer.trace][-2] == "synthesize"


def test_without_xbrl_no_fact_is_looked_up() -> None:
    agent, _, writer, _ = make_agent([INVENTED], AgentConfig(use_xbrl=False, use_verifier=False))

    agent.run(QUESTION)

    assert "XBRL" not in writer.prompts[0][1]["content"]


# Briques isolées


def test_calculation_inputs_must_come_from_the_cited_evidence() -> None:
    evidence = [Evidence("E1", "xbrl", "xbrl:x", "t", "1,577,000,000 USD", value=1577000000)]
    synthesis = Synthesis.model_validate(
        {
            "calculations": [
                {
                    "name": "ok",
                    "expression": "a / 1000000",
                    "inputs": {"a": {"evidence": "E1", "value": 1577000000}},
                },
                {
                    "name": "bad",
                    "expression": "b * 2",
                    "inputs": {"b": {"evidence": "E1", "value": 999}},
                },
            ]
        }
    )

    results, problems = run_calculations(synthesis, evidence)

    assert results["ok"].value == 1577
    assert problems == ["bad: 999.0 does not appear in E1"]


def test_placeholders_are_filled_and_unknown_ones_reported() -> None:
    results = {"growth": calculate("(a - b) / b * 100", {"a": 1577, "b": 1373})}

    text, problems = fill_placeholders("Growth was {growth}% and {missing}.", results)

    assert text == "Growth was 14.86% and {missing}."
    assert problems == ["unknown calculation {missing}"]


@pytest.mark.parametrize(
    ("text", "unverified"),
    [
        ("Capex was $1,577 million in FY2018 [E1].", []),
        ("Capex was $1.577 billion.", []),
        ("Growth was 14.86% over 2 years.", []),
        ("Capex was $1,999 million.", ["$1,999 million"]),
    ],
)
def test_verify_numbers(text: str, unverified: list[str]) -> None:
    evidence = [Evidence("E1", "xbrl", "xbrl:x", "t", "", value=1577000000)]
    results = [calculate("(a - b) / b * 100", {"a": 1577, "b": 1373})]

    assert verify_numbers(text, evidence, results).unverified == unverified


def test_json_call_retries_once_on_invalid_output() -> None:
    llm = ScriptedLLM("m", {"S": ["not json", '{"ticker": "MMM"}']})
    messages: list[Message] = [{"role": "system", "content": "S"}, {"role": "user", "content": "q"}]

    plan, calls = json_call(llm, messages, CompletionParams(), Plan)

    assert plan is not None and plan.ticker == "MMM"
    assert len(calls) == 2


def test_invalid_planner_output_falls_back_to_rules() -> None:
    agent, planner, _, retriever = make_agent([GOOD_SYNTHESIS])
    planner.replies["You plan"] = ["garbage"]

    agent.run(QUESTION)

    # Le ticker et l'exercice viennent alors du repérage par règles
    assert retriever.filters[0] == SearchFilters(ticker="MMM", fiscal_year=2018)


def test_a_calculation_can_reuse_an_earlier_result() -> None:
    # Cas réel : rotation des stocks = coût des ventes / stock moyen, le stock moyen étant calculé
    evidence = [
        Evidence("E1", "xbrl", "xbrl:cogs", "t", "", value=16830),
        Evidence("E2", "xbrl", "xbrl:inv19", "t", "", value=2721),
        Evidence("E3", "xbrl", "xbrl:inv18", "t", "", value=2667),
    ]
    synthesis = Synthesis.model_validate(
        {
            "calculations": [
                {
                    "name": "avg_inventory",
                    "expression": "(a + b) / 2",
                    "inputs": {
                        "a": {"evidence": "E2", "value": 2721},
                        "b": {"evidence": "E3", "value": 2667},
                    },
                },
                {
                    "name": "turnover",
                    "expression": "cogs / avg_inventory",
                    "inputs": {"cogs": {"evidence": "E1", "value": 16830}},
                },
            ]
        }
    )

    results, problems = run_calculations(synthesis, evidence)

    assert problems == []
    assert round(results["turnover"].value, 2) == 6.25
