"""Agent ReAct avec un faux LLM qui joue un scénario d'appels d'outils."""

import json
from typing import Any

from test_agent import QUESTION, StaticRetriever, fact_store

from ledger_agent.agent.evidence import Evidence
from ledger_agent.agent.react import ReactAgent, ReactConfig
from ledger_agent.agent.toolbox import Toolbox
from ledger_agent.agent.verify import verify_numbers
from ledger_agent.llm.types import (
    CompletionParams,
    LLMResponse,
    LLMToolCallError,
    Message,
    ToolCall,
)
from ledger_agent.tools.calculator import calculate
from ledger_agent.tools.search_filings import SearchFilingsTool


def call(name: str, **arguments: Any) -> ToolCall:
    return {
        "id": f"call-{name}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class ToolScriptLLM:
    """Faux LLM : chaque appel renvoie l'étape suivante du scénario."""

    def __init__(self, script: list[Any]) -> None:
        self.script = script
        self.seen: list[list[Message]] = []
        self.tools_offered: list[list[str]] = []
        self.choices: list[str] = []

    @property
    def model(self) -> str:
        return "fake"

    def complete(self, messages: list[Message], params: CompletionParams) -> LLMResponse:
        self.seen.append(list(messages))
        self.tools_offered.append([t["function"]["name"] for t in params.tools])
        self.choices.append(params.tool_choice)
        step = self.script.pop(0) if self.script else "rien"
        if isinstance(step, Exception):
            raise step
        if isinstance(step, str):
            return LLMResponse(text=step, model="fake", prompt_tokens=50)
        return LLMResponse(text="", model="fake", prompt_tokens=50, tool_calls=step)


CAPEX = "PaymentsToAcquirePropertyPlantAndEquipment"
GOOD_RUN: list[list[ToolCall] | str] = [
    [call("find_concepts", label="capital expenditure", ticker="MMM")],
    [call("get_fact", ticker="MMM", concept=CAPEX, fiscal_year=2018)],
    [
        call(
            "calculate",
            expression="capex / 1000000",
            inputs={"capex": {"source": "E1", "value": 1577000000}},
        )
    ],
    [
        call(
            "submit_answer",
            answer="3M spent $1,577 million on PP&E in FY2018 [E1].",
            final_answer="$1,577 million",
            cited=["E1"],
        )
    ],
]


def make_agent(
    script: list[Any], config: ReactConfig | None = None
) -> tuple[ReactAgent, ToolScriptLLM]:
    llm = ToolScriptLLM(script)
    agent = ReactAgent(
        llm=llm,
        search=SearchFilingsTool(StaticRetriever()),
        facts=fact_store(),
        config=config or ReactConfig(max_steps=6),
        params=CompletionParams(),
    )
    return agent, llm


def test_the_model_chooses_tools_and_reads_their_results() -> None:
    agent, llm = make_agent(list(GOOD_RUN))

    answer = agent.run(QUESTION)

    assert "Final answer: $1,577 million" in answer.text
    assert not answer.abstained
    # Chaque décision est prise après avoir lu le résultat de l'outil précédent
    last_observation = llm.seen[2][-1]
    assert last_observation["role"] == "tool"
    assert json.loads(last_observation["content"] or "{}")["id"] == "E1"
    assert [t["node"] for t in answer.trace][:2] == ["agent", "tool"]
    assert "verify" in [t["node"] for t in answer.trace]


def test_tool_errors_are_observations_the_model_can_correct() -> None:
    script = [
        [call("get_fact", ticker="MMM", concept="Revenues", fiscal_year=2018)],
        *GOOD_RUN,
    ]
    agent, llm = make_agent(script)

    answer = agent.run(QUESTION)

    first = json.loads(llm.seen[1][-1]["content"] or "{}")
    assert "error" in first  # 3M ne déclare pas « Revenues » dans ce faux jeu de données
    assert "$1,577 million" in answer.text


def test_rejected_answer_sends_the_agent_back_to_work() -> None:
    invented = [
        call(
            "submit_answer",
            answer="Capex was $1,999 million.",
            final_answer="$1,999 million",
            cited=[],
        )
    ]
    agent, llm = make_agent([invented, *GOOD_RUN])

    answer = agent.run(QUESTION)

    feedback = next(
        m for m in llm.seen[1] if m["role"] == "user" and "rejected" in (m["content"] or "")
    )
    assert "$1,999 million" in (feedback["content"] or "")
    assert "$1,577 million" in answer.text


def test_step_budget_ends_in_abstention() -> None:
    looping = [[call("search_filings", query="capex")] for _ in range(10)]
    agent, llm = make_agent(looping, ReactConfig(max_steps=2))

    answer = agent.run(QUESTION)

    assert answer.abstained
    # Au dernier tour, seuls les outils de fin sont proposés
    assert llm.tools_offered[-1] == ["submit_answer", "abstain"]


def test_cited_evidence_missing_from_the_text_is_appended() -> None:
    script = [
        *GOOD_RUN[:2],
        [
            call(
                "submit_answer",
                answer="Capex was $1,577 million.",
                final_answer="$1,577 million",
                cited=["E1"],
            )
        ],
    ]
    agent, _ = make_agent(script)

    answer = agent.run(QUESTION)

    assert "Capex was $1,577 million. [E1]" in answer.text


def test_abstain_tool_ends_the_run() -> None:
    agent, _ = make_agent([[call("abstain", reason="not in the filings")]])

    answer = agent.run(QUESTION)

    assert answer.abstained
    assert "not in the filings" in answer.text


def test_without_xbrl_the_fact_tools_are_not_offered() -> None:
    agent, llm = make_agent([[call("abstain", reason="x")]], ReactConfig(use_xbrl=False))

    agent.run(QUESTION)

    assert llm.tools_offered[0] == ["search_filings", "calculate", "submit_answer", "abstain"]


def test_stream_emits_steps_then_an_answer_with_calculations() -> None:
    agent, _ = make_agent(list(GOOD_RUN))

    events = list(agent.stream(QUESTION))

    assert events[-1]["type"] == "answer"
    assert events[-1]["calculations"][0]["name"] == "C1"
    assert events[-1]["cited"] == ["E1"]
    assert {e["node"] for e in events[:-1]} >= {"agent", "tool", "verify"}


# Boîte à outils


def test_calculation_inputs_must_match_their_source() -> None:
    box = Toolbox(SearchFilingsTool(StaticRetriever()), fact_store())
    box.execute("get_fact", json.dumps({"ticker": "MMM", "concept": CAPEX, "fiscal_year": 2018}))

    wrong = box.execute(
        "calculate",
        json.dumps({"expression": "a * 2", "inputs": {"a": {"source": "E1", "value": 999}}}),
    )
    chained = box.execute(
        "calculate",
        json.dumps(
            {"expression": "a / 1000000", "inputs": {"a": {"source": "E1", "value": 1577000000}}}
        ),
    )
    reused = box.execute(
        "calculate",
        json.dumps({"expression": "m * 2", "inputs": {"m": {"source": "C1", "value": 1577}}}),
    )

    assert "n'apparaît pas" in wrong["error"]
    assert chained == {"id": "C1", "value": 1577.0}
    assert reused["value"] == 3154.0


def test_unknown_or_internal_tool_names_are_refused() -> None:
    box = Toolbox(SearchFilingsTool(StaticRetriever()), None)

    assert "inconnu" in box.execute("register", "{}")["error"]
    assert "inconnu" in box.execute("get_fact", "{}")["error"]  # XBRL désactivé


def test_repeated_search_reuses_evidence_ids() -> None:
    box = Toolbox(SearchFilingsTool(StaticRetriever()), None, passages_per_search=2)

    first = box.execute("search_filings", json.dumps({"query": "capex"}))
    second = box.execute("search_filings", json.dumps({"query": "capex again"}))

    assert [p["id"] for p in first["passages"]] == [p["id"] for p in second["passages"]]
    assert len(box.evidence) == 2


# Contrôle du signe


def test_a_negative_result_presented_as_an_increase_is_rejected() -> None:
    results = [calculate("(a - b) / b * 100", {"a": 1373, "b": 1577})]  # ≈ -12.94
    evidence: list[Evidence] = []

    assert verify_numbers("Capex rose by 12.94% [E1].", evidence, results).unverified
    assert not verify_numbers("Capex declined by 12.94%.", evidence, results).unverified
    assert not verify_numbers("Capex changed by -12.94%.", evidence, results).unverified


def test_a_malformed_tool_call_is_reported_to_the_model_instead_of_crashing() -> None:
    rejected = LLMToolCallError("missing properties: 'answer'")
    agent, llm = make_agent([rejected, *GOOD_RUN])

    answer = agent.run(QUESTION)

    assert "$1,577 million" in answer.text
    assert "malformed" in (llm.seen[1][-1]["content"] or "")


def test_citing_a_calculation_and_its_formula_constants_is_accepted() -> None:
    growth = [
        call("get_fact", ticker="MMM", concept=CAPEX, fiscal_year=2018),
        call(
            "calculate",
            expression="a / b * 100",
            inputs={
                "a": {"source": "E1", "value": 1577000000},
                "b": {"source": "E1", "value": 1577000000},
            },
        ),
    ]
    submit = call(
        "submit_answer",
        answer="Capex [E1] divided by itself times 100 gives 100 [C1].",
        final_answer="100%",
        cited=["E1", "C1"],
    )
    agent, _ = make_agent([[growth[0]], [growth[1]], [submit]])

    answer = agent.run(QUESTION)

    assert not answer.abstained
    assert "verify" in [t["node"] for t in answer.trace]


def test_text_without_a_final_answer_line_forces_a_tool_call_next() -> None:
    # Sans « Final answer: », le texte peut n'être qu'une étape de raisonnement
    agent, llm = make_agent(["Now I will fetch the 2018 value.", *GOOD_RUN])

    answer = agent.run(QUESTION)

    assert llm.choices[:2] == ["auto", "required"]
    assert "$1,577 million" in answer.text


def test_the_last_turn_requires_submitting_or_abstaining() -> None:
    looping = [[call("search_filings", query="capex")] for _ in range(10)]
    agent, llm = make_agent(looping, ReactConfig(max_steps=2))

    agent.run(QUESTION)

    assert llm.choices[-1] == "required"


def test_numbers_written_inside_a_formula_are_refused_but_constants_allowed() -> None:
    box = Toolbox(SearchFilingsTool(StaticRetriever()), fact_store())
    box.execute("get_fact", json.dumps({"ticker": "MMM", "concept": CAPEX, "fiscal_year": 2018}))

    smuggled = box.execute("calculate", json.dumps({"expression": "6043 - 3694", "inputs": {}}))
    named = box.execute(
        "calculate",
        json.dumps(
            {
                "expression": "capex_2018 / 1000000 * 100",
                "inputs": {"capex_2018": {"source": "E1", "value": 1577000000}},
            }
        ),
    )

    assert "inputs" in smuggled["error"]
    assert named["value"] == 157700.0


def test_a_text_answer_with_a_final_line_is_accepted_and_verified() -> None:
    text = "Capex was $1,577 million [E1].\n\nFinal answer: $1,577 million"
    agent, llm = make_agent([*GOOD_RUN[:2], text])

    answer = agent.run(QUESTION)

    assert not answer.abstained
    assert "Final answer: $1,577 million" in answer.text
    assert len(llm.seen) == 3  # aucun tour supplémentaire pour resoumettre
    assert "verify" in [t["node"] for t in answer.trace]


def test_an_invented_figure_in_a_text_answer_is_still_rejected() -> None:
    text = "Capex was $1,999 million.\n\nFinal answer: $1,999 million"
    agent, llm = make_agent([*GOOD_RUN[:2], text, *GOOD_RUN[2:]])

    answer = agent.run(QUESTION)

    assert any("rejected" in (m["content"] or "") for m in llm.seen[3] if m["role"] == "user")
    assert "$1,577 million" in answer.text


def test_a_text_refusal_with_a_final_line_is_an_abstention() -> None:
    agent, _ = make_agent(["The filings do not say.\n\nFinal answer: INSUFFICIENT_CONTEXT"])

    answer = agent.run(QUESTION)

    assert answer.abstained


def test_uncited_numbers_get_the_citations_of_the_evidence_they_come_from() -> None:
    uncited = call(
        "submit_answer",
        answer="Capex was $1,577 million.",
        final_answer="$1,577 million",
        cited=[],
    )
    agent, _ = make_agent([*GOOD_RUN[:2], [uncited]])

    events = list(agent.stream(QUESTION))

    assert events[-1]["cited"] == ["E1"]
    assert "Capex was $1,577 million. [E1]" in events[-1]["text"]


def test_escaped_newlines_written_by_the_model_are_restored() -> None:
    escaped = call(
        "submit_answer",
        answer=r"Capex was $1,577 million [E1].\n\n",
        final_answer=r"$1,577 million\n",
        cited=["E1"],
    )
    agent, _ = make_agent([*GOOD_RUN[:2], [escaped]])

    answer = agent.run(QUESTION)

    assert r"\n" not in answer.text  # plus de « \n » écrit en toutes lettres
    assert answer.text.endswith("Final answer: $1,577 million")
