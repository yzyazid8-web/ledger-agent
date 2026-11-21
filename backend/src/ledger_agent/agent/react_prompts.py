"""Prompt de l'agent ReAct (en anglais, langue des documents et des modèles)."""

from functools import cache

from ledger_agent.ingestion.universe import load_universe


@cache
def react_system_prompt(max_steps: int) -> str:
    # Prompt volontairement court : il est renvoyé à chaque tour, avec l'historique
    companies = ", ".join(_short(c.ticker, c.name) for c in load_universe())
    return f"""You are a financial analyst agent answering questions about US public companies
from their SEC filings, using tools. Companies: {companies}.
- Reported line items: find_concepts then get_fact (raw units). Anything else: search_filings
  (tables are often in millions).
- Never do arithmetic yourself: use calculate.
- If a tool fails or returns nothing useful, try another query, concept, year or period.
- Finish with submit_answer, citing evidence after each fact (e.g. "$1,577 million [E1]"), or
  abstain. Every number you write must come from the evidence or a calculation, with the right
  sign. A plain-text answer must end with "Final answer: <answer>".
At most {max_steps} tool turns."""


def _short(ticker: str, name: str) -> str:
    # « AMD (AMD) » : le nom n'apporte rien quand il répète le ticker
    return ticker if name.upper() == ticker else f"{ticker} {name}"


BUDGET_EXHAUSTED = (
    "Tool budget exhausted. Answer now with submit_answer using the evidence you have, "
    "or call abstain if it is not sufficient."
)
NO_TOOL_CALLED = (
    "Use the tools. To finish, call submit_answer or abstain, or write your answer ending "
    'with "Final answer: ...".'
)


def verification_failed(problems: list[str]) -> str:
    return (
        "Your answer was rejected by the number checker: "
        + "; ".join(problems)
        + ". Fix it: search or calculate what is missing, then call submit_answer again. "
        "Every number must come from the evidence or a calculation result."
    )


def invalid_tool_call(reason: str) -> str:
    return (
        f"Your last tool call was rejected as malformed: {reason}. "
        "Call the tool again with exactly the documented arguments."
    )
