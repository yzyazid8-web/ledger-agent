"""Prompts de l'agent (en anglais, langue des documents et des modèles)."""

import hashlib
from functools import cache
from importlib.resources import files

from ledger_agent.answer_format import ABSTAIN_MARKER
from ledger_agent.ingestion.universe import load_universe


@cache
def planner_prompt() -> str:
    companies = ", ".join(f"{c.ticker} ({c.name})" for c in load_universe())
    return f"""You plan how to answer a question about US public company filings.
Covered companies: {companies}.

Return JSON only, with these fields:
- "ticker": the company's ticker from the list above, or null if none is named.
- "fiscal_year": the fiscal year the question is mainly about (the latest one if several), or null.
- "answer_type": "numeric" if the expected answer is a number or ratio, else "text".
- "metrics": reported line items to look up in the official XBRL financial data, each as
  {{"label": "<plain English line item>", "fiscal_year": <int>, "fiscal_period": "FY"}}.
  For a ratio or growth rate, list every component and every year needed
  (e.g. a quick ratio needs cash, short-term investments, receivables, current liabilities).
  Use fiscal_period "Q1".."Q4" only for quarterly figures. Leave empty for qualitative questions.
- "search_queries": 1 to 3 short keyword queries to find the relevant passages in the
  filing text (e.g. "capital expenditures purchases of property plant equipment")."""


GRADER_PROMPT = """You check whether the evidence below is enough to answer the question.
Return JSON only: {"sufficient": true|false, "missing": "<what is missing>",
"new_queries": ["<up to 2 short keyword queries to find what is missing>"]}.
Say sufficient only if every figure or fact the answer needs is present in the evidence."""


SYNTHESIZER_PROMPT = f"""You are a financial analyst. Answer the question using ONLY the
numbered evidence (E1, E2, ...). Never use outside knowledge.

Strict rules on numbers:
- You never do arithmetic yourself. Every derived number (sum, difference, ratio, growth,
  margin, unit conversion) must come from a calculation you declare:
  {{"name": "<snake_case>", "expression": "<arithmetic on variable names>",
    "inputs": {{"<variable>": {{"evidence": "E<n>", "value": <number from that evidence>}}}}}}
  Write {{name}} in your text where the result must appear; it will be computed for you.
  Example: {{"name": "capex_musd", "expression": "capex / 1000000",
  "inputs": {{"capex": {{"evidence": "E1", "value": 1577000000}}}}}},
  then write "${{capex_musd}} million" in the text.
- XBRL facts are in raw units (USD, shares). Passage tables are often in millions.
- Any number you write directly must appear as such in the evidence.
- Cite the evidence you rely on as [E<n>] in the answer.

Return JSON only:
{{"sufficient": true|false, "calculations": [...], "answer": "<short reasoning with citations>",
  "final_answer": "<the direct answer, concise>", "cited": ["E1", ...]}}
If the evidence is not enough, return sufficient false and final_answer "{ABSTAIN_MARKER}"."""


def prompts_sha() -> str:
    """Empreinte de tous les prompts de l'agent, pour la reproductibilité des évaluations."""
    raw = planner_prompt() + GRADER_PROMPT + SYNTHESIZER_PROMPT
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


@cache
def agent_code_sha() -> str:
    """Empreinte du code de l'agent (graphes, outils, vérification, prompts ReAct)."""
    digest = hashlib.sha256()
    package = files("ledger_agent.agent")
    for name in sorted(p.name for p in package.iterdir() if p.name.endswith(".py")):
        # Fins de ligne normalisées : la même empreinte sous Windows et sous Linux
        digest.update(name.encode())
        digest.update((package / name).read_text(encoding="utf-8").replace("\r\n", "\n").encode())
    return digest.hexdigest()[:12]
