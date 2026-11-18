"""Preuves collectées par l'agent : faits XBRL et passages, chacun avec sa source."""

from dataclasses import dataclass
from typing import Literal

from ledger_agent.tools.financial_facts import FactResult
from ledger_agent.tools.search_filings import Passage


@dataclass(frozen=True, slots=True)
class Evidence:
    id: str
    kind: Literal["xbrl", "passage"]
    source_id: str
    title: str
    text: str
    value: float | None = None
    doc_name: str | None = None
    page_index: int | None = None
    chunk_id: str | None = None
    score: float = 0.0


def _format_amount(value: float, unit: str) -> str:
    text = f"{value:,.0f}" if value == int(value) else f"{value:,.4f}"
    if unit == "USD" and abs(value) >= 1e6:
        text += f" USD (= {value / 1e6:,.1f} million)"
    else:
        text += f" {unit}"
    return text


def fact_evidence(evidence_id: str, fact: FactResult) -> Evidence:
    period = f"{fact.period_start} to {fact.period_end}" if fact.period_start else fact.period_end
    restated = (
        f" Later restated to {_format_amount(fact.latest_value, fact.unit)}."
        if fact.latest_value is not None
        else ""
    )
    return Evidence(
        id=evidence_id,
        kind="xbrl",
        source_id=fact.source_id,
        # Libellé officiel du concept : le LLM voit ce que le chiffre est vraiment
        title=(
            f"XBRL {fact.ticker} {fact.label or fact.concept} [{fact.concept}], "
            f"{fact.fiscal_year} {fact.fiscal_period}"
        ),
        text=(
            f"{_format_amount(fact.value, fact.unit)} for period {period}, "
            f"reported in {fact.form} filed {fact.filed}.{restated}"
        ),
        value=fact.value,
    )


def passage_evidence(evidence_id: str, passage: Passage) -> Evidence:
    return Evidence(
        id=evidence_id,
        kind="passage",
        source_id=passage.source_id,
        title=passage.header,
        text=passage.text,
        doc_name=passage.doc_name,
        page_index=passage.page_number - 1,
        chunk_id=passage.chunk_id,
        score=passage.score,
    )


def render_evidence(evidence: list[Evidence]) -> str:
    return "\n\n".join(f"[{e.id}] {e.title}\n{e.text}" for e in evidence) or "(no evidence)"
