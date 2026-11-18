"""Vérification par règles : chaque nombre de la réponse doit avoir une source."""

import re
from collections.abc import Iterable
from dataclasses import dataclass

from ledger_agent.agent.evidence import Evidence
from ledger_agent.numbers import NumberMention, parse_numbers
from ledger_agent.tools.calculator import CalculationResult

REL_TOL = 0.005
# Petits entiers sans unité (« 3 segments », « 2 ans ») : trop ambigus pour être vérifiés
MAX_UNCHECKED_INTEGER = 12
_SCALE_STEPS = (1.0, 1e3, 1e6, 1e9)


@dataclass(frozen=True, slots=True)
class VerificationResult:
    unverified: list[str]

    @property
    def ok(self) -> bool:
        return not self.unverified


def _source_values(
    evidence: Iterable[Evidence], results: Iterable[CalculationResult]
) -> list[float]:
    values: list[float] = []
    for item in evidence:
        if item.value is not None:
            # Un fait XBRL en dollars peut être cité en milliers, millions ou milliards
            values += [abs(item.value) / step for step in _SCALE_STEPS]
        for mention in parse_numbers(item.text):
            values += list(mention.variants(allow_unscaled=True))
    for result in results:
        values += [abs(result.value), abs(result.value) / 100]
    return values


def _skippable(mention: NumberMention) -> bool:
    if mention.looks_like_year:
        return True
    plain = mention.scale == 1.0 and not (mention.is_percent or mention.monetary)
    return plain and mention.decimals == 0 and mention.raw <= MAX_UNCHECKED_INTEGER


def _traced(mention: NumberMention, sources: list[float]) -> bool:
    rounding = 0.5 * 10 ** (-mention.decimals)
    for candidate in mention.variants(allow_unscaled=True):
        factor = candidate / mention.raw if mention.raw else 1.0
        tolerance = max(REL_TOL * abs(candidate), rounding * abs(factor)) + 1e-9
        if any(abs(candidate - s) <= tolerance for s in sources):
            return True
    return False


def verify_numbers(
    text: str, evidence: list[Evidence], results: list[CalculationResult]
) -> VerificationResult:
    """Liste les nombres de `text` qui ne viennent ni d'une preuve, ni d'un calcul."""
    # Les références [E3] ne sont pas des nombres à vérifier
    cleaned = re.sub(r"\[E\d+\]", " ", text)
    sources = _source_values(evidence, results)
    unverified = [
        m.text for m in parse_numbers(cleaned) if not _skippable(m) and not _traced(m, sources)
    ]
    return VerificationResult(unverified=list(dict.fromkeys(unverified)))


def input_matches_evidence(value: float, evidence: Evidence) -> bool:
    """Vrai si la valeur passée à un calcul figure bien dans la preuve citée."""
    sources = _source_values([evidence], [])
    return any(abs(abs(value) - s) <= max(REL_TOL * abs(value), 1e-9) for s in sources)
