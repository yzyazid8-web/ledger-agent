"""Vérification par règles : chaque nombre de la réponse doit avoir une source."""

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from ledger_agent.agent.evidence import Evidence
from ledger_agent.numbers import NumberMention, parse_numbers
from ledger_agent.tools.calculator import CalculationResult

REL_TOL = 0.005
# Petits entiers sans unité (« 3 segments », « 2 ans ») : trop ambigus pour être vérifiés
MAX_UNCHECKED_INTEGER = 12
_SCALE_STEPS = (1.0, 1e3, 1e6, 1e9)


@dataclass(frozen=True, slots=True)
class VerificationResult:
    # Nombres sans source, et nombres qui présentent un résultat négatif comme positif
    unverified: list[str]
    sign_conflicts: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.unverified or self.sign_conflicts)

    def problems(self) -> list[str]:
        """Messages adressés au modèle : chacun dit précisément quoi corriger."""
        return [f"untraceable number {n}" for n in self.unverified] + [
            f"{n} restates a negative calculation result as positive: write it with a minus "
            "sign (-) or a word such as 'decrease' or 'fewer'"
            for n in self.sign_conflicts
        ]


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
        # Un ratio peut être écrit en pourcentage, avec ou sans « % » (0.5403 → 54.03)
        values += [abs(result.value), abs(result.value) / 100, abs(result.value) * 100]
        # Les constantes d'une formule (le « 100 » d'un pourcentage) peuvent être recopiées
        values += formula_literals(result.expression)
    return values


# Seules constantes admises en clair dans une formule ; tout autre nombre est une entrée sourcée
FORMULA_CONSTANTS = frozenset({*range(13), 100, 360, 365, 1000, 1e6, 1e9})
# Nombre isolé, pas un morceau de nom de variable comme capex_2018
_LITERAL = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?![\w.])")


def formula_literals(expression: str) -> list[float]:
    return [float(n) for n in _LITERAL.findall(expression)]


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


def _sign_conflict(mention: NumberMention, results: list[CalculationResult]) -> bool:
    """Vrai si le nombre reprend un résultat négatif en le présentant comme positif."""
    matching = [r for r in results if _traced(mention, [abs(r.value), abs(r.value) / 100])]
    # « une hausse de 4 % » pour un résultat de -4 : le texte inverse le sens du calcul.
    # L'inverse (« une baisse de 4 % » pour +4) reste accepté : c'est souvent une amplitude
    return bool(matching) and all(r.value < 0 for r in matching) and not mention.negative


def verify_numbers(
    text: str, evidence: list[Evidence], results: list[CalculationResult]
) -> VerificationResult:
    """Liste les nombres de `text` sans source, ou dont le signe contredit le calcul."""
    # Les références aux preuves et aux calculs ([E3], E3, C1) ne sont pas des nombres
    cleaned = re.sub(r"\[?\b[EC]\d+\b\]?", " ", text)
    sources = _source_values(evidence, results)
    unverified: list[str] = []
    conflicts: list[str] = []
    for mention in parse_numbers(cleaned):
        if _skippable(mention):
            continue
        if not _traced(mention, sources):
            unverified.append(mention.text)
        elif _sign_conflict(mention, results):
            conflicts.append(mention.text)
    return VerificationResult(
        unverified=list(dict.fromkeys(unverified)), sign_conflicts=list(dict.fromkeys(conflicts))
    )


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(REL_TOL * abs(a), 1e-9)


def input_matches_evidence(value: float, evidence: Evidence) -> bool:
    """Vrai si la valeur passée à un calcul figure bien dans la preuve citée."""
    if any(_close(abs(value), s) for s in _source_values([evidence], [])):
        return True
    # Un tableau en millions (« 5,207 ») peut être passé en valeur brute (5 207 000 000).
    # La tolérance d'échelle ne vaut que pour de vrais montants : une année, un jour de date
    # ou un petit entier mis à l'échelle laisserait passer n'importe quelle valeur
    amounts = [
        v
        for m in parse_numbers(evidence.text)
        if not _skippable(m)
        for v in m.variants(allow_unscaled=True)
    ]
    scaled = [abs(value) / step for step in _SCALE_STEPS[1:]]
    return any(_close(c, s) for c in scaled for s in amounts)


def evidence_for_numbers(text: str, evidence: list[Evidence]) -> list[str]:
    """Preuves d'où viennent les nombres de `text`, selon la correspondance du vérificateur."""
    cleaned = re.sub(r"\[?\b[EC]\d+\b\]?", " ", text)
    mentions = [m for m in parse_numbers(cleaned) if not _skippable(m)]
    found: list[str] = []
    for item in evidence:
        sources = _source_values([item], [])
        if item.id not in found and any(_traced(m, sources) for m in mentions):
            found.append(item.id)
    return found
