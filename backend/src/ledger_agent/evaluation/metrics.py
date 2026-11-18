"""Métriques calculées par règles, sans LLM : exactitude numérique et recherche."""

import re
from collections.abc import Iterable
from dataclasses import dataclass

from ledger_agent.answer_format import extract_final_answer

_SCALES = {
    "thousand": 1e3,
    "k": 1e3,
    "million": 1e6,
    "mn": 1e6,
    "mm": 1e6,
    "m": 1e6,
    "billion": 1e9,
    "bn": 1e9,
    "b": 1e9,
    "trillion": 1e12,
}

# Pas de lettre ni de chiffre juste avant : « FY2018 » ou « Q4 » ne sont pas des montants
_NUMBER = re.compile(
    r"(?<![A-Za-z0-9.])(?P<neg>[-(])?\s*(?P<dollar>\$)?\s*"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    r"\s*(?P<pct>%|percent\b)?"
    r"\s*(?P<scale>thousand|million|billion|trillion|mn|mm|bn|k|m|b)?\b",
    re.IGNORECASE,
)

_NUMERIC_ANSWER = re.compile(
    r"[-(]?\s*\$?\s*(\d{1,3}(,\d{3})+|\d+)?(\.\d+)?\s*"
    r"(%|percent|thousand|million|billion|trillion)?\s*\)?\.?",
    re.IGNORECASE,
)

# Mots qui expriment un écart négatif sans signe moins : « a decrease of 0.02 »
_NEGATIVE_WORDS = re.compile(
    r"\b(decrease|decreased|decline|declined|drop|dropped|fell|fall|lower|loss|negative|"
    r"down|reduction|reduced|shrunk|shrank|contraction)\b",
    re.IGNORECASE,
)
NEGATIVE_WINDOW = 40

DEFAULT_REL_TOL = 0.01


@dataclass(frozen=True, slots=True)
class NumberMention:
    raw: float
    decimals: int
    is_percent: bool
    scale: float
    negative: bool = False
    monetary: bool = False

    def variants(self, allow_unscaled: bool) -> set[float]:
        """Valeurs plausibles d'un même nombre selon l'unité sous-entendue."""
        absolute = self.raw * self.scale
        values = {absolute}
        if self.scale == 1.0 or allow_unscaled:
            values.add(self.raw)
        if self.scale != 1.0:
            # FinanceBench exprime souvent les montants en millions sans le dire ($1577.00)
            values |= {absolute / 1e6, absolute / 1e3}
        if self.is_percent:
            values.add(self.raw / 100)
        return values


def parse_numbers(text: str) -> list[NumberMention]:
    mentions = []
    for match in _NUMBER.finditer(text):
        # « 3M » est le nom de l'entreprise, pas 3 millions
        if re.fullmatch(r"3M", match.group(0).strip()):
            continue
        digits = match.group("num").replace(",", "")
        scale_word = (match.group("scale") or "").lower()
        before = text[max(0, match.start() - NEGATIVE_WINDOW) : match.start()]
        mentions.append(
            NumberMention(
                raw=abs(float(digits)),
                decimals=len(digits.split(".")[1]) if "." in digits else 0,
                is_percent=match.group("pct") is not None,
                scale=_SCALES.get(scale_word, 1.0),
                negative=match.group("neg") is not None or bool(_NEGATIVE_WORDS.search(before)),
                monetary=match.group("dollar") is not None,
            )
        )
    return mentions


def is_numeric_answer(answer: str) -> bool:
    stripped = answer.strip()
    return bool(re.search(r"\d", stripped)) and _NUMERIC_ANSWER.fullmatch(stripped) is not None


def numbers_match(gold: NumberMention, predicted: NumberMention, rel_tol: float) -> bool:
    # Une référence négative exige une prédiction négative ; une référence positive
    # accepte les deux, car « baisse de 1.9 % » répond juste à « de combien a-t-il baissé »
    if gold.negative and not predicted.negative:
        return False
    # Un montant mis à l'échelle (« 1.5 billion ») ne vaut 1.5 que face à un montant en dollars
    allow_unscaled = gold.monetary
    rounding_tol = 0.5 * 10 ** (-gold.decimals)
    for g in gold.variants(allow_unscaled=True):
        factor = abs(g / gold.raw) if gold.raw else 1.0
        tolerance = max(rel_tol * abs(g), rounding_tol * factor) + 1e-12
        if any(abs(p - g) <= tolerance for p in predicted.variants(allow_unscaled)):
            return True
    return False


def numeric_correct(gold_answer: str, prediction: str, rel_tol: float = DEFAULT_REL_TOL) -> bool:
    gold_numbers = parse_numbers(gold_answer)
    if not gold_numbers:
        return False
    gold = gold_numbers[0]
    # Seule la réponse finale compte : un chiffre intermédiaire juste ne suffit pas
    candidates = parse_numbers(extract_final_answer(prediction))
    return any(numbers_match(gold, p, rel_tol) for p in candidates)


def page_hit(pages: Iterable[tuple[str, int]], gold: Iterable[tuple[str, int]]) -> bool:
    return not set(pages).isdisjoint(set(gold))


def citation_precision(
    cited: list[tuple[str, int]], gold: Iterable[tuple[str, int]]
) -> float | None:
    """Part des passages cités qui sont des pages justificatives, ou None si rien n'est cité."""
    if not cited:
        return None
    gold_set = set(gold)
    return sum(page in gold_set for page in cited) / len(cited)
