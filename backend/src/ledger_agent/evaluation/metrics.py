"""Métriques calculées par règles, sans LLM : exactitude numérique, abstention, recherche."""

import re
from collections.abc import Iterable
from dataclasses import dataclass

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

_NUMBER = re.compile(
    r"(?P<neg>[-(])?\s*\$?\s*"
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

ABSTAIN_PATTERNS = (
    "insufficient_context",
    "insufficient context",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "unable to determine",
    "not enough information",
    "insufficient information",
    "do not contain",
    "does not contain",
    "not provided in the",
)

DEFAULT_REL_TOL = 0.01


@dataclass(frozen=True, slots=True)
class NumberMention:
    raw: float
    decimals: int
    is_percent: bool
    scale: float

    def variants(self) -> set[float]:
        """Valeurs plausibles d'un même nombre selon l'unité sous-entendue.

        FinanceBench exprime souvent les montants en millions sans le dire ($1577.00) :
        « $1.577 billion » doit donc aussi valoir 1577 une fois exprimé en millions.
        """
        absolute = self.raw * self.scale
        values = {self.raw, absolute}
        if self.scale != 1.0:
            values |= {absolute / 1e6, absolute / 1e3}
        if self.is_percent:
            values.add(self.raw / 100)
        return values


def parse_numbers(text: str) -> list[NumberMention]:
    mentions = []
    for match in _NUMBER.finditer(text):
        digits = match.group("num").replace(",", "")
        scale_word = (match.group("scale") or "").lower()
        mentions.append(
            NumberMention(
                # Le signe est ignoré : « baisse de 4 % » et « -4 % » désignent le même écart
                raw=abs(float(digits)),
                decimals=len(digits.split(".")[1]) if "." in digits else 0,
                is_percent=match.group("pct") is not None,
                scale=_SCALES.get(scale_word, 1.0),
            )
        )
    return mentions


def is_numeric_answer(answer: str) -> bool:
    stripped = answer.strip()
    return bool(re.search(r"\d", stripped)) and _NUMERIC_ANSWER.fullmatch(stripped) is not None


def numbers_match(gold: NumberMention, predicted: NumberMention, rel_tol: float) -> bool:
    # Tolérance d'arrondi : « 14.9 » accepte tout ce qui s'arrondit à 14.9
    rounding_tol = 0.5 * 10 ** (-gold.decimals)
    for g in gold.variants():
        # La tolérance d'arrondi suit l'échelle de la variante (x1e6 pour « million »…)
        factor = abs(g / gold.raw) if gold.raw else 1.0
        tolerance = max(rel_tol * abs(g), rounding_tol * factor) + 1e-12
        if any(abs(p - g) <= tolerance for p in predicted.variants()):
            return True
    return False


_FINAL_ANSWER = re.compile(r"final answer\s*:", re.IGNORECASE)


def extract_final_answer(prediction: str) -> str:
    """Renvoie ce qui suit le dernier « Final answer: », ou tout le texte à défaut."""
    matches = list(_FINAL_ANSWER.finditer(prediction))
    return prediction[matches[-1].end() :].strip() if matches else prediction


def numeric_correct(gold_answer: str, prediction: str, rel_tol: float = DEFAULT_REL_TOL) -> bool:
    gold_numbers = parse_numbers(gold_answer)
    if not gold_numbers:
        return False
    gold = gold_numbers[0]
    # Seule la réponse finale compte : un chiffre intermédiaire juste ne suffit pas
    candidates = parse_numbers(extract_final_answer(prediction))
    return any(numbers_match(gold, p, rel_tol) for p in candidates)


def is_abstention(prediction: str) -> bool:
    lowered = prediction.lower()
    return any(pattern in lowered for pattern in ABSTAIN_PATTERNS)


def page_hit(pages: Iterable[tuple[str, int]], gold: Iterable[tuple[str, int]]) -> bool:
    return not set(pages).isdisjoint(set(gold))


def citation_precision(
    cited: list[tuple[str, int]], gold: Iterable[tuple[str, int]]
) -> float | None:
    """Part des passages cités qui sont des pages justificatives. None si rien n'est cité."""
    if not cited:
        return None
    gold_set = set(gold)
    return sum(page in gold_set for page in cited) / len(cited)
