"""Métriques calculées par règles, sans LLM : exactitude numérique et recherche."""

import re
from collections.abc import Iterable

from ledger_agent.answer_format import extract_final_answer
from ledger_agent.numbers import NumberMention, parse_numbers

_NUMERIC_ANSWER = re.compile(
    r"[-(]?\s*\$?\s*(\d{1,3}(,\d{3})+|\d+)?(\.\d+)?\s*"
    r"(%|percent|thousand|million|billion|trillion)?\s*\)?\.?",
    re.IGNORECASE,
)

DEFAULT_REL_TOL = 0.01


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
