"""Statistiques pour petits échantillons : intervalles bootstrap, McNemar, kappa de Cohen."""

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 7


def bootstrap_ci(
    values: Sequence[bool | float], level: float = 0.95, samples: int = BOOTSTRAP_SAMPLES
) -> tuple[float, float]:
    """Intervalle de confiance par bootstrap percentile sur la moyenne."""
    if not values:
        return (math.nan, math.nan)
    rng = random.Random(BOOTSTRAP_SEED)
    data = [float(v) for v in values]
    n = len(data)
    means = sorted(sum(rng.choices(data, k=n)) / n for _ in range(samples))
    alpha = (1 - level) / 2
    return (means[int(alpha * samples)], means[int((1 - alpha) * samples) - 1])


def mcnemar_exact_p(only_a: int, only_b: int) -> float:
    """p-valeur bilatérale exacte du test de McNemar (loi binomiale sur les désaccords)."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail: float = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2.0 * tail)


@dataclass(frozen=True, slots=True)
class PairedComparison:
    n: int
    accuracy_a: float
    accuracy_b: float
    only_a: int
    only_b: int
    p_value: float
    diff_ci: tuple[float, float]

    @property
    def difference(self) -> float:
        return self.accuracy_b - self.accuracy_a


def compare_paired(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> PairedComparison:
    """Compare deux systèmes évalués sur les mêmes questions, dans le même ordre."""
    if len(correct_a) != len(correct_b) or not correct_a:
        raise ValueError("Les deux séries doivent être non vides et de même longueur")
    n = len(correct_a)
    only_a = sum(a and not b for a, b in zip(correct_a, correct_b, strict=True))
    only_b = sum(b and not a for a, b in zip(correct_a, correct_b, strict=True))
    diffs = [float(b) - float(a) for a, b in zip(correct_a, correct_b, strict=True)]
    return PairedComparison(
        n=n,
        accuracy_a=sum(correct_a) / n,
        accuracy_b=sum(correct_b) / n,
        only_a=only_a,
        only_b=only_b,
        p_value=mcnemar_exact_p(only_a, only_b),
        diff_ci=bootstrap_ci(diffs),
    )


def cohen_kappa(a: Sequence[bool], b: Sequence[bool]) -> float | None:
    """Accord entre deux évaluateurs binaires, corrigé du hasard (None si indéfini)."""
    if len(a) != len(b) or not a:
        return None
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    p_a, p_b = sum(a) / n, sum(b) / n
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    if expected == 1:
        return None
    return (observed - expected) / (1 - expected)
