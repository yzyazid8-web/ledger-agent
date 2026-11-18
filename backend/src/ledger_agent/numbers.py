"""Repérage des nombres dans un texte : montant, échelle, pourcentage, signe."""

import re
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

# Pas de lettre ni de chiffre juste avant : « FY2018 » ou « Q4 » ne sont pas des montants
_NUMBER = re.compile(
    r"(?<![A-Za-z0-9.])(?P<neg>[-(])?\s*(?P<dollar>\$)?\s*"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    r"\s*(?P<pct>%|percent\b)?"
    r"\s*(?P<scale>thousand|million|billion|trillion|mn|mm|bn|k|m|b)?\b",
    re.IGNORECASE,
)

# Mots qui expriment un écart négatif sans signe moins : « a decrease of 0.02 »
_NEGATIVE_WORDS = re.compile(
    r"\b(decrease|decreased|decline|declined|drop|dropped|fell|fall|lower|loss|negative|"
    r"down|reduction|reduced|shrunk|shrank|contraction)\b",
    re.IGNORECASE,
)
NEGATIVE_WINDOW = 40


@dataclass(frozen=True, slots=True)
class NumberMention:
    raw: float
    decimals: int
    is_percent: bool
    scale: float
    negative: bool = False
    monetary: bool = False
    text: str = ""

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

    @property
    def looks_like_year(self) -> bool:
        return (
            self.decimals == 0
            and self.scale == 1.0
            and not (self.is_percent or self.monetary)
            and 1900 <= self.raw <= 2100
        )


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
                text=match.group(0).strip(),
            )
        )
    return mentions
