"""Format des réponses générées : réponse finale et abstention, définis en un seul endroit."""

import re

ABSTAIN_MARKER = "INSUFFICIENT_CONTEXT"

# Refus explicites, cherchés seulement dans une réponse finale courte et sans chiffre
REFUSAL_PATTERNS = (
    "cannot answer",
    "can't answer",
    "unable to answer",
    "unable to determine",
    "cannot be determined",
    "not enough information",
    "insufficient information",
)
MAX_REFUSAL_LENGTH = 200

_FINAL_ANSWER = re.compile(r"final answer\s*\**\s*:\s*\**", re.IGNORECASE)


def extract_final_answer(text: str) -> str:
    """Renvoie ce qui suit le dernier « Final answer: », ou tout le texte à défaut."""
    matches = list(_FINAL_ANSWER.finditer(text))
    return text[matches[-1].end() :].strip() if matches else text.strip()


def is_abstention(text: str) -> bool:
    if ABSTAIN_MARKER in text:
        return True
    final = extract_final_answer(text)
    # Une réponse finale qui avance un chiffre n'est jamais une abstention
    if re.search(r"\d", final) or len(final) > MAX_REFUSAL_LENGTH:
        return False
    lowered = final.lower()
    return any(pattern in lowered for pattern in REFUSAL_PATTERNS)
