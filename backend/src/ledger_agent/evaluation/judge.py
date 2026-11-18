"""Juge LLM pour les réponses rédigées, avec une grille fixe et une sortie JSON."""

import hashlib
import json
from dataclasses import dataclass

from ledger_agent.llm.types import CompletionParams, LLMClient, Message

# La longueur de la réponse ne doit pas jouer : les juges LLM favorisent sinon les réponses longues
JUDGE_PROMPT = """You grade answers to financial questions against a reference answer.

Rules:
- "correct": the candidate reaches the same conclusion as the reference and its key
  figures agree (small rounding differences are fine).
- "incorrect": wrong conclusion, wrong key figure, missing the key point of the
  reference, or contradicts the reference.
- "abstained": the candidate declines to answer or says the information is missing.
Judge only the substance. The length, tone or amount of detail of the candidate
must not influence the verdict.

Reply with JSON only, in this shape:
{"verdict": "correct" | "incorrect" | "abstained", "reason": "<one sentence>"}"""

JUDGE_PROMPT_SHA = hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest()[:12]

VERDICTS = ("correct", "incorrect", "abstained")


@dataclass(frozen=True, slots=True)
class Verdict:
    verdict: str
    reason: str


def build_judge_messages(question: str, reference: str, candidate: str) -> list[Message]:
    return [
        {"role": "system", "content": JUDGE_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\nReference answer:\n{reference}\n\n"
                f"Candidate answer:\n{candidate}"
            ),
        },
    ]


def parse_verdict(text: str) -> Verdict:
    # Certains modèles entourent le JSON de balises Markdown
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        payload = json.loads(cleaned)
        verdict = str(payload.get("verdict", "")).lower()
        reason = str(payload.get("reason", ""))
    except (json.JSONDecodeError, AttributeError):
        return Verdict("invalid", text[:200])
    return Verdict(verdict if verdict in VERDICTS else "invalid", reason)


class LLMJudge:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self._params = CompletionParams(max_tokens=300, temperature=0.0, json_mode=True)

    @property
    def model(self) -> str:
        return self._llm.model

    def grade(self, question: str, reference: str, candidate: str) -> Verdict:
        response = self._llm.complete(
            build_judge_messages(question, reference, candidate), self._params
        )
        return parse_verdict(response.text)
