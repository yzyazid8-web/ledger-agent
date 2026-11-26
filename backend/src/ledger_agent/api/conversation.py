"""Questions de suivi : devient une question autonome avant d'atteindre l'agent."""

import logging
import re
from typing import Any

from ledger_agent.llm.types import CompletionParams, LLMClient, LLMError, Message

logger = logging.getLogger(__name__)

# Seuls les derniers échanges servent : au-delà, ils coûtent des tokens sans aider
MAX_TURNS = 3
MAX_REWRITE_CHARS = 400

REWRITE_PROMPT = """You rewrite follow-up questions about US companies' SEC filings.
Given the conversation and the new question, return ONE standalone question in English that
keeps the company, the financial metric and the period implied by the conversation.
If the new question is already standalone, return it unchanged (translated to English if needed).
Answer with the question only, on one line, without quotes or explanation."""

# Préfixes que le modèle ajoute parfois malgré la consigne
_PREFIX = re.compile(r"^(?:standalone question|question|rewritten question)\s*:\s*", re.IGNORECASE)


def _conversation(history: list[dict[str, Any]], question: str) -> str:
    lines = []
    for turn in history[-MAX_TURNS:]:
        lines.append(f"User: {turn.get('question', '')}")
        if turn.get("answer"):
            lines.append(f"Assistant: {turn['answer']}")
    lines.append(f"New question: {question}")
    return "\n".join(lines)


class QuestionRewriter:
    """Reformule une question de suivi ; en cas de doute ou d'échec, la question reste inchangée."""

    def __init__(self, llm: LLMClient | None, params: CompletionParams | None = None) -> None:
        self._llm = llm
        self._params = params or CompletionParams(max_tokens=400, temperature=0.0)

    def rewrite(self, question: str, history: list[dict[str, Any]] | None) -> str:
        # Sans historique, rien à reformuler : aucun appel au modèle
        if self._llm is None or not history:
            return question
        messages: list[Message] = [
            {"role": "system", "content": REWRITE_PROMPT},
            {"role": "user", "content": _conversation(history, question)},
        ]
        try:
            response = self._llm.complete(messages, self._params)
        except LLMError as exc:
            # Mieux vaut la question telle quelle qu'une erreur : l'agent pourra s'abstenir
            logger.warning("Reformulation impossible, question transmise telle quelle : %s", exc)
            return question
        candidate = _PREFIX.sub("", response.text.strip()).strip().strip('"').strip()
        if not candidate or "\n" in candidate or len(candidate) > MAX_REWRITE_CHARS:
            logger.warning("Reformulation inutilisable, question transmise telle quelle")
            return question
        return candidate
