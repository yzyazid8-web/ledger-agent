"""La clé du cache ne doit pas changer pour les appels sans outils."""

import hashlib
import json

from ledger_agent.llm.cache import cache_key
from ledger_agent.llm.types import CompletionParams, Message

MESSAGES: list[Message] = [{"role": "user", "content": "hello"}]


def test_calls_without_tools_keep_their_previous_cache_key() -> None:
    params = CompletionParams(max_tokens=10)
    old_fields = {
        "max_tokens": 10,
        "temperature": 0.0,
        "reasoning_effort": None,
        "json_mode": False,
    }
    raw = json.dumps(
        {"model": "m", "messages": MESSAGES, "params": old_fields},
        sort_keys=True,
        ensure_ascii=False,
    )

    assert cache_key("m", MESSAGES, params) == hashlib.sha256(raw.encode("utf-8")).hexdigest()


def test_tools_and_tool_choice_change_the_key() -> None:
    tool = {"type": "function", "function": {"name": "f"}}
    plain = cache_key("m", MESSAGES, CompletionParams())
    with_tools = cache_key("m", MESSAGES, CompletionParams(tools=(tool,)))
    forced = cache_key("m", MESSAGES, CompletionParams(tools=(tool,), tool_choice="required"))

    assert len({plain, with_tools, forced}) == 3
