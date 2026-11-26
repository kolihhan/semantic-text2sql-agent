# MIT-licensed upstream-derived component; copyright notice in this directory's LICENSE.
"""Minimal SQL-only conversation memory."""

import json
from collections import deque
from dataclasses import dataclass

from .prompts import PROMPT_PROFILES


@dataclass(frozen=True)
class Turn:
    question: str
    sql: str
    row_count: int
    truncated: bool


class ConversationContext:
    def __init__(self, schema: str, max_turns: int = 4, profile: str = "product"):
        self.schema = schema
        self.template = PROMPT_PROFILES[profile]
        self.turns: deque[Turn] = deque(maxlen=max_turns)

    def build_messages(self, question: str) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.template.format(schema=self.schema)}]
        for turn in self.turns:
            note = f"at least {turn.row_count}" if turn.truncated else str(turn.row_count)
            messages.extend(
                [
                    {"role": "user", "content": turn.question},
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "response_type": "query",
                                "sql": turn.sql,
                                "message": None,
                            }
                        )
                        + f"\nRows returned: {note}.",
                    },
                ]
            )
        messages.append({"role": "user", "content": question})
        return messages

    def record(self, turn: Turn) -> None:
        self.turns.append(turn)

    def clear(self) -> None:
        self.turns.clear()
