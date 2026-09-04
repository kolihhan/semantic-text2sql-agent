from __future__ import annotations

import re

from .contracts import SQLCandidate
from .providers import ModelProvider


_SYSTEM = """Generate one read-only SQLite SELECT query that answers the user's question using only the supplied schema context.
Return SQL only. Do not invent tables or columns that are not present in the schema context."""


def _strip_fence(text: str) -> str:
    text = text.strip()
    match = re.fullmatch(r"```(?:sql)?\s*(.*?)\s*```", text, flags=re.I | re.S)
    return match.group(1).strip() if match else text


def generate_direct_sql(
    question: str,
    schema_context: str,
    provider: ModelProvider,
    *,
    external_evidence: str | None = None,
) -> SQLCandidate:
    user = f"Question: {question}\nSchema context:\n{schema_context}"
    if external_evidence is not None:
        user += f"\nExternal evidence:\n{external_evidence}"
    sql = _strip_fence(provider.complete_text(system=_SYSTEM, user=user))
    return SQLCandidate(sql=sql, attempt=0)
