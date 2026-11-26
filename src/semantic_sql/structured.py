"""Application adapter for structured SQL generation and bounded recovery.

The existing read-only verifier remains the final authority for delivered output.
"""
from __future__ import annotations

import asyncio
from contextlib import closing
from pathlib import Path
import time

from .contracts import AgentResult, SQLCandidate, StageRecord
from .execution import SQLExecutionError, execute_readonly
from .verifier import verify_sql_preflight
from ._native.agent import Agent
from ._native.conversation import ConversationContext
from ._native.costs import BudgetExceeded
from ._native.db import connect_readonly
from ._native.llm import LLMResult


def run_structured(
    *, database: str | Path, provider, question: str, schema_context: str,
    evidence: str | None = None, max_rows: int = 100, max_repairs: int = 1,
) -> AgentResult:
    """Use a bounded structured exchange, then independently verify delivered SQL."""
    db = Path(database)
    calls = 0

    async def complete(messages, model, **kwargs):
        nonlocal calls
        if calls >= 3:
            raise BudgetExceeded("Three model calls per question is the hard limit")
        calls += 1
        started = time.monotonic()
        answer = provider.complete_messages(
            messages=messages, response_format=kwargs.get("response_format"),
            max_tokens=1024,
        )
        if not isinstance(answer, dict) or not isinstance(answer.get("text"), str):
            raise ValueError("Structured provider returned no text")
        return LLMResult(
            text=answer["text"],
            input_tokens=int(answer.get("input_tokens") or 0),
            cached_tokens=0,
            output_tokens=int(answer.get("output_tokens") or 0),
            latency_ms=float(answer.get("wall_s") or time.monotonic()-started)*1000,
            retry_ms=0,
            model=model,
            cost_usd=0,
        )

    prompt = question
    if evidence:
        prompt += "\nExternal evidence:\n" + evidence
    with closing(connect_readonly(db, timeout_seconds=30)) as conn:
        pipeline = Agent(
            model=getattr(provider, "model", "local-model"),
            conn=conn, budget=None, db_path=db,
            max_repairs=min(max_repairs, 1), complete_fn=complete,
            max_tokens=1024, temperature=0.0,
            pipeline_repairs=True,
        )
        outcome = asyncio.run(
            pipeline.ask(prompt, ConversationContext(schema_context, profile="benchmark"))
        )

    stages = [StageRecord("structured_generate", f"model_calls={len(outcome.llm_calls)}")]
    if outcome.refusal_retried:
        stages.append(StageRecord("refusal_retry", "answerable schema recheck"))
    if outcome.repaired:
        stages.append(StageRecord("repair", "bounded validation/execution correction"))
    if outcome.empty_retried:
        stages.append(StageRecord("empty_recheck", "read-only result re-evaluation"))
    if outcome.response_type != "query" or not outcome.sql:
        msg = outcome.message or outcome.error or "No executable SQL was produced"
        return AgentResult(status="verification_failed", question=question,
                           message=msg, stages=tuple(stages))
    candidate = SQLCandidate(outcome.sql, attempt=max(0, len(outcome.llm_calls)-1))
    verification = verify_sql_preflight(db, outcome.sql)
    stages.append(StageRecord("verify", "PASS" if verification.ok else "FAILED"))
    if not verification.ok:
        return AgentResult(status="verification_failed", question=question,
                           candidate=candidate, verification=verification,
                           message=outcome.error or "SQL failed deterministic verification",
                           stages=tuple(stages))
    try:
        result = execute_readonly(db, outcome.sql, max_rows=max_rows)
    except SQLExecutionError as exc:
        return AgentResult(status="execution_failed", question=question,
                           candidate=candidate, verification=verification,
                           message=str(exc), stages=tuple(stages))
    stages.append(StageRecord("execute", f"rows={len(result.rows)}"))
    return AgentResult(status="ok", question=question, candidate=candidate,
                       verification=verification, rows=result.rows,
                       columns=result.columns, truncated=result.truncated,
                       stages=tuple(stages))
