"""Optional one-pass entity-grain node for verified Guarded SQL.

No changes to Qwen generation or to the two-repair budget.
"""
from __future__ import annotations

from .contracts import SQLCandidate, StageRecord
from .grain_guard import correct_entity_grain
from .verifier import verify_sql_preflight


def apply_entity_grain_guard(state, runtime):
    original = state["candidate"]
    new_sql, reason = correct_entity_grain(
        runtime.context.database, state["question"], original.sql
    )
    if new_sql == original.sql:
        return {
            "grain_checked": True,
            "stages": (*state.get("stages", ()), StageRecord(
                "grain_guard", "unchanged:" + reason)),
        }
    verification = verify_sql_preflight(runtime.context.database, new_sql)
    if not verification.ok:
        return {
            "grain_checked": True,
            "stages": (*state.get("stages", ()), StageRecord(
                "grain_guard", "rejected_by_existing_preflight")),
        }
    return {
        "candidate": SQLCandidate(sql=new_sql, attempt=original.attempt),
        "verification": verification,
        "grain_checked": True,
        "stages": (*state.get("stages", ()), StageRecord(
            "grain_guard", "changed:" + reason)),
    }
