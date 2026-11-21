from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from .contracts import AgentResult, SQLCandidate, StageRecord, VerificationIssue, VerificationResult
from .direct import _strip_fence, generate_direct_sql
from .execution import SQLExecutionError, execute_readonly
from .providers import ModelProvider
from .verifier import verify_sql_preflight
from .semantic_contracts import detect_semantic_concerns
from .semantic_contract import extract_semantic_contract
from .catalog import DatabaseCatalog
# Typed SQLGlot plan verification is lazy-loaded ONLY when explicitly opted in.


GUARDED_TREATMENT_ID = "langgraph_preflight_plus_internal_cot_repair_v1"
SEMANTIC_REVISION_TREATMENT_ID = "clause_guided_semantic_revision_v1"
_SEMANTIC_ISSUE_TYPES = {
    "NONE",
    "PROJECTION",
    "GRAIN_AGGREGATION",
    "JOIN",
    "FILTER_VALUE",
    "ORDER_TOPK",
    "OTHER",
}


@dataclass(frozen=True)
class GuardedContext:
    database: Path
    provider: ModelProvider
    max_repairs: int = 2
    max_rows: int = 100
    semantic_revision: bool = False
    semantic_contract_checks: bool = False
    semantic_cost_guard: bool = False
    semantic_plan_verifier: bool = False
    semantic_plan_policy: str = "always"


class GuardedState(TypedDict, total=False):
    question: str
    schema_context: str
    evidence: str | None
    candidate: SQLCandidate | None
    verification: VerificationResult | None
    semantic_revised: bool
    semantic_backup: SQLCandidate | None
    semantic_repair_attempted: bool
    semantic_plan_extraction: Any
    semantic_plan_state: str
    semantic_plan_repair_attempted: bool
    semantic_router_action: str
    stages: tuple[StageRecord, ...]
    result: AgentResult | None


def _append_stage(state: GuardedState, name: str, summary: str) -> tuple[StageRecord, ...]:
    return (*state.get("stages", ()), StageRecord(name, summary))


def _generate(state: GuardedState, runtime) -> dict[str, Any]:
    candidate = state.get("candidate")
    if candidate is None:
        candidate = generate_direct_sql(
            state["question"], state["schema_context"], runtime.context.provider,
            external_evidence=state.get("evidence"),
        )
    return {"candidate": candidate, "stages": _append_stage(state, "sql", candidate.sql)}


def _verify(state: GuardedState, runtime) -> dict[str, Any]:
    candidate = state["candidate"]
    verification = verify_sql_preflight(runtime.context.database, candidate.sql)
    advisories = ()
    backup = None
    # Opt-in only. Preserve verified executable original before any
    # speculative semantic repair, rather than losing a safe fallback.
    if verification.ok and runtime.context.semantic_contract_checks:
        concerns = detect_semantic_concerns(state["question"], candidate.sql)
        blocking = tuple(
            VerificationIssue(code="semantic_" + concern.code, message=concern.message)
            for concern in concerns if concern.blocking
        )
        advisories = tuple(concern.code for concern in concerns if not concern.blocking)
        if blocking:
            if runtime.context.semantic_cost_guard and state.get("semantic_backup") is None:
                backup = candidate
            verification = VerificationResult(ok=False, issues=blocking)
    detail = "PASS" if verification.ok else "; ".join(issue.code for issue in verification.issues)
    stages = _append_stage(state, "verify", detail)
    if advisories:
        stages = (*stages, StageRecord("semantic_advisory", "; ".join(advisories)))
    result = {"verification": verification, "stages": stages}
    if backup is not None:
        result["semantic_backup"] = backup


    return result


def _semantic_router(state: GuardedState, runtime) -> dict[str, Any]:
    """Cheap deterministic routing; skipped checks remain UNVERIFIED, never PASS.

    Always-on is the reference arm. Selective routing is explicitly experimental
    because risk heuristics can miss executable-but-semantically-wrong queries.
    """
    policy = runtime.context.semantic_plan_policy
    q = state["question"].casefold()
    sql = state["candidate"].sql.casefold()
    risky = bool(re.search(
        r"\\b(count|how many|average|sum|total|percentage|percent|ratio|rate|"
        r"maximum|minimum|most|least|top|first|last|group|per|each|"
        r"connected|parent|child|between|where|with|whose|which)\\b", q
    ) or re.search(
        r"\\b(join|group\\s+by|having|order\\s+by|count\\s*\\(|"
        r"sum\\s*\\(|avg\\s*\\(|distinct|union|intersect|except)\\b", sql
    ))
    action = "semantic_plan_contract" if policy == "always" or risky else "execute"
    reason = ("always" if policy == "always" else "structural_risk" if risky else "no_trigger_abstain")
    return {
        "semantic_router_action": action,
        "semantic_plan_state": "unverified",
        "stages": _append_stage(state, "semantic_router", reason + ":" + action),
    }


def _semantic_router_route(state: GuardedState) -> str:
    return state["semantic_router_action"]


def _semantic_plan_contract(state: GuardedState, runtime) -> dict[str, Any]:
    """Candidate-independent Qwen call, once per query, stored in Graph state."""
    extraction = extract_semantic_contract(
        provider=runtime.context.provider,
        question=state["question"],
        schema_context=state["schema_context"],
        evidence=state.get("evidence"),
    )
    detail = "ok" if extraction.status == "ok" else "skipped:" + extraction.reason
    return {
        "semantic_plan_extraction": extraction,
        "stages": _append_stage(state, "semantic_plan_contract", detail),
    }


def _semantic_plan_verify(state: GuardedState, runtime) -> dict[str, Any]:
    """AST/schema comparison. NEVER calls Qwen, uses frozen cached plan."""
    from types import SimpleNamespace

    try:
        from .typed_plan_gate import verify_plan_against_sql
        verdict = verify_plan_against_sql(
            extraction=state["semantic_plan_extraction"],
            sql=state["candidate"].sql,
            catalog=DatabaseCatalog.from_sqlite(runtime.context.database),
        )
    except (ImportError, ValueError):
        verdict = SimpleNamespace(status="unverified", reason="plan_runtime_unavailable", reasons=())
    result = {"semantic_plan_state": verdict.status}
    if verdict.status == "conflict":
        result["verification"] = VerificationResult(ok=False, issues=tuple(
            VerificationIssue(
                code="semantic_plan_" + code.casefold(),
                message="Independent question/schema semantic contract conflicts with SQL: " + code,
            )
            for code in verdict.reasons
        ))
        detail = "CONFLICT: " + ",".join(verdict.reasons)
    elif verdict.status == "consistent":
        detail = "CONTRACT_CONSISTENT_NOT_GOLD_CORRECT"
    else:
        detail = "UNVERIFIED: " + verdict.reason
    result["stages"] = _append_stage(state, "semantic_plan_verify", detail)
    return result


def _semantic_plan_route(state: GuardedState, runtime) -> str:
    if state.get("semantic_plan_state") != "conflict":
        return "execute"
    if (state.get("semantic_plan_repair_attempted", False)
        or state["candidate"].attempt >= runtime.context.max_repairs):
        return "semantic_plan_refuse"
    return "semantic_plan_repair"


def _semantic_plan_repair(state: GuardedState, runtime) -> dict[str, Any]:
    """Exactly one violation-targeted model call, preserving the original plan."""
    codes = ", ".join(issue.code for issue in state["verification"].issues)
    prompt = (
        f"Question:\\n{state['question']}\\n\\n"
        f"Schema context:\\n{state['schema_context']}\\n\\n"
        f"SQL with semantic contradictions:\\n{state['candidate'].sql}\\n\\n"
        f"Verifier violations:\\n{codes}\\n\\n"
        "Correct only the named contradictions. Preserve unrelated clauses. "
        "Return one read-only SQLite SELECT or CTE with no explanation."
    )
    fixed = runtime.context.provider.complete_text(
        system="Fix verified SQL semantic contradictions. Return SQLite SQL only.",
        user=prompt,
    )
    candidate = SQLCandidate(
        sql=_strip_fence(fixed),
        attempt=state["candidate"].attempt + 1,
    )
    return {
        "candidate": candidate,
        "semantic_plan_repair_attempted": True,
        "stages": _append_stage(state, "semantic_plan_repair", codes),
    }


def _semantic_plan_refuse(state: GuardedState) -> dict[str, Any]:
    return {"result": AgentResult(
        status="verification_failed", question=state["question"],
        candidate=state.get("candidate"),
        verification=state.get("verification"),
        message="semantic plan contradiction unresolved under one-repair budget; SQL not executed",
        stages=state.get("stages", ()),
    )}


def _repair(state: GuardedState, runtime) -> dict[str, Any]:
    candidate = state["candidate"]
    verification = state["verification"]
    issue_lines = "\n".join(f"- {issue.code}: {issue.message}" for issue in verification.issues)
    prompt = f"Question:\n{state['question']}\n\nSchema context:\n{state['schema_context']}\n\n"
    if state.get("evidence") is not None:
        prompt += f"External evidence:\n{state['evidence']}\n\n"
    prompt += (
        f"Rejected SQL:\n{candidate.sql}\n\n"
        f"Deterministic verifier issues:\n{issue_lines}\n\n"
        "Reason step-by-step internally before answering. Check the user intent, the relevant schema/table/column relationships, "
        "the rejected SQL, and every deterministic verifier/compiler issue. Prefer the smallest correction that fixes the failure "
        "without changing the requested semantics. Do not reveal that reasoning. Return corrected SQL only."
    )
    response = runtime.context.provider.complete_text(
        system=(
            "Repair one SQLite query. Reason internally, but return exactly one read-only SELECT or CTE and no markdown, "
            "commentary, chain-of-thought, or additional statements."
        ),
        user=prompt,
    )
    repaired = SQLCandidate(sql=_strip_fence(response), attempt=candidate.attempt + 1)
    result = {"candidate": repaired, "stages": _append_stage(state, "repair", f"attempt={repaired.attempt}: {repaired.sql}")}
    if getattr(runtime.context, "semantic_cost_guard", False) and state.get("semantic_backup") is not None:
        result["semantic_repair_attempted"] = True
    return result


def _semantic_revise(state: GuardedState, runtime) -> dict[str, Any]:
    candidate = state["candidate"]
    prompt = f"Question:\n{state['question']}\n\nSchema context:\n{state['schema_context']}\n\n"
    if state.get("evidence") is not None:
        prompt += f"External evidence:\n{state['evidence']}\n\n"
    prompt += f"Candidate SQL:\n{candidate.sql}\n"
    response = runtime.context.provider.complete_text(
        system=(
            "Review one read-only SQLite query for semantic faithfulness to the user question. "
            "Check exactly these clause-level failure types: PROJECTION (requested output), "
            "GRAIN_AGGREGATION (entity grain, COUNT/AVG/SUM/GROUP BY/percentage), JOIN (relationship or join key), "
            "FILTER_VALUE (missing/wrong condition or literal), and ORDER_TOPK (most/least/first/last/top-k). "
            "Prefer no change unless a concrete mismatch exists. Return JSON only with keys: "
            "changed (boolean), issue_type (NONE, PROJECTION, GRAIN_AGGREGATION, JOIN, FILTER_VALUE, ORDER_TOPK, OTHER), "
            "and sql (one read-only SELECT or CTE). If correct, return changed=false, issue_type=NONE, and the original SQL."
        ),
        user=prompt,
    )

    revised_sql = candidate.sql
    issue_type = "OTHER"
    changed = False
    try:
        payload = json.loads(response.strip())
        if not isinstance(payload, dict):
            raise ValueError("semantic revision response must be a JSON object")
        issue_type = str(payload.get("issue_type", "OTHER")).upper()
        if issue_type not in _SEMANTIC_ISSUE_TYPES:
            issue_type = "OTHER"
        requested_change = payload.get("changed") is True
        proposed_sql = str(payload.get("sql", "")).strip()
        if requested_change and proposed_sql:
            revised_sql = _strip_fence(proposed_sql)
            changed = revised_sql != candidate.sql
        elif not requested_change:
            issue_type = "NONE" if issue_type == "NONE" else issue_type
    except (TypeError, ValueError, json.JSONDecodeError):
        issue_type = "OTHER"

    revised = SQLCandidate(sql=revised_sql, attempt=candidate.attempt)
    summary = f"{issue_type}: changed={str(changed).lower()}"
    return {
        "candidate": revised,
        "semantic_revised": True,
        "stages": _append_stage(state, "semantic_revision", summary),
    }


def _execute(state: GuardedState, runtime) -> dict[str, Any]:
    candidate = state["candidate"]
    verification = state["verification"]
    try:
        execution = execute_readonly(runtime.context.database, candidate.sql, max_rows=runtime.context.max_rows)
    except SQLExecutionError as exc:
        stages = _append_stage(state, "execute", f"failed: {exc}")
        return {"result": AgentResult(status="execution_failed", question=state["question"], candidate=candidate, verification=verification, message=str(exc), stages=stages), "stages": stages}
    stages = _append_stage(state, "execute", f"rows={len(execution.rows)}")
    semantic_unknown = (getattr(runtime.context, "semantic_plan_verifier", False)
                        and state.get("semantic_plan_state") != "consistent")
    status = "semantic_unverified" if semantic_unknown else "ok"
    message = ("SQL executed but semantic plan could not be independently verified"
               if semantic_unknown else "")
    return {"result": AgentResult(status=status, question=state["question"], candidate=candidate, verification=verification, rows=execution.rows, columns=execution.columns, truncated=execution.truncated, message=message, stages=stages), "stages": stages}


def _refuse(state: GuardedState) -> dict[str, Any]:
    return {"result": AgentResult(status="verification_failed", question=state["question"], candidate=state.get("candidate"), verification=state.get("verification"), message="deterministic SQL verification failed after repair budget; SQL was not executed", stages=state.get("stages", ()))}


def _semantic_fallback(state: GuardedState, runtime) -> dict[str, Any]:
    """Fail open to original *preflight-valid* SQL; never invent a success score."""
    original = state.get("semantic_backup")
    if original is None:
        raise RuntimeError("semantic fallback requires saved original SQL")
    checked = verify_sql_preflight(runtime.context.database, original.sql)
    if not checked.ok:
        raise RuntimeError("original previously preflight-valid SQL was not stable")
    return {
        "candidate": original,
        "verification": checked,
        "stages": _append_stage(state, "semantic_fallback",
                                "original preflight-valid SQL retained; one semantic repair failed or budget=0"),
    }


def _verify_route(state: GuardedState, runtime) -> str:
    # One independent semantic-plan repair only; do not loop until the SQL passes.
    # A still-contradictory or syntactically broken repair cannot be executed.
    if (getattr(runtime.context, "semantic_plan_verifier", False)
        and state.get("semantic_plan_repair_attempted", False)
        and not state["verification"].ok):
        return "refuse"
    if (runtime.context.semantic_cost_guard
        and state.get("semantic_backup") is not None
        and not state["verification"].ok
        and (state.get("semantic_repair_attempted", False)
             or state["candidate"].attempt >= runtime.context.max_repairs)):
        return "semantic_fallback"
    if state["verification"].ok:
        if runtime.context.semantic_revision and not state.get("semantic_revised", False):
            return "semantic_revision"
        return "execute"
    if state["candidate"].attempt < runtime.context.max_repairs:
        return "repair"
    return "refuse"


def _build_guarded_graph(*, semantic_revision: bool = False, semantic_cost_guard: bool = False):
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(GuardedState, context_schema=GuardedContext)
    graph.add_node("generate", _generate)
    graph.add_node("verify", _verify)
    graph.add_node("repair", _repair)
    if semantic_revision:
        graph.add_node("semantic_revision", _semantic_revise)
    graph.add_node("execute", _execute)
    graph.add_node("refuse", _refuse)
    if semantic_cost_guard:
        graph.add_node("semantic_fallback", _semantic_fallback)
        graph.add_edge("semantic_fallback", "execute")
    graph.add_edge(START, "generate")
    graph.add_edge("generate", "verify")
    routes = {"execute": "execute", "repair": "repair", "refuse": "refuse"}
    if semantic_revision:
        routes["semantic_revision"] = "semantic_revision"
    if semantic_cost_guard:
        routes["semantic_fallback"] = "semantic_fallback"
    graph.add_conditional_edges("verify", _verify_route, routes)
    graph.add_edge("repair", "verify")
    if semantic_revision:
        graph.add_edge("semantic_revision", "verify")
    graph.add_edge("execute", END)
    graph.add_edge("refuse", END)
    return graph.compile()


def run_guarded(
    *, database: str | Path, provider: ModelProvider, question: str,
    schema_context: str, evidence: str | None = None,
    initial_candidate: SQLCandidate | None = None, max_repairs: int = 2,
    max_rows: int = 100, semantic_revision: bool = False,
    semantic_contract_checks: bool = False,
    semantic_cost_guard: bool = False,
    semantic_plan_verifier: bool = False,
) -> AgentResult:
    if semantic_plan_verifier and (semantic_cost_guard or semantic_contract_checks or semantic_revision):
        raise ValueError("typed semantic plan verifier is exclusive to other experimental semantic layers")
    if semantic_cost_guard and (not semantic_contract_checks or semantic_revision):
        raise ValueError("cost guard requires semantic contracts and excludes universal semantic revision")
    if max_repairs < 0:
        raise ValueError("repair budget must be non-negative")
    database = Path(database)
    output = _build_guarded_graph(
        semantic_revision=semantic_revision, semantic_cost_guard=semantic_cost_guard
    ).invoke(
        {
            "question": question,
            "schema_context": schema_context,
            "evidence": evidence,
            "candidate": initial_candidate,
            "semantic_revised": False,
            "semantic_backup": None,
            "semantic_repair_attempted": False,
            "semantic_plan_extraction": None,
            "semantic_plan_state": "unverified",
            "semantic_plan_repair_attempted": False,
            "stages": (),
        },
        config={"recursion_limit": 2 * max(0, max_repairs) + 10},
        context=GuardedContext(
            database=database,
            provider=provider,
            max_repairs=max_repairs,
            max_rows=max_rows,
            semantic_revision=semantic_revision,
            semantic_contract_checks=semantic_contract_checks,
            semantic_cost_guard=semantic_cost_guard,
            semantic_plan_verifier=semantic_plan_verifier,
        ),
    )
    result = output["result"] if isinstance(output, dict) else output.result
    if result is None:
        raise RuntimeError("Guarded graph completed without a result")
    return result
