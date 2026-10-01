from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from .contracts import AgentResult, SQLCandidate, StageRecord, VerificationResult
from .direct import _strip_fence, generate_direct_sql
from .execution import SQLExecutionError, execute_readonly
from .providers import ModelProvider
from .verifier import verify_sql_preflight


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


class GuardedState(TypedDict, total=False):
    question: str
    schema_context: str
    evidence: str | None
    candidate: SQLCandidate | None
    verification: VerificationResult | None
    semantic_revised: bool
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
    detail = "PASS" if verification.ok else "; ".join(issue.code for issue in verification.issues)
    return {"verification": verification, "stages": _append_stage(state, "verify", detail)}


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
    return {"candidate": repaired, "stages": _append_stage(state, "repair", f"attempt={repaired.attempt}: {repaired.sql}")}


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
    return {"result": AgentResult(status="ok", question=state["question"], candidate=candidate, verification=verification, rows=execution.rows, columns=execution.columns, truncated=execution.truncated, stages=stages), "stages": stages}


def _refuse(state: GuardedState) -> dict[str, Any]:
    return {"result": AgentResult(status="verification_failed", question=state["question"], candidate=state.get("candidate"), verification=state.get("verification"), message="deterministic SQL verification failed after repair budget; SQL was not executed", stages=state.get("stages", ()))}


def _verify_route(state: GuardedState, runtime) -> str:
    if state["verification"].ok:
        if runtime.context.semantic_revision and not state.get("semantic_revised", False):
            return "semantic_revision"
        return "execute"
    if state["candidate"].attempt < runtime.context.max_repairs:
        return "repair"
    return "refuse"


def _build_guarded_graph(*, semantic_revision: bool = False):
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(GuardedState, context_schema=GuardedContext)
    graph.add_node("generate", _generate)
    graph.add_node("verify", _verify)
    graph.add_node("repair", _repair)
    if semantic_revision:
        graph.add_node("semantic_revision", _semantic_revise)
    graph.add_node("execute", _execute)
    graph.add_node("refuse", _refuse)
    graph.add_edge(START, "generate")
    graph.add_edge("generate", "verify")
    routes = {"execute": "execute", "repair": "repair", "refuse": "refuse"}
    if semantic_revision:
        routes["semantic_revision"] = "semantic_revision"
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
) -> AgentResult:
    database = Path(database)
    output = _build_guarded_graph(semantic_revision=semantic_revision).invoke(
        {
            "question": question,
            "schema_context": schema_context,
            "evidence": evidence,
            "candidate": initial_candidate,
            "semantic_revised": False,
            "stages": (),
        },
        config={"recursion_limit": 2 * max(0, max_repairs) + 10},
        context=GuardedContext(
            database=database,
            provider=provider,
            max_repairs=max_repairs,
            max_rows=max_rows,
            semantic_revision=semantic_revision,
        ),
    )
    result = output["result"] if isinstance(output, dict) else output.result
    if result is None:
        raise RuntimeError("Guarded graph completed without a result")
    return result
