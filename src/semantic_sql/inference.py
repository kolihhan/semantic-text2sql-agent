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
from .adaptive_probe import semantic_risk, schema_probe, data_probe, preserves_filter_guard


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
    adaptive_physical: bool = False


class GuardedState(TypedDict, total=False):
    question: str
    schema_context: str
    evidence: str | None
    candidate: SQLCandidate | None
    verification: VerificationResult | None
    semantic_revised: bool
    adaptive_issue: str | None
    adaptive_original: SQLCandidate | None
    adaptive_evidence: str
    adaptive_action: str
    adaptive_probed: tuple[str, ...]
    adaptive_attempted: bool
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


def _adaptive_decide(state: GuardedState, runtime) -> dict[str, Any]:
    """Graph state router: inspect the candidate, not a generic LLM reviewer."""
    issue = semantic_risk(state["question"], state["candidate"].sql)
    if issue is None or runtime.context.max_repairs == 0:
        return {
            "adaptive_issue": issue,
            "adaptive_action": "execute",
            "stages": _append_stage(state, "adaptive_decide", "NO_PROBE: no supported high-signal risk or zero budget"),
        }
    target = "adaptive_schema_probe" if issue == "COUNT_JOIN_FANOUT_RISK" else "adaptive_data_probe"
    return {
        "adaptive_issue": issue,
        "adaptive_original": state["candidate"],
        "adaptive_action": target,
        "stages": _append_stage(state, "adaptive_decide", f"{issue}: {target}"),
    }


def _adaptive_route(state: GuardedState) -> str:
    return state["adaptive_action"]


def _adaptive_schema_probe(state: GuardedState, runtime) -> dict[str, Any]:
    if "schema" in state.get("adaptive_probed", ()):
        raise RuntimeError("adaptive schema probe repeated")
    evidence, has_fk = schema_probe(runtime.context.database, state["candidate"].sql)
    # Without an explicitly registered physical link we cannot infer join grain.
    action = "adaptive_data_probe" if has_fk else "execute"
    return {
        "adaptive_evidence": evidence,
        "adaptive_action": action,
        "adaptive_probed": (*state.get("adaptive_probed", ()), "schema"),
        "stages": _append_stage(state, "adaptive_schema_probe", "FK_FOUND" if has_fk else "NO_FK_ABSTAIN"),
    }


def _adaptive_data_probe(state: GuardedState, runtime) -> dict[str, Any]:
    if "data" in state.get("adaptive_probed", ()):
        raise RuntimeError("adaptive data probe repeated")
    evidence = data_probe(runtime.context.database, state["candidate"].sql)
    # Real SQLite data are extra evidence, not semantic oracle labels.
    action = "adaptive_repair" if evidence.strip() else "execute"
    return {
        "adaptive_evidence": (state.get("adaptive_evidence", "") + "\n" + evidence)[:3000],
        "adaptive_action": action,
        "adaptive_probed": (*state.get("adaptive_probed", ()), "data"),
        "stages": _append_stage(state, "adaptive_data_probe", "READ_ONLY_SAMPLES" if evidence else "NO_EVIDENCE_ABSTAIN"),
    }


def _adaptive_repair(state: GuardedState, runtime) -> dict[str, Any]:
    if state.get("adaptive_attempted", False):
        raise RuntimeError("adaptive semantic model call repeated")
    original = state["adaptive_original"]
    prompt = (
        f"Question:\n{state['question']}\n\n"
        f"Schema:\n{state['schema_context']}\n\n"
        f"Initial preflight-valid SQL (NOT known correct):\n{original.sql}\n\n"
        f"Specific unresolved question/SQL risk:\n{state['adaptive_issue']}\n\n"
        f"Actual physical database evidence (NOT gold):\n{state['adaptive_evidence']}\n\n"
        "Return ONE defensible alternative read-only SQLite SELECT query that "
        "addresses the named risk, grounded in the actual schema. Preserve every "
        "original WHERE predicate unless the question explicitly contradicts it. "
        "Do not invent a table, join or value. If evidence does not support a "
        "different SQL, return the identical original SQL. SQL only."
    )
    response = runtime.context.provider.complete_text(
        system="Improve SQL only when physically justified; avoid semantic overcorrection. Return only one SQLite SELECT.",
        user=prompt,
    )
    return {
        "candidate": SQLCandidate(sql=_strip_fence(response), attempt=original.attempt + 1),
        "adaptive_attempted": True,
        "stages": _append_stage(state, "adaptive_repair", "one bounded Qwen SQL counterproposal"),
    }


def _adaptive_validate(state: GuardedState, runtime) -> dict[str, Any]:
    original = state["adaptive_original"]
    revised = state["candidate"]
    checked = verify_sql_preflight(runtime.context.database, revised.sql)
    original_risk = state["adaptive_issue"]
    remaining = semantic_risk(state["question"], revised.sql)
    if not checked.ok:
        reason = "preflight_invalid"
    elif original.sql.strip() == revised.sql.strip():
        reason = "unchanged"
    elif not preserves_filter_guard(original.sql, revised.sql):
        reason = "predicate_changed_or_complex"
    elif remaining == original_risk:
        reason = "structural_risk_unresolved"
    else:
        # This is merely a physically executable alternative, not proof of
        # semantic correctness. Both answers are scored separately by BIRD.
        reason = "candidate_eligible_for_experimental_selection"
    action = "execute" if reason == "candidate_eligible_for_experimental_selection" else "adaptive_fallback"
    return {
        "verification": checked,
        "adaptive_action": action,
        "stages": _append_stage(state, "adaptive_validate", reason),
    }


def _adaptive_fallback(state: GuardedState, runtime) -> dict[str, Any]:
    backup = state.get("adaptive_original")
    if backup is None:
        raise RuntimeError("adaptive fallback without original")
    checked = verify_sql_preflight(runtime.context.database, backup.sql)
    if not checked.ok:
        raise RuntimeError("original preflight-valid SQL changed")
    return {
        "candidate": backup,
        "verification": checked,
        "stages": _append_stage(state, "adaptive_fallback", "original retained after uncertain proposal"),
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
        if runtime.context.adaptive_physical:
            return "adaptive_decide"
        if runtime.context.semantic_revision and not state.get("semantic_revised", False):
            return "semantic_revision"
        return "execute"
    if state["candidate"].attempt < runtime.context.max_repairs:
        return "repair"
    return "refuse"


def _build_guarded_graph(*, semantic_revision: bool = False, adaptive_physical: bool = False):
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(GuardedState, context_schema=GuardedContext)
    graph.add_node("generate", _generate)
    graph.add_node("verify", _verify)
    graph.add_node("repair", _repair)
    if semantic_revision:
        graph.add_node("semantic_revision", _semantic_revise)
    if adaptive_physical:
        graph.add_node("adaptive_decide", _adaptive_decide)
        graph.add_node("adaptive_schema_probe", _adaptive_schema_probe)
        graph.add_node("adaptive_data_probe", _adaptive_data_probe)
        graph.add_node("adaptive_repair", _adaptive_repair)
        graph.add_node("adaptive_validate", _adaptive_validate)
        graph.add_node("adaptive_fallback", _adaptive_fallback)
    graph.add_node("execute", _execute)
    graph.add_node("refuse", _refuse)
    graph.add_edge(START, "generate")
    graph.add_edge("generate", "verify")
    routes = {"execute": "execute", "repair": "repair", "refuse": "refuse"}
    if adaptive_physical:
        routes["adaptive_decide"] = "adaptive_decide"
    if semantic_revision:
        routes["semantic_revision"] = "semantic_revision"
    graph.add_conditional_edges("verify", _verify_route, routes)
    graph.add_edge("repair", "verify")
    if semantic_revision:
        graph.add_edge("semantic_revision", "verify")
    if adaptive_physical:
        graph.add_conditional_edges(
            "adaptive_decide", _adaptive_route,
            {"execute": "execute",
             "adaptive_schema_probe": "adaptive_schema_probe",
             "adaptive_data_probe": "adaptive_data_probe"},
        )
        graph.add_conditional_edges(
            "adaptive_schema_probe", _adaptive_route,
            {"execute": "execute", "adaptive_data_probe": "adaptive_data_probe"},
        )
        graph.add_conditional_edges(
            "adaptive_data_probe", _adaptive_route,
            {"execute": "execute", "adaptive_repair": "adaptive_repair"},
        )
        graph.add_edge("adaptive_repair", "adaptive_validate")
        graph.add_conditional_edges(
            "adaptive_validate", _adaptive_route,
            {"execute": "execute", "adaptive_fallback": "adaptive_fallback"},
        )
        graph.add_edge("adaptive_fallback", "execute")
    graph.add_edge("execute", END)
    graph.add_edge("refuse", END)
    return graph.compile()


def run_guarded(
    *, database: str | Path, provider: ModelProvider, question: str,
    schema_context: str, evidence: str | None = None,
    initial_candidate: SQLCandidate | None = None, max_repairs: int = 2,
    max_rows: int = 100, semantic_revision: bool = False,
    adaptive_physical: bool = False,
) -> AgentResult:
    database = Path(database)
    if adaptive_physical and semantic_revision:
        raise ValueError("adaptive_physical and legacy semantic_revision are mutually exclusive")
    output = _build_guarded_graph(
        semantic_revision=semantic_revision, adaptive_physical=adaptive_physical,
    ).invoke(
        {
            "question": question,
            "schema_context": schema_context,
            "evidence": evidence,
            "candidate": initial_candidate,
            "semantic_revised": False,
            "adaptive_issue": None,
            "adaptive_original": None,
            "adaptive_evidence": "",
            "adaptive_action": "",
            "adaptive_probed": (),
            "adaptive_attempted": False,
            "stages": (),
        },
        config={"recursion_limit": 2 * max(0, max_repairs) + (20 if adaptive_physical else 10)},
        context=GuardedContext(
            database=database,
            provider=provider,
            max_repairs=max_repairs,
            max_rows=max_rows,
            semantic_revision=semantic_revision,
            adaptive_physical=adaptive_physical,
        ),
    )
    result = output["result"] if isinstance(output, dict) else output.result
    if result is None:
        raise RuntimeError("Guarded graph completed without a result")
    return result
