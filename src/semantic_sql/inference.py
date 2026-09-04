from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from .contracts import AgentResult, SQLCandidate, StageRecord, VerificationResult
from .direct import _strip_fence, generate_direct_sql
from .execution import SQLExecutionError, execute_readonly
from .providers import ModelProvider
from .verifier import verify_sql_preflight


GUARDED_TREATMENT_ID = "langgraph_preflight_plus_internal_cot_repair_v1"


@dataclass(frozen=True)
class GuardedContext:
    database: Path
    provider: ModelProvider
    max_repairs: int = 2
    max_rows: int = 100


class GuardedState(TypedDict, total=False):
    question: str
    schema_context: str
    evidence: str | None
    candidate: SQLCandidate | None
    verification: VerificationResult | None
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
        return "execute"
    if state["candidate"].attempt < runtime.context.max_repairs:
        return "repair"
    return "refuse"


def _build_guarded_graph():
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(GuardedState, context_schema=GuardedContext)
    graph.add_node("generate", _generate)
    graph.add_node("verify", _verify)
    graph.add_node("repair", _repair)
    graph.add_node("execute", _execute)
    graph.add_node("refuse", _refuse)
    graph.add_edge(START, "generate")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges("verify", _verify_route, {"execute": "execute", "repair": "repair", "refuse": "refuse"})
    graph.add_edge("repair", "verify")
    graph.add_edge("execute", END)
    graph.add_edge("refuse", END)
    return graph.compile()


def run_guarded(
    *, database: str | Path, provider: ModelProvider, question: str,
    schema_context: str, evidence: str | None = None,
    initial_candidate: SQLCandidate | None = None, max_repairs: int = 2,
    max_rows: int = 100,
) -> AgentResult:
    database = Path(database)
    output = _build_guarded_graph().invoke(
        {"question": question, "schema_context": schema_context, "evidence": evidence, "candidate": initial_candidate, "stages": ()},
        config={"recursion_limit": 2 * max(0, max_repairs) + 8},
        context=GuardedContext(database=database, provider=provider, max_repairs=max_repairs, max_rows=max_rows),
    )
    result = output["result"] if isinstance(output, dict) else output.result
    if result is None:
        raise RuntimeError("Guarded graph completed without a result")
    return result
