"""Candidate-independent, opt-in semantic-plan SQL verification.

A Qwen-generated contract is *hypothesis*, never ground truth. The gateway
requires physical schema-grounded references, AST parse, and an explicit
structural contradiction before it requests a repair. It never uses reference
SQL, labels or benchmark correctness when deciding what to do.
"""
from __future__ import annotations

from dataclasses import dataclass

from .catalog import DatabaseCatalog
from .semantic_contract import ContractExtraction, SemanticContract
from .contract_verifier import verify_semantic_contract
from .sql_semantics import parse_sql_semantics


@dataclass(frozen=True)
class PlanVerdict:
    status: str  # "consistent" | "conflict" | "unverified"
    reasons: tuple[str, ...] = ()
    reason: str = ""


# Unsafe-to-block from the recovered historical verifier:
# - joins can be valid even with no declared SQLite foreign key
# - normalized ORDER expressions and SQL aliases may be equivalent
# - distinct is only justified if the question independently requires it
# - a projected/calculated expression may satisfy an entity output request
# - literal may be represented by equivalent codes, LIKE, dates, parameters
BLOCKABLE = frozenset({
    "AGGREGATION_MISSING", "AGGREGATION_FUNCTION_MISMATCH",
    "GROUP_BY_MISSING", "GROUP_BY_MISMATCH",
    "ORDER_DIRECTION_MISMATCH", "TOPK_MISMATCH",
})


def _schema_grounded(contract: SemanticContract, catalog: DatabaseCatalog) -> bool:
    tables = {
        name.casefold(): {col.casefold() for col in info.columns}
        for name, info in catalog.tables.items()
    }
    refs = list(contract.projection) + list(contract.group_by)
    if contract.aggregation is not None:
        for ref in (contract.aggregation.target, contract.aggregation.entity_grain):
            if ref:
                refs.append(ref)
    for filter_spec in contract.filters:
        refs.append(filter_spec.field)
    for entity in contract.join_entities:
        if entity.casefold() not in tables:
            return False
    for raw in refs:
        if raw.casefold() in {"unresolved", "unknown", "none"}:
            continue
        if "." in raw:
            t, c = raw.rsplit(".", 1)
            if t.casefold() not in tables or c.casefold() not in tables[t.casefold()]:
                return False
        elif raw.casefold() not in tables:
            # Could be a conceptual entity or expression. Don't call this grounded.
            return False
    return True


def verify_plan_against_sql(
    *, extraction: ContractExtraction, sql: str, catalog: DatabaseCatalog,
) -> PlanVerdict:
    if extraction.status != "ok" or extraction.contract is None:
        return PlanVerdict("unverified", reason="contract_missing_or_low_confidence")
    plan = extraction.contract
    if plan.confidence != "high":
        return PlanVerdict("unverified", reason="contract_not_high_confidence")
    if not _schema_grounded(plan, catalog):
        return PlanVerdict("unverified", reason="contract_not_physically_grounded")

    # A confident plan without verifiable requirements cannot certify SQL.
    has_checks = bool(
        (plan.aggregation and plan.aggregation.function)
        or plan.group_by
        or (plan.ranking and (plan.ranking.direction or plan.ranking.top_k))
    )
    if not has_checks:
        return PlanVerdict("unverified", reason="no_provable_structural_requirements")

    try:
        parsed = parse_sql_semantics(sql)
    except (ImportError, ModuleNotFoundError):
        return PlanVerdict("unverified", reason="sqlglot_not_installed")
    if parsed.status != "ok" or parsed.semantics is None:
        return PlanVerdict("unverified", reason="sql_ast_unavailable")
    if parsed.semantics.has_derived_sources:
        return PlanVerdict("unverified", reason="derived_sql_not_yet_supported")

    check = verify_semantic_contract(plan, parsed.semantics, catalog)
    if check.status == "skipped":
        return PlanVerdict("unverified", reason=check.reason)
    strong = tuple(
        v.code for v in check.violations if v.code in BLOCKABLE
    )
    if strong:
        return PlanVerdict("conflict", reasons=strong)
    if check.violations:
        # Do not assert consistency when other semantic concerns remain.
        return PlanVerdict("unverified", reasons=tuple(v.code for v in check.violations),
                           reason="unproven_projection_relation_filter_or_expression")
    # Consistent == checked requirements did not contradict SQL, NOT SQL correctness.
    return PlanVerdict("consistent")
