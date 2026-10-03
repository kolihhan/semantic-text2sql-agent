from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .catalog import DatabaseCatalog
from .schema_graph import SchemaGraph
from .semantic_contract import SemanticContract
from .sql_semantics import ColumnRef, SQLSemantics


@dataclass(frozen=True)
class SCVViolation:
    code: str
    expected: str
    actual: str
    evidence: str
    confidence: Literal["high", "medium"] = "high"


@dataclass(frozen=True)
class SCVVerification:
    status: Literal["pass", "violation", "skipped"]
    violations: tuple[SCVViolation, ...] = ()
    reason: str = ""


def _contract_ref(value: str | None) -> ColumnRef | None:
    if not value:
        return None
    text = value.strip().casefold()
    if text in {"unresolved", "unknown", "none", "null"} or "." not in text:
        return None
    table, column = text.rsplit(".", 1)
    if not table or not column:
        return None
    return ColumnRef(table, column)


def _ref_compatible(expected: ColumnRef, actual: ColumnRef) -> bool:
    if expected == actual:
        return True
    return actual.table is None and expected.column == actual.column


def _refs_text(refs: tuple[ColumnRef, ...] | list[ColumnRef]) -> str:
    return ", ".join(
        f"{ref.table}.{ref.column}" if ref.table else ref.column for ref in refs
    ) or "(none)"


def _normalize_expression(value: str) -> str:
    return re.sub(r'[\s"\[\]]+', "", value.casefold())


def _literal_values(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        result: list[str] = []
        for item in value:
            result.extend(_literal_values(item))
        return tuple(result)
    if value is None:
        return ()
    return (str(value),)


def verify_semantic_contract(
    contract: SemanticContract,
    semantics: SQLSemantics,
    catalog: DatabaseCatalog,
) -> SCVVerification:
    if contract.confidence != "high":
        return SCVVerification(
            status="skipped",
            reason="contract_not_high_confidence",
        )

    violations: list[SCVViolation] = []

    expected_projection = tuple(
        ref for ref in (_contract_ref(value) for value in contract.projection) if ref is not None
    )
    if expected_projection:
        actual = set(semantics.projections)
        missing = tuple(
            ref
            for ref in expected_projection
            if not any(_ref_compatible(ref, candidate) for candidate in actual)
        )
        if missing:
            violations.append(
                SCVViolation(
                    code="PROJECTION_MISMATCH",
                    expected=_refs_text(list(expected_projection)),
                    actual=_refs_text(list(semantics.projections)),
                    evidence="resolved requested projection is absent from the SQL projection",
                    confidence=contract.confidence,
                )
            )

    aggregation = contract.aggregation
    matching_aggregates = []
    if aggregation is not None and aggregation.function is not None:
        expected_function = aggregation.function.upper()
        if not semantics.aggregations:
            violations.append(
                SCVViolation(
                    code="AGGREGATION_MISSING",
                    expected=expected_function,
                    actual="(none)",
                    evidence="semantic contract requires an aggregate operation",
                    confidence=contract.confidence,
                )
            )
        else:
            matching_aggregates = [
                item for item in semantics.aggregations if item.function.upper() == expected_function
            ]
            if not matching_aggregates:
                violations.append(
                    SCVViolation(
                        code="AGGREGATION_FUNCTION_MISMATCH",
                        expected=expected_function,
                        actual=", ".join(item.function for item in semantics.aggregations),
                        evidence="SQL uses a different aggregate function from the resolved contract",
                        confidence=contract.confidence,
                    )
                )
            else:
                expected_target = _contract_ref(aggregation.target)
                unresolved_target = any(item.target is None for item in matching_aggregates)
                actual_targets = [
                    item.target for item in matching_aggregates if item.target is not None
                ]
                if (
                    expected_target is not None
                    and not unresolved_target
                    and actual_targets
                    and not any(
                        _ref_compatible(expected_target, item)
                        for item in actual_targets
                    )
                ):
                    violations.append(
                        SCVViolation(
                            code="AGGREGATION_TARGET_MISMATCH",
                            expected=_refs_text([expected_target]),
                            actual=_refs_text(actual_targets),
                            evidence="aggregate function matches but operates on a different resolved field",
                            confidence=contract.confidence,
                        )
                    )
        if aggregation.distinct is True and matching_aggregates:
            expected_target = _contract_ref(aggregation.target)
            candidates = matching_aggregates
            if expected_target is not None:
                candidates = [item for item in candidates if item.target == expected_target]
            if candidates and not any(item.distinct for item in candidates):
                violations.append(
                    SCVViolation(
                        code="DISTINCT_REQUIRED",
                        expected=f"{expected_function}(DISTINCT ...)",
                        actual=f"{expected_function}(...)",
                        evidence="contract explicitly requires entity-grain deduplication",
                        confidence=contract.confidence,
                    )
                )

    expected_group = tuple(
        ref for ref in (_contract_ref(value) for value in contract.group_by) if ref is not None
    )
    if expected_group:
        if not semantics.group_by:
            violations.append(
                SCVViolation(
                    code="GROUP_BY_MISSING",
                    expected=_refs_text(list(expected_group)),
                    actual="(none)",
                    evidence="contract requires grouped aggregation",
                    confidence=contract.confidence,
                )
            )
        elif any(
            not any(_ref_compatible(expected, actual) for actual in semantics.group_by)
            for expected in expected_group
        ):
            violations.append(
                SCVViolation(
                    code="GROUP_BY_MISMATCH",
                    expected=_refs_text(list(expected_group)),
                    actual=_refs_text(list(semantics.group_by)),
                    evidence="SQL groups at a different resolved grain",
                    confidence=contract.confidence,
                )
            )

    ranking = contract.ranking
    if ranking is not None:
        first_order = semantics.order_by[0] if semantics.order_by else None
        if ranking.metric:
            expected_metric = _normalize_expression(ranking.metric)
            actual_metric = _normalize_expression(first_order.expression) if first_order else "(none)"
            if first_order is None or expected_metric != actual_metric:
                violations.append(
                    SCVViolation(
                        code="ORDER_METRIC_MISMATCH",
                        expected=ranking.metric,
                        actual=first_order.expression if first_order else "(none)",
                        evidence="ranking metric in SQL differs from the resolved semantic contract",
                        confidence=contract.confidence,
                    )
                )
        if ranking.direction is not None and (
            first_order is None or first_order.direction != ranking.direction
        ):
            violations.append(
                SCVViolation(
                    code="ORDER_DIRECTION_MISMATCH",
                    expected=ranking.direction,
                    actual=first_order.direction if first_order else "(none)",
                    evidence="ranking direction differs from the semantic contract",
                    confidence=contract.confidence,
                )
            )
        if ranking.ties == "all" and semantics.limit is not None:
            violations.append(
                SCVViolation(
                    code="TOPK_MISMATCH",
                    expected="all ties without a hard row LIMIT",
                    actual=f"LIMIT {semantics.limit}",
                    evidence="contract explicitly requires returning every row tied at the ranked boundary",
                    confidence=contract.confidence,
                )
            )
        elif ranking.top_k is not None and semantics.limit != ranking.top_k:
            violations.append(
                SCVViolation(
                    code="TOPK_MISMATCH",
                    expected=str(ranking.top_k),
                    actual=str(semantics.limit) if semantics.limit is not None else "(none)",
                    evidence="SQL LIMIT does not match the requested top-k",
                    confidence=contract.confidence,
                )
            )

    graph = SchemaGraph.from_catalog(catalog)
    physical_tables = {table.casefold() for table in catalog.tables}
    for join in semantics.joins:
        if join.left.table is None or join.right.table is None:
            continue
        if join.left.table not in physical_tables or join.right.table not in physical_tables:
            continue
        if join.left.table == join.right.table:
            continue
        if not graph.supports_fk_equality(
            join.left.table,
            join.left.column,
            join.right.table,
            join.right.column,
        ):
            violations.append(
                SCVViolation(
                    code="JOIN_EDGE_INVALID",
                    expected="declared foreign-key equality",
                    actual=(
                        f"{join.left.table}.{join.left.column} = "
                        f"{join.right.table}.{join.right.column}"
                    ),
                    evidence="explicit join equality is not an edge in the database FK graph",
                    confidence=contract.confidence,
                )
            )

    actual_literals = {literal.casefold() for literal in semantics.literals}
    for filter_spec in contract.filters:
        expected_literals = _literal_values(filter_spec.value)
        missing_literals = [
            literal for literal in expected_literals if literal.casefold() not in actual_literals
        ]
        if missing_literals:
            violations.append(
                SCVViolation(
                    code="FILTER_LITERAL_MISSING",
                    expected=", ".join(expected_literals),
                    actual=", ".join(semantics.literals) or "(none)",
                    evidence=f"high-confidence literal for {filter_spec.field} is absent from SQL",
                    confidence=contract.confidence,
                )
            )

    if violations:
        return SCVVerification(status="violation", violations=tuple(violations))
    return SCVVerification(status="pass")
