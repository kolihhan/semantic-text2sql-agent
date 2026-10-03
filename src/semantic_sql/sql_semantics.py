from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError


@dataclass(frozen=True, order=True)
class ColumnRef:
    table: str | None
    column: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "table", self.table.casefold() if self.table else None)
        object.__setattr__(self, "column", self.column.casefold())


@dataclass(frozen=True)
class NormalizedAggregation:
    function: str
    target: ColumnRef | None
    distinct: bool = False


@dataclass(frozen=True)
class NormalizedOrder:
    expression: str
    direction: Literal["ASC", "DESC"]


@dataclass(frozen=True)
class NormalizedJoin:
    left: ColumnRef
    right: ColumnRef


@dataclass(frozen=True)
class SQLSemantics:
    projections: tuple[ColumnRef, ...] = ()
    aggregations: tuple[NormalizedAggregation, ...] = ()
    group_by: tuple[ColumnRef, ...] = ()
    order_by: tuple[NormalizedOrder, ...] = ()
    limit: int | None = None
    literals: tuple[str, ...] = ()
    joins: tuple[NormalizedJoin, ...] = ()
    distinct: bool = False


@dataclass(frozen=True)
class SQLSemanticsParse:
    status: Literal["ok", "skipped"]
    semantics: SQLSemantics | None
    reason: str = ""


def _outer_select(tree: exp.Expression) -> exp.Select | None:
    if isinstance(tree, exp.Select):
        return tree
    return tree.find(exp.Select)


def _alias_context(select: exp.Select) -> tuple[dict[str, str], set[str]]:
    aliases: dict[str, str] = {}
    physical_sources: set[str] = set()
    # Use tables visible in the outer query's FROM/JOIN clauses. CTE internals
    # may also appear in descendants, so only map every table name conservatively;
    # unqualified columns are resolved only when one source is visible.
    from_clause = select.args.get("from_")
    roots: list[exp.Expression] = []
    if from_clause is not None:
        roots.append(from_clause)
    roots.extend(select.args.get("joins") or ())

    for root in roots:
        for table in root.find_all(exp.Table):
            name = table.name.casefold()
            alias = table.alias_or_name.casefold()
            aliases[alias] = name
            aliases[name] = name
            physical_sources.add(name)
    return aliases, physical_sources


def _column_ref(
    column: exp.Column,
    aliases: dict[str, str],
    physical_sources: set[str],
) -> ColumnRef:
    raw_table = column.table.casefold() if column.table else None
    if raw_table:
        table = aliases.get(raw_table, raw_table)
    elif len(physical_sources) == 1:
        table = next(iter(physical_sources))
    else:
        table = None
    return ColumnRef(table, column.name)


def _direct_column(
    expression: exp.Expression,
    aliases: dict[str, str],
    physical_sources: set[str],
) -> ColumnRef | None:
    node = expression.this if isinstance(expression, exp.Alias) else expression
    return _column_ref(node, aliases, physical_sources) if isinstance(node, exp.Column) else None


def _aggregate_target(
    aggregate: exp.Expression,
    aliases: dict[str, str],
    physical_sources: set[str],
) -> tuple[ColumnRef | None, bool]:
    target = aggregate.this
    distinct = False
    if isinstance(target, exp.Distinct):
        distinct = True
        expressions = list(target.expressions)
        target = expressions[0] if len(expressions) == 1 else None
    if isinstance(target, exp.Star):
        return None, distinct
    if isinstance(target, exp.Column):
        return _column_ref(target, aliases, physical_sources), distinct
    return None, distinct


def _normalize_expression(
    expression: exp.Expression,
    aliases: dict[str, str],
    physical_sources: set[str],
) -> str:
    if isinstance(expression, exp.Column):
        ref = _column_ref(expression, aliases, physical_sources)
        return f"{ref.table}.{ref.column}" if ref.table else ref.column
    if isinstance(expression, (exp.Sum, exp.Count, exp.Avg, exp.Min, exp.Max)):
        target, distinct = _aggregate_target(expression, aliases, physical_sources)
        if target is None:
            inside = "*"
        else:
            inside = f"{target.table}.{target.column}" if target.table else target.column
        if distinct:
            inside = f"distinct {inside}"
        return f"{expression.key.casefold()}({inside})"
    return expression.sql(dialect="sqlite", normalize=True).casefold()


def parse_sql_semantics(sql: str) -> SQLSemanticsParse:
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except (ParseError, ValueError, TypeError) as exc:
        return SQLSemanticsParse(status="skipped", semantics=None, reason=str(exc) or "parse_error")

    select = _outer_select(tree)
    if select is None:
        return SQLSemanticsParse(status="skipped", semantics=None, reason="no_select")

    aliases, physical_sources = _alias_context(select)
    select_aliases: dict[str, str] = {}
    for expression in select.expressions:
        if isinstance(expression, exp.Alias):
            select_aliases[expression.alias.casefold()] = _normalize_expression(
                expression.this,
                aliases,
                physical_sources,
            )

    projections: list[ColumnRef] = []
    for expression in select.expressions:
        ref = _direct_column(expression, aliases, physical_sources)
        if ref is not None:
            projections.append(ref)

    aggregations: list[NormalizedAggregation] = []
    seen_aggregates: set[tuple[str, ColumnRef | None, bool]] = set()
    for aggregate in select.find_all(exp.AggFunc):
        function = aggregate.key.upper()
        target, distinct = _aggregate_target(aggregate, aliases, physical_sources)
        key = (function, target, distinct)
        if key not in seen_aggregates:
            aggregations.append(NormalizedAggregation(function, target, distinct))
            seen_aggregates.add(key)

    group_refs: list[ColumnRef] = []
    group = select.args.get("group")
    if group is not None:
        for expression in group.expressions:
            ref = _direct_column(expression, aliases, physical_sources)
            if ref is not None:
                group_refs.append(ref)

    orders: list[NormalizedOrder] = []
    order = select.args.get("order")
    if order is not None:
        for ordered in order.expressions:
            node = ordered.this if isinstance(ordered, exp.Ordered) else ordered
            direction: Literal["ASC", "DESC"] = (
                "DESC" if isinstance(ordered, exp.Ordered) and bool(ordered.args.get("desc")) else "ASC"
            )
            if (
                isinstance(node, exp.Column)
                and not node.table
                and node.name.casefold() in select_aliases
            ):
                normalized_order = select_aliases[node.name.casefold()]
            else:
                normalized_order = _normalize_expression(node, aliases, physical_sources)
            orders.append(
                NormalizedOrder(
                    expression=normalized_order,
                    direction=direction,
                )
            )

    limit_value: int | None = None
    limit = select.args.get("limit")
    if limit is not None:
        limit_expr = limit.expression
        if isinstance(limit_expr, exp.Literal) and not limit_expr.is_string:
            try:
                limit_value = int(limit_expr.this)
            except (TypeError, ValueError):
                limit_value = None

    joins: list[NormalizedJoin] = []
    for join in select.args.get("joins") or ():
        on = join.args.get("on")
        if on is None:
            continue
        for equality in on.find_all(exp.EQ):
            if isinstance(equality.left, exp.Column) and isinstance(equality.right, exp.Column):
                joins.append(
                    NormalizedJoin(
                        _column_ref(equality.left, aliases, physical_sources),
                        _column_ref(equality.right, aliases, physical_sources),
                    )
                )

    literals = tuple(str(literal.this) for literal in select.find_all(exp.Literal))

    return SQLSemanticsParse(
        status="ok",
        semantics=SQLSemantics(
            projections=tuple(projections),
            aggregations=tuple(aggregations),
            group_by=tuple(group_refs),
            order_by=tuple(orders),
            limit=limit_value,
            literals=literals,
            joins=tuple(joins),
            distinct=select.args.get("distinct") is not None,
        ),
    )
