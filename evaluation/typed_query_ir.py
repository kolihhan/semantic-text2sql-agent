"""Research-only typed query IR -> read-only SQLite SQL; fail closed on uncertainty.

No model, benchmark labels, SQL gold, value probing, or new dependencies.
Unknown operations and ambiguous grain return None so the existing Qwen
Guarded baseline can be kept unchanged by the future paired runner.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

from semantic_sql.catalog import DatabaseCatalog

_KEYS = {"from", "select", "joins", "where", "group_by", "grain", "order_by", "limit"}
_OPERATORS = {"=", "!=", "<>", "<", "<=", ">", ">=", "LIKE"}
_AGGREGATES = {"count", "sum", "avg", "min", "max"}
_ALIAS = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")


class UnsupportedIntent(ValueError):
    """The intent cannot be safely and unambiguously compiled."""


def _identifier(name: str) -> str:
    if not isinstance(name, str) or not name or "\x00" in name:
        raise UnsupportedIntent("invalid physical identifier")
    return '"' + name.replace('"', '""') + '"'


def _literal(value: Any) -> str:
    if isinstance(value, str):
        if "\x00" in value:
            raise UnsupportedIntent("NUL literal")
        return "'" + value.replace("'", "''") + "'"
    if type(value) is int:
        return str(value)
    if type(value) is float and math.isfinite(value):
        return repr(value)
    raise UnsupportedIntent("unsupported literal type")


def compile_typed_query(plan: object, catalog: DatabaseCatalog) -> str | None:
    """Compile only validated physical identifiers and a small typed SQL subset.

    A None result is an explicit abstention, NOT evidence that the baseline is wrong.
    """
    try:
        return _compile(plan, catalog)
    except (UnsupportedIntent, TypeError, KeyError):
        return None


def _compile(plan: object, catalog: DatabaseCatalog) -> str:
    if not isinstance(plan, dict) or set(plan) != _KEYS:
        raise UnsupportedIntent("unknown or missing intent fields")
    root = plan["from"]
    if not isinstance(root, str) or root not in catalog.tables:
        raise UnsupportedIntent("missing root table")
    visible = {root}

    def field(qualified: str, allowed: set[str] | None = None) -> str:
        if not isinstance(qualified, str) or qualified.count(".") != 1:
            raise UnsupportedIntent("field must be table.column")
        table, column = qualified.split(".")
        if table not in (visible if allowed is None else allowed):
            raise UnsupportedIntent("table out of scope")
        if table not in catalog.tables or column not in catalog.tables[table].columns:
            raise UnsupportedIntent("column not in physical schema")
        return f"{_identifier(table)}.{_identifier(column)}"

    joins = plan["joins"]
    if not isinstance(joins, list) or len(joins) > 8:
        raise UnsupportedIntent("invalid joins")
    sql_joins: list[str] = []
    declared = {(a, b, c, d) for a, b, c, d in catalog.foreign_keys}
    declared |= {(c, d, a, b) for a, b, c, d in catalog.foreign_keys}
    for join in joins:
        if not isinstance(join, dict) or set(join) != {"table", "on"}:
            raise UnsupportedIntent("invalid join")
        table, on = join["table"], join["on"]
        if not isinstance(table, str) or table not in catalog.tables or table in visible:
            raise UnsupportedIntent("invalid joined table")
        if not isinstance(on, list) or len(on) != 2 or not all(isinstance(c, str) for c in on):
            raise UnsupportedIntent("invalid join edge")
        left, right = on
        if left.count(".") != 1 or right.count(".") != 1:
            raise UnsupportedIntent("invalid join fields")
        lt, lc = left.split(".")
        rt, rc = right.split(".")
        if {lt, rt} & {table} != {table} or not ({lt, rt} - {table}) <= visible:
            raise UnsupportedIntent("join must attach one new table")
        if (lt, lc, rt, rc) not in declared:
            raise UnsupportedIntent("join must follow declared foreign key")
        permitted = visible | {table}
        sql_joins.append(f"JOIN {_identifier(table)} ON {field(left, permitted)} = {field(right, permitted)}")
        visible.add(table)

    select = plan["select"]
    if not isinstance(select, list) or not 1 <= len(select) <= 12:
        raise UnsupportedIntent("invalid projections")
    sql_projections: list[str] = []
    aliases: set[str] = set()
    nonaggregate: set[str] = set()
    has_aggregate = False
    for item in select:
        if not isinstance(item, dict) or not {"op", "column"} <= set(item) or set(item) - {"op", "column", "as", "distinct"}:
            raise UnsupportedIntent("unsupported projection")
        op = item["op"]
        if op != "column" and op not in _AGGREGATES:
            raise UnsupportedIntent("unknown projection operator")
        qualified = item["column"]
        distinct = item.get("distinct", False)
        if type(distinct) is not bool or (distinct and op != "count"):
            raise UnsupportedIntent("unsupported distinct")
        if op == "count" and qualified == "*":
            expression = "COUNT(*)"
            if distinct:
                raise UnsupportedIntent("COUNT DISTINCT star invalid")
        else:
            expression = field(qualified)
            if op != "column":
                expression = f"{op.upper()}({'DISTINCT ' if distinct else ''}{expression})"
            else:
                nonaggregate.add(qualified)
        has_aggregate |= op != "column"
        if "as" in item:
            alias = item["as"]
            if not isinstance(alias, str) or _ALIAS.fullmatch(alias) is None or alias in aliases:
                raise UnsupportedIntent("invalid alias")
            aliases.add(alias)
            expression += f" AS {_identifier(alias)}"
        sql_projections.append(expression)

    grain = plan["grain"]
    grouped = plan["group_by"]
    if not isinstance(grain, list) or not isinstance(grouped, list) or not all(isinstance(x, str) for x in grain + grouped):
        raise UnsupportedIntent("invalid grain")
    if len(set(grouped)) != len(grouped) or len(set(grain)) != len(grain):
        raise UnsupportedIntent("duplicate grain")
    if has_aggregate:
        if grain != grouped or not nonaggregate <= set(grouped):
            raise UnsupportedIntent("aggregation grain disagrees with projection")
    elif grain or grouped:
        raise UnsupportedIntent("unsupported non-aggregate grouping")
    sql_group = [field(name) for name in grouped]

    conditions = plan["where"]
    if not isinstance(conditions, list) or len(conditions) > 12:
        raise UnsupportedIntent("invalid filters")
    sql_where: list[str] = []
    for condition in conditions:
        if not isinstance(condition, dict) or set(condition) != {"column", "op", "value"}:
            raise UnsupportedIntent("invalid predicate")
        operator = condition["op"]
        if not isinstance(operator, str) or operator not in _OPERATORS:
            raise UnsupportedIntent("unsupported predicate")
        sql_where.append(f"{field(condition['column'])} {operator} {_literal(condition['value'])}")

    sorting = plan["order_by"]
    if not isinstance(sorting, list) or len(sorting) > 6:
        raise UnsupportedIntent("invalid ordering")
    sql_sort: list[str] = []
    for item in sorting:
        if not isinstance(item, dict) or set(item) not in ({"alias", "direction"}, {"column", "direction"}):
            raise UnsupportedIntent("invalid ordering term")
        direction = item["direction"]
        if direction not in ("ASC", "DESC"):
            raise UnsupportedIntent("invalid direction")
        if "alias" in item:
            if item["alias"] not in aliases:
                raise UnsupportedIntent("unselected sort alias")
            expression = _identifier(item["alias"])
        else:
            if has_aggregate and item["column"] not in grouped:
                raise UnsupportedIntent("ordering by ungrouped column")
            expression = field(item["column"])
        sql_sort.append(f"{expression} {direction}")

    limit = plan["limit"]
    if limit is not None and (type(limit) is not int or not 1 <= limit <= 1000):
        raise UnsupportedIntent("invalid limit")
    parts = ["SELECT " + ", ".join(sql_projections), "FROM " + _identifier(root), *sql_joins]
    if sql_where:
        parts.append("WHERE " + " AND ".join(sql_where))
    if sql_group:
        parts.append("GROUP BY " + ", ".join(sql_group))
    if sql_sort:
        parts.append("ORDER BY " + ", ".join(sql_sort))
    if limit is not None:
        parts.append("LIMIT " + str(limit))
    return " ".join(parts)


def draft_typed_query_sql(
    question: str,
    schema_context: str,
    catalog: DatabaseCatalog,
    provider: object,
    *,
    evidence: str | None = None,
) -> str | None:
    """One same-Qwen structured intent call; never accept model-generated raw SQL.

    Experimental caller selects which Qwen provider/digest to use and retains
    original Guarded output on None. This has no production entrypoint.
    """
    system = (
        "Translate the question into a single JSON query-intent object using only "
        "the supplied physical SQLite schema. Return JSON ONLY, never SQL or prose. "
        "Use exactly eight keys: from, select, joins, where, group_by, grain, "
        "order_by, limit. from is one physical table string. select is a nonempty "
        "list of {op,column,optional as,optional distinct} with op column/count/"
        "sum/avg/min/max and physical qualified table.column or COUNT(*). joins "
        "is a list of {table,on:[table.column,table.column]} along DECLARED foreign "
        "keys only. where is a list of {column,op,value} primitive comparisons; "
        "group_by and grain are physical qualified-column lists, identical for "
        "aggregations; otherwise both empty. order_by is a list of {alias,direction} "
        "or {column,direction}; direction ASC/DESC. limit is integer 1..1000 or null. "
        "Do not guess unsupported semantics, physical identifiers or join edges. "
        "For a query requiring nested SELECT, arithmetic expressions, HAVING, OR, "
        "window functions, or ambiguous grain, output {} to abstain."
    )
    user = f"Question:\n{question}\n\nPhysical schema:\n{schema_context}"
    if evidence:
        user += f"\n\nExternal evidence:\n{evidence}"
    response = provider.complete_text(system=system, user=user)
    try:
        return compile_typed_query(json.loads(response), catalog)
    except (ValueError, TypeError):
        return None
