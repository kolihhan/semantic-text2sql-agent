from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.grounding import GroundingPack


_SQL_KEYWORDS = {
    "AS", "ON", "WHERE", "GROUP", "ORDER", "BY", "LIMIT", "OFFSET", "JOIN",
    "LEFT", "RIGHT", "INNER", "OUTER", "FULL", "CROSS", "UNION", "HAVING",
}


@dataclass(frozen=True)
class GoldSchemaUsage:
    tables: frozenset[str]
    columns: frozenset[str]
    fk_edges: frozenset[tuple[str, str, str, str]]
    string_literals: frozenset[str]


def _strip_string_literals(sql: str) -> str:
    return re.sub(r"'(?:''|[^'])*'", "''", sql)


def _canonical_edge(
    edge: tuple[str, str, str, str],
) -> tuple[tuple[str, str], tuple[str, str]]:
    source, source_col, target, target_col = edge
    left = (source, source_col)
    right = (target, target_col)
    ordered = sorted((left, right))
    return ordered[0], ordered[1]


def extract_gold_schema_usage(sql: str, catalog: DatabaseCatalog) -> GoldSchemaUsage:
    structural = _strip_string_literals(sql)
    aliases: dict[str, str] = {}
    tables: set[str] = set()

    for table in catalog.tables:
        pattern = re.compile(
            rf"\b(?:FROM|JOIN)\s+[`\"\[]?{re.escape(table)}[`\"\]]?"
            rf"(?:\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*))?",
            flags=re.I,
        )
        for match in pattern.finditer(structural):
            tables.add(table)
            alias = match.group(1)
            if alias and alias.upper() not in _SQL_KEYWORDS:
                aliases[alias] = table
            aliases[table] = table

    columns: set[str] = set()
    qualified_refs = re.finditer(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*[`\"\[]?([A-Za-z_][A-Za-z0-9_]*)[`\"\]]?",
        structural,
    )
    for match in qualified_refs:
        qualifier, column = match.group(1), match.group(2)
        table = aliases.get(qualifier, qualifier if qualifier in catalog.tables else None)
        if table in catalog.tables and column in catalog.tables[table].columns:
            tables.add(table)
            columns.add(f"{table}.{column}")

    referenced = tables or set(catalog.tables)
    unique_columns: dict[str, str | None] = {}
    for table in referenced:
        for column in catalog.tables[table].columns:
            if column not in unique_columns:
                unique_columns[column] = table
            else:
                unique_columns[column] = None
    for token in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", structural):
        table = unique_columns.get(token)
        if table:
            columns.add(f"{table}.{token}")

    catalog_edges = {
        _canonical_edge(edge): edge
        for edge in catalog.foreign_keys
    }
    fk_edges: set[tuple[str, str, str, str]] = set()
    join_equalities = re.finditer(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)"
        r"\s*=\s*"
        r"([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b",
        structural,
    )
    for match in join_equalities:
        left_alias, left_col, right_alias, right_col = match.groups()
        left_table = aliases.get(
            left_alias,
            left_alias if left_alias in catalog.tables else None,
        )
        right_table = aliases.get(
            right_alias,
            right_alias if right_alias in catalog.tables else None,
        )
        if not left_table or not right_table:
            continue
        canonical = tuple(sorted(((left_table, left_col), (right_table, right_col))))
        edge = catalog_edges.get(canonical)
        if edge is not None:
            fk_edges.add(edge)

    literals = {
        value.replace("''", "'").strip()
        for value in re.findall(r"'((?:''|[^'])*)'", sql)
        if value.replace("''", "'").strip()
    }

    return GoldSchemaUsage(
        tables=frozenset(tables),
        columns=frozenset(columns),
        fk_edges=frozenset(fk_edges),
        string_literals=frozenset(literals),
    )


def _precision_recall(
    predicted: set[str],
    gold: set[str],
) -> tuple[float | None, float | None]:
    precision = None if not predicted else len(predicted & gold) / len(predicted)
    recall = None if not gold else len(predicted & gold) / len(gold)
    return precision, recall


def _edge_recall(
    predicted: Iterable[tuple[str, str, str, str]],
    gold: Iterable[tuple[str, str, str, str]],
) -> float | None:
    predicted_keys = {_canonical_edge(edge) for edge in predicted}
    gold_keys = {_canonical_edge(edge) for edge in gold}
    if not gold_keys:
        return None
    return len(predicted_keys & gold_keys) / len(gold_keys)


def _value_literal_recall(
    pack: GroundingPack,
    literals: Iterable[str],
) -> float | None:
    gold = {value.casefold().strip() for value in literals if value.strip()}
    if not gold:
        return None
    predicted = {hit.value.casefold().strip() for hit in pack.value_hits}
    return len(predicted & gold) / len(gold)


def score_grounding_pack(
    pack: GroundingPack,
    *,
    gold_sql: str,
    catalog: DatabaseCatalog,
) -> dict[str, float | int | None]:
    usage = extract_gold_schema_usage(gold_sql, catalog)
    selected_tables = set(pack.selected_tables)
    context_columns = set(pack.context_columns)
    anchor_columns = set(pack.anchor_columns)

    table_precision, table_recall = _precision_recall(
        selected_tables,
        set(usage.tables),
    )
    column_precision, column_recall = _precision_recall(
        context_columns,
        set(usage.columns),
    )
    anchor_precision, anchor_recall = _precision_recall(
        anchor_columns,
        set(usage.columns),
    )

    return {
        "gold_table_count": len(usage.tables),
        "gold_column_count": len(usage.columns),
        "gold_fk_edge_count": len(usage.fk_edges),
        "gold_string_literal_count": len(usage.string_literals),
        "table_precision": table_precision,
        "table_recall": table_recall,
        "column_precision": column_precision,
        "column_recall": column_recall,
        "anchor_column_precision": anchor_precision,
        "anchor_column_recall": anchor_recall,
        "fk_bridge_recall": _edge_recall(pack.join_edges, usage.fk_edges),
        "value_grounding_recall": _value_literal_recall(
            pack,
            usage.string_literals,
        ),
    }


def score_selected_tables(
    selected_tables: Iterable[str],
    *,
    gold_sql: str,
    catalog: DatabaseCatalog,
) -> dict[str, float | int | None]:
    """Score the existing lexical selector on the same observable schema surface."""
    usage = extract_gold_schema_usage(gold_sql, catalog)
    selected = set(selected_tables)
    context_columns = {
        f"{table}.{column}"
        for table in selected
        if table in catalog.tables
        for column in catalog.tables[table].columns
    }
    join_edges = tuple(
        edge
        for edge in catalog.foreign_keys
        if edge[0] in selected and edge[2] in selected
    )
    table_precision, table_recall = _precision_recall(
        selected,
        set(usage.tables),
    )
    column_precision, column_recall = _precision_recall(
        context_columns,
        set(usage.columns),
    )
    return {
        "gold_table_count": len(usage.tables),
        "gold_column_count": len(usage.columns),
        "gold_fk_edge_count": len(usage.fk_edges),
        "gold_string_literal_count": len(usage.string_literals),
        "table_precision": table_precision,
        "table_recall": table_recall,
        "column_precision": column_precision,
        "column_recall": column_recall,
        "fk_bridge_recall": _edge_recall(join_edges, usage.fk_edges),
        "value_grounding_recall": 0.0 if usage.string_literals else None,
    }
