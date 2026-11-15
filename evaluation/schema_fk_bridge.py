"""Deterministic schema-only FK join paths for disjoint query anchor tables.

No reference SQL, no database row inspection, no additional model call.
Original lexical schema and its selected-table set remain intact.
"""
from __future__ import annotations

from collections import deque
from itertools import combinations
import re

from semantic_sql.catalog import DatabaseCatalog
from evaluation.schema_context import select_schema_context

_COMMON = {
    "a", "all", "an", "and", "are", "by", "each", "every", "find", "for", "from",
    "has", "have", "how", "in", "is", "many", "most", "name", "number", "of",
    "on", "or", "show", "the", "their", "to", "total", "what", "which", "who",
    "with", "without", "get", "list", "give",
}


def terms(value: str) -> set[str]:
    parts = re.findall(r"[a-z0-9]+", value.casefold().replace("_", " "))
    return {word.removesuffix("s") for word in parts if len(word) >= 3 and word not in _COMMON}


def _graph(catalog: DatabaseCatalog) -> dict[str, list[tuple[str, str]]]:
    graph: dict[str, list[tuple[str, str]]] = {t: [] for t in catalog.tables}
    for child, child_key, parent, parent_key in catalog.foreign_keys:
        if child not in graph or parent not in graph:
            continue
        expr = f"{child}.{child_key} = {parent}.{parent_key}"
        graph[child].append((parent, expr))
        graph[parent].append((child, expr))
    for table in graph:
        graph[table].sort()
    return graph


def shortest_path(catalog: DatabaseCatalog, first: str, last: str, max_edges: int = 5):
    if first == last or first not in catalog.tables or last not in catalog.tables:
        return None
    graph = _graph(catalog)
    queue = deque([(first, (first,), ())])
    visited = {first}
    while queue:
        here, tables, edges = queue.popleft()
        if len(edges) >= max_edges:
            continue
        for neighbor, expression in graph[here]:
            if neighbor in visited:
                continue
            new_tables, new_edges = tables + (neighbor,), edges + (expression,)
            if neighbor == last:
                return new_tables, new_edges
            visited.add(neighbor)
            queue.append((neighbor, new_tables, new_edges))
    return None


def augment_fk_paths(
    catalog: DatabaseCatalog, root, db_id: str, question: str, *,
    max_routes: int = 3
) -> tuple[str, str, dict]:
    """Return untouched baseline, enriched schema, and source-only diagnostics."""
    base, diagnostics = select_schema_context(catalog, root, db_id, question)
    question_terms = terms(question)
    # Only table-name mentions can anchor a route; generic Name/ID columns
    # would be high-frequency false anchors and are intentionally excluded.
    anchors = [
        t for t in catalog.tables
        if terms(t) & question_terms
    ]
    candidates = []
    for first, last in combinations(sorted(anchors), 2):
        route = shortest_path(catalog, first, last)
        if route is None or not 2 <= len(route[1]) <= 5:
            continue
        candidate = {
            "endpoints": (first, last), "tables": route[0],
            "edges": route[1],
            "missing_bridge_tables": tuple(
                t for t in route[0][1:-1]
                if t not in diagnostics["selected_tables"]
            ),
        }
        candidates.append(candidate)
    # Prefer paths that restore missing bridge tables, then shortest simple
    # joint paths. Stable across machines and independent of reference SQL.
    candidates.sort(key=lambda c: (
        -len(c["missing_bridge_tables"]), len(c["edges"]),
        c["endpoints"], c["edges"]
    ))
    chosen = candidates[:max_routes]
    if not chosen:
        return base, base, {
            "intervention": "no_reachable_nonadjacent_table_mentions",
            "anchors": sorted(anchors), "routes": [],
            "missing_bridge_table_count": 0,
        }
    lines = [
        "FK connection paths between table names mentioned in the question "
        "(literal database constraints; use only if relevant):"
    ]
    new_tables = set()
    for candidate in chosen:
        lines.append(
            "- " + candidate["endpoints"][0] + " to " +
            candidate["endpoints"][1] + ": " + " ; ".join(candidate["edges"])
        )
        new_tables.update(candidate["missing_bridge_tables"])
    if new_tables:
        lines.append("Physical bridge tables (previously outside one-hop schema selection):")
        lines.extend(
            f"- {table}({', '.join(catalog.tables[table].columns)})"
            for table in sorted(new_tables)
        )
    enriched = base + "\n" + "\n".join(lines)
    return base, enriched, {
        "intervention": "schema_fk_routes",
        "anchors": sorted(anchors),
        "missing_bridge_table_count": len(new_tables),
        "routes": [{
            "endpoints": list(c["endpoints"]),
            "tables": list(c["tables"]),
            "edges": list(c["edges"]),
            "missing_bridge_tables": list(c["missing_bridge_tables"]),
        } for c in chosen],
        "baseline_schema_bytes": len(base),
        "augmented_schema_bytes": len(enriched),
    }
