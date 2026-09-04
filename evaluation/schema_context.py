from __future__ import annotations

import csv
from pathlib import Path
import re
from hashlib import sha256

from semantic_sql.catalog import DatabaseCatalog


def _description_rows(database_root: Path, db_id: str) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    root = database_root / db_id / "database_description"
    if not root.is_dir():
        return rows

    for path in sorted(root.glob("*.csv")):
        table = path.stem
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    physical = str(
                        row.get("original_column_name")
                        or row.get("column_name")
                        or row.get("column")
                        or ""
                    ).strip()
                    if not physical:
                        continue
                    description = str(row.get("column_description") or "").strip()
                    display_name = str(row.get("column_name") or physical).strip()
                    rows.append((table, physical, description or display_name))
        except (OSError, csv.Error, UnicodeError):
            continue
    return rows


_STOPWORDS = {
    "a", "an", "and", "are", "be", "by", "for", "from", "has", "have", "how",
    "in", "is", "of", "on", "or", "the", "to", "what", "which", "who", "with",
}


def _tokens(value: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", value.casefold()))
    return {word.rstrip("s") for word in words if word not in _STOPWORDS and len(word.rstrip("s")) > 1}


def select_schema_context(
    catalog: DatabaseCatalog, database_root: Path, db_id: str, question: str,
) -> tuple[str, dict[str, object]]:
    """Select a compact lexical schema context; fall back closed to the full schema."""
    query_tokens = _tokens(question)
    descriptions = _description_rows(database_root, db_id)
    by_column = {(table, column): description for table, column, description in descriptions}
    scores: dict[str, int] = {}
    matched: dict[str, list[str]] = {}
    for table, info in catalog.tables.items():
        table_tokens = _tokens(table)
        column_tokens = set().union(*(_tokens(column) for column in info.columns)) if info.columns else set()
        description_tokens = set().union(*(
            _tokens(description) for (desc_table, column), description in by_column.items() if desc_table == table
        )) if info.columns else set()
        hits = sorted(query_tokens & (table_tokens | column_tokens | description_tokens))
        if hits:
            scores[table] = len(query_tokens & table_tokens) * 3 + len(query_tokens & column_tokens) * 2 + len(query_tokens & description_tokens)
            matched[table] = hits
    selected = {table for table, score in scores.items() if score > 0}
    # Include one-hop FK neighbors, preserving joinability of a grounded table set.
    if selected:
        for source, _source_col, target, _target_col in catalog.foreign_keys:
            if source in selected or target in selected:
                selected.update(table for table in (source, target) if table in catalog.tables)
    uncertain = not query_tokens or not selected or max(scores.values(), default=0) < 2
    if uncertain:
        selected = set(catalog.tables)
    schema_lines = []
    for table in sorted(selected):
        schema_lines.append(f"{table}({', '.join(catalog.tables[table].columns)})")
    for source_table, source_col, target_table, target_col in catalog.foreign_keys:
        if source_table in selected and target_table in selected:
            schema_lines.append(f"FK {source_table}.{source_col} -> {target_table}.{target_col}")
    selected_descriptions = [
        (table, column, description) for table, column, description in descriptions
        if table in selected and column in catalog.tables[table].columns
    ]
    if selected_descriptions:
        schema_lines.append("Column descriptions:")
        schema_lines.extend(f"- {table}.{column}: {description}" for table, column, description in selected_descriptions)
    context = "\n".join(("Physical SQLite schema:", *schema_lines))
    diagnostics = {
        "selector": "deterministic_lexical_v1",
        "fallback_full_schema": uncertain,
        "query_tokens": sorted(query_tokens),
        "selected_tables": sorted(selected),
        "table_scores": {table: scores.get(table, 0) for table in sorted(selected)},
        "matched_tokens": {table: matched.get(table, []) for table in sorted(selected)},
        "schema_context_sha256": sha256(context.encode("utf-8")).hexdigest(),
    }
    return context, diagnostics


def build_schema_context(
    catalog: DatabaseCatalog, database_root: Path, db_id: str, question: str | None = None,
) -> str:
    if question is not None:
        return select_schema_context(catalog, database_root, db_id, question)[0]
    lines = ["Physical SQLite schema:", catalog.schema_text()]
    descriptions = _description_rows(database_root, db_id)
    if descriptions:
        lines.append("Column descriptions:")
        for table, column, description in descriptions:
            if table in catalog.tables and column in catalog.tables[table].columns:
                lines.append(f"- {table}.{column}: {description}")
    return "\n".join(lines)
