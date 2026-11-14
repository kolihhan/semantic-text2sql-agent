"""Deterministic SQLite M-Schema-like representation for XiYan ablation.

Uses only selected physical tables, SQLite PRAGMA metadata, original column
descriptions and (optionally) bounded samples from the source DB.
No training, reference SQL or labels are used in constructing the schema.
The official M-Schema library depends on a broader stack; this deliberately
reuses our catalog and sqlite3 stdlib, and is named M-Schema-like, not official.
"""
from __future__ import annotations

import sqlite3

from evaluation.schema_context import _tokens
from semantic_sql.catalog import DatabaseCatalog


def _descriptions(lexical_context: str) -> dict[tuple[str, str], str]:
    out = {}
    for line in lexical_context.splitlines():
        if not line.startswith("- ") or ": " not in line:
            continue
        key, _, description = line[2:].partition(": ")
        if "." not in key:
            continue
        table, col = key.rsplit(".", 1)
        if table and col and description:
            out[(table, col)] = " ".join(description.split())[:90]
    return out


def to_mschema_like(
    catalog: DatabaseCatalog, *, db_id: str, selected_tables: list[str],
    question: str, lexical_context: str, with_examples: bool,
) -> str:
    if not selected_tables or len(selected_tables) != len(set(selected_tables)):
        raise ValueError("empty or repeated selected tables")
    if not set(selected_tables).issubset(catalog.tables):
        raise ValueError("schema selection contains unknown physical table")
    descriptions = _descriptions(lexical_context)
    question_tokens = _tokens(question)
    lines = [f"〖DB_ID〗 {db_id}", "〖Schema〗"]
    example_budget = 12
    with sqlite3.connect(catalog.database) as con:
        for table in sorted(selected_tables):
            escaped = table.replace('"', '""')
            columns = con.execute(f'PRAGMA table_info("{escaped}")').fetchall()
            if not columns or tuple(row[1] for row in columns) != catalog.tables[table].columns:
                raise ValueError("physical table metadata changed during schema rendering")
            lines.extend([f"# Table: {table}", "["])
            fields = []
            for _, name, dbtype, _nullable, _default, pk in columns:
                coltype = (str(dbtype or "UNKNOWN").split("(", 1)[0] or "UNKNOWN").upper()
                desc = descriptions.get((table, name), "")
                chunks = [f"{name}:{coltype}"]
                if desc:
                    chunks.append(desc)
                if pk:
                    chunks.append("Primary Key")
                if with_examples and example_budget > 0 and (
                    _tokens(name) | _tokens(desc)
                ) & question_tokens:
                    identifier = name.replace('"', '""')
                    samples = [row[0] for row in con.execute(
                        f'SELECT DISTINCT "{identifier}" FROM "{escaped}" '
                        f'WHERE "{identifier}" IS NOT NULL LIMIT 2'
                    )]
                    usable = [" ".join(str(v).split()) for v in samples if v is not None]
                    usable = [v for v in usable if v and len(v) <= 50][:2]
                    if usable:
                        chunks.append("Examples: [" + ", ".join(usable) + "]")
                        example_budget -= 1
                fields.append("(" + ", ".join(chunks) + ")")
            lines.append(",\n".join(fields))
            lines.append("]")
    subset = set(selected_tables)
    fk = [
        f"{src}.{col}={dst}.{target}"
        for src, col, dst, target in catalog.foreign_keys
        if src in subset and dst in subset
    ]
    if fk:
        lines.append("〖Foreign keys〗")
        lines.extend(sorted(set(fk)))
    return "\n".join(lines)
