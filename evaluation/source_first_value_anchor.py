"""Positive exact-entity anchors from real SQLite data before schema selection.

This is a CPU-only source gate, NOT an LLM, reviewer, fuzzy guess, or SQL
selector. A probe can find true evidence; failed/interrupted/absent probes do
not prove a table cannot contain an entity. No benchmark reference SQL used.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3
from urllib.parse import quote

from semantic_sql.catalog import DatabaseCatalog


@dataclass(frozen=True)
class ValueAnchor:
    table: str
    column: str
    value: str


@dataclass(frozen=True)
class ValueAnchorProbe:
    hits: tuple[ValueAnchor, ...]
    outside_schema: tuple[str, ...]
    probes: int
    interrupted: int
    ambiguous: bool
    empty_probes_are_not_negative_evidence: bool = True


def _quoted_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _literals(question: str) -> tuple[str, ...]:
    # Single quotes must be independent quotation marks, not apostrophes
    # within possessives such as "driver's". Numeric IDs are not entity names.
    pairs = re.findall(
        r'"([^"\n]{3,96})"|(?<![A-Za-z0-9])\'([^\'\n]{3,96})\'(?![A-Za-z0-9])',
        question,
    )
    values = (x.strip() for pair in pairs for x in pair if x.strip())
    return tuple(dict.fromkeys(x for x in values if not x.isdigit()))[:2]

def observed_value_anchors(
    database: str | Path,
    catalog: DatabaseCatalog,
    question: str,
    *,
    selected_tables: set[str],
    max_probes: int = 100,
    max_tables: int = 40,
    max_literal_chars: int = 96,
) -> ValueAnchorProbe:
    if min(max_probes, max_tables, max_literal_chars) < 0:
        raise ValueError("budgets must be nonnegative")
    if max_probes == 0 or max_tables == 0:
        return ValueAnchorProbe((), (), 0, 0, False)
    literals = [v for v in _literals(question) if 3 <= len(v) <= max_literal_chars]
    if not literals:
        return ValueAnchorProbe((), (), 0, 0, False)
    tables = sorted(catalog.tables)[:max_tables]
    column_candidates = []
    uri = "file:" + quote(str(Path(database).resolve())) + "?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=5) as connection:
        connection.execute("PRAGMA query_only=ON")
        for table in tables:
            # Do not trust schema-only guesses: search *all* real TEXT columns,
            # especially the value-bearing name/title/translation fields.
            rows = connection.execute(f"PRAGMA table_info({_quoted_ident(table)})").fetchall()
            for column in rows:
                name, affinity = column[1], str(column[2]).upper()
                if affinity and not any(t in affinity for t in ("TEXT", "CHAR", "CLOB", "VARCHAR")):
                    continue
                priority = 0 if re.search(r"name|title|translation|label|description",name,re.I) else 1
                column_candidates.append((priority, table, name))
        column_candidates.sort()
        hits: list[ValueAnchor] = []
        probes = interrupted = 0
        for value in literals:
            for _priority, table, column in column_candidates:
                if probes >= max_probes:
                    break
                probes += 1
                progress = [0]
                def interrupt() -> int:
                    progress[0] += 1
                    return int(progress[0] > 150)
                connection.set_progress_handler(interrupt, 1000)
                try:
                    exists = connection.execute(
                        f"SELECT 1 FROM {_quoted_ident(table)} "
                        f"WHERE {_quoted_ident(column)} = ? COLLATE NOCASE LIMIT 1",
                        (value,),
                    ).fetchone()
                    if exists:
                        hits.append(ValueAnchor(table,column,value))
                except sqlite3.DatabaseError:
                    interrupted += 1
                finally:
                    connection.set_progress_handler(None,0)
            if probes >= max_probes:
                break
    anchors = tuple(dict.fromkeys(hits))
    missing = tuple(sorted({f"{hit.table}.{hit.column}" for hit in anchors
                            if hit.table not in selected_tables}))
    by_value: dict[str, set[tuple[str,str]]] = {}
    for hit in anchors:
        by_value.setdefault(hit.value,set()).add((hit.table,hit.column))
    ambiguous = any(len(locations)>1 for locations in by_value.values())
    return ValueAnchorProbe(
        hits=anchors,outside_schema=missing,probes=probes,
        interrupted=interrupted,ambiguous=ambiguous,
    )
