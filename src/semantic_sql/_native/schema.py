# MIT-licensed upstream-derived component; copyright notice in this directory's LICENSE.
"""Compact, cached SQLite schema introspection for any database, not just Chinook."""

from __future__ import annotations

import re
import sqlite3
from functools import lru_cache
from pathlib import Path

DEFAULT_DB_PATH = Path("data/Chinook.db")

# Any TEXT-affinity column with at most this many distinct values gets sample values shown
# in the prompt (illustrative examples, e.g. genre/media-type names) — cheap and high value
# for reducing case/spelling mismatches in generated WHERE clauses.
LOW_CARDINALITY_THRESHOLD = 50
MAX_SAMPLE_VALUES = 5
# Only bother sampling columns on tables at least this small a fraction of the introspection
# budget cares about; skip sampling on very wide tables to bound prompt-build latency.
MAX_SAMPLED_COLUMNS_PER_TABLE = 10
_DATE_LIKE = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}:\d{2})?$")
_BARE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
QUOTED_SCHEMA_HEADER = "SQLite schema (names in backticks must be written with the backticks):"


def _render_name(name: str, quote: bool) -> str:
    """Backtick names SQLite can't read bare (``T-BIL``, ``aCL IgM``) when quoting is on,
    so the safe spelling is what the model sees and copies."""
    if not quote or _BARE_IDENTIFIER.match(name):
        return name
    return "`" + name.replace("`", "``") + "`"


def _db_key(db_path: str | Path) -> str:
    return str(Path(db_path).resolve())


@lru_cache(maxsize=32)
def _metadata(db_key: str) -> dict[str, tuple[tuple[str, str, bool], ...]]:
    connection = sqlite3.connect(f"file:{db_key}?mode=ro", uri=True)
    try:
        tables = connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        return {
            str(table[0]): tuple(
                (str(row[1]), str(row[2] or "ANY"), bool(row[5]))
                for row in connection.execute(f'PRAGMA table_info("{table[0]}")')
            )
            for table in tables
        }
    finally:
        connection.close()


def get_identifier_allowlist(
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, frozenset[str]]:
    """Return case-folded table-to-column identifiers for AST validation."""
    return {
        table.casefold(): frozenset(column.casefold() for column, _kind, _pk in columns)
        for table, columns in _metadata(_db_key(db_path)).items()
    }


def _sample_column(
    connection: sqlite3.Connection, table: str, column: str
) -> tuple[str, list[str]] | None:
    """Return (\"cardinality\" | \"date\", examples) for one column, or None if not worth showing.

    A single bounded DISTINCT query decides both cases: if it returns at most
    ``LOW_CARDINALITY_THRESHOLD`` rows, the column is a low-cardinality categorical (illustrative
    values reduce case/spelling mismatches, e.g. 'Rock' vs 'rock'); otherwise, if the sampled
    values look like dates, a second bounded query samples distinct years instead.
    """
    rows = connection.execute(
        f'SELECT DISTINCT "{column}" FROM "{table}" '
        f'WHERE "{column}" IS NOT NULL ORDER BY 1 LIMIT {LOW_CARDINALITY_THRESHOLD + 1}'
    ).fetchall()
    if not rows:
        return None
    if len(rows) <= LOW_CARDINALITY_THRESHOLD:
        return "cardinality", [repr(row[0]) for row in rows[:MAX_SAMPLE_VALUES]]
    if isinstance(rows[0][0], str) and _DATE_LIKE.match(rows[0][0]):
        years = connection.execute(
            f'SELECT DISTINCT strftime(\'%Y\', "{column}") FROM "{table}" '
            f'WHERE "{column}" IS NOT NULL ORDER BY 1 LIMIT {MAX_SAMPLE_VALUES}'
        ).fetchall()
        return "date", [repr(row[0]) for row in years]
    return None


@lru_cache(maxsize=32)
def _foreign_key_edges(db_key: str) -> tuple[tuple[str, str, str, str], ...]:
    """Return sorted, de-duplicated (table, column, parent_table, parent_column) edges.

    SQLite leaves the parent column NULL for ``REFERENCES parent`` without a column list,
    meaning the parent's primary key; resolve it (by PK position for composite keys) instead
    of rendering ``parent.None``. Parent tables matched case-insensitively, as SQLite does.
    """
    metadata = _metadata(db_key)
    tables = {name.casefold(): name for name in metadata}
    connection = sqlite3.connect(f"file:{db_key}?mode=ro", uri=True)
    try:
        edges: set[tuple[str, str, str, str]] = set()
        for table in metadata:
            rows = connection.execute(f'PRAGMA foreign_key_list("{table}")').fetchall()
            for row in rows:
                parent, column, parent_column, seq = str(row[2]), str(row[3]), row[4], row[1]
                parent = tables.get(parent.casefold(), parent)
                if parent_column is None:
                    pk = [
                        str(info[1])
                        for info in sorted(
                            connection.execute(f'PRAGMA table_info("{parent}")').fetchall(),
                            key=lambda info: info[5],
                        )
                        if info[5]
                    ]
                    if seq >= len(pk):
                        continue  # dangling reference; nothing trustworthy to show
                    parent_column = pk[seq]
                edges.add((table, column, parent, str(parent_column)))
        return tuple(sorted(edges))
    finally:
        connection.close()


@lru_cache(maxsize=64)
def _schema_text(db_key: str, quote: bool = False) -> str:
    metadata = _metadata(db_key)
    connection = sqlite3.connect(f"file:{db_key}?mode=ro", uri=True)

    def q(name: str) -> str:
        return _render_name(name, quote)

    try:
        needs_quoting = quote and any(
            not _BARE_IDENTIFIER.match(name)
            for table, columns in metadata.items()
            for name in (table, *(column for column, _kind, _pk in columns))
        )
        lines = [QUOTED_SCHEMA_HEADER if needs_quoting else "SQLite schema:"]
        for table, columns in metadata.items():
            rendered = ", ".join(f"{q(name)} {kind}" for name, kind, _pk in columns)
            lines.append(f"- {q(table)}({rendered})")
            # Skip primary keys: always unique, so sampling them is wasted work.
            candidates = [name for name, _kind, is_pk in columns if not is_pk]
            for column in candidates[:MAX_SAMPLED_COLUMNS_PER_TABLE]:
                sampled = _sample_column(connection, table, column)
                if sampled is None:
                    continue
                kind, examples = sampled
                label = f"sample {q(column)}" + ("" if kind == "cardinality" else " year")
                lines.append(f"  {label}: {', '.join(examples)}")

        foreign_keys = _foreign_key_edges(db_key)
        if foreign_keys:
            lines.append(
                "Foreign keys: "
                + "; ".join(
                    f"{q(src)}.{q(col)} -> {q(dst)}.{q(ref)}"
                    for src, col, dst, ref in foreign_keys
                )
            )
        return "\n".join(lines)
    finally:
        connection.close()


def get_schema(db_path: str | Path = DEFAULT_DB_PATH, *, quote_identifiers: bool = False) -> str:
    """Return a compact schema string, cached by resolved database path and quoting."""
    return _schema_text(_db_key(db_path), quote_identifiers)


def table_count(db_path: str | Path = DEFAULT_DB_PATH) -> int:
    """Return the number of user tables."""
    return len(_metadata(_db_key(db_path)))


def get_foreign_keys(
    db_path: str | Path = DEFAULT_DB_PATH,
) -> tuple[tuple[str, str, str, str], ...]:
    """Return resolved foreign-key edges as rendered in the schema block."""
    return _foreign_key_edges(_db_key(db_path))


def get_table_columns(db_path: str | Path = DEFAULT_DB_PATH) -> dict[str, list[str]]:
    """Original-case table -> column names (for "did you mean" repair hints)."""
    return {
        table: [name for name, _kind, _pk in columns]
        for table, columns in _metadata(_db_key(db_path)).items()
    }
