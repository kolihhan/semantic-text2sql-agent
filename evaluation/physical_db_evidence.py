"""Read-only physical relational and entity evidence, not another model verifier.

Only positive observations are emitted. FK cardinality claims require a physical
primary/unique parent key; otherwise label the relation as uncertain. Never use
reference SQL or sample model answers to assemble evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3
from urllib.parse import quote

from semantic_sql.catalog import DatabaseCatalog


@dataclass(frozen=True)
class PhysicalFacts:
    text: str
    value_matches: int
    probes: int
    fk_lines: int
    pk_lines: int


def _ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _literal_candidates(question: str) -> tuple[str, ...]:
    found = re.findall(r'"([^"\n]{2,72})"|\'([^\'\n]{2,72})\'', question)
    return tuple(dict.fromkeys(x.strip() for pair in found for x in pair if x.strip()))[:2]


def _primary_keys(db: sqlite3.Connection, table: str) -> tuple[str, ...]:
    rows = db.execute(f"PRAGMA table_info({_ident(table)})").fetchall()
    return tuple(r[1] for r in sorted(rows, key=lambda row: row[5]) if r[5] > 0)


def _text_columns(db: sqlite3.Connection, table: str) -> tuple[str, ...]:
    info = db.execute(f"PRAGMA table_info({_ident(table)})").fetchall()
    return tuple(row[1] for row in info
                 if any(kind in str(row[2]).upper() for kind in ("TEXT", "CHAR", "CLOB", "VARCHAR"))
                 or str(row[2]).strip() == "")


def verified_db_facts(
    database: str | Path,
    catalog: DatabaseCatalog,
    selected_tables: set[str],
    question: str,
    *,
    max_tables: int = 8,
    max_scans: int = 24,
) -> PhysicalFacts:
    """Return bounded proven facts; zero confidence from unobserved/failed probes."""
    if max_tables < 0 or max_scans < 0:
        raise ValueError("budgets must not be negative")
    selected = tuple(sorted(set(catalog.tables) & set(selected_tables)))[:max_tables]
    lines = []
    probes = matches = fk_count = pk_count = 0
    uri = "file:" + quote(str(Path(database).resolve())) + "?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=5.0) as db:
        db.execute("PRAGMA query_only=ON")
        keys = {table: _primary_keys(db, table) for table in selected}
        for table in selected:
            if keys[table]:
                pk_count += 1
                lines.append(f"Verified PRIMARY KEY {table}: {', '.join(keys[table])}")
        for src, sc, target, tc in catalog.foreign_keys:
            if src not in selected or target not in selected:
                continue
            target_unique = keys.get(target) == (tc,)
            note = ("at most one parent per child; not a proven 1:1"
                    if target_unique else "not a proven 1:1 or many-to-one relation")
            fk_count += 1
            lines.append(f"Declared FK {src}.{sc} -> {target}.{tc}; {note}")
        literals = _literal_candidates(question)
        if literals and max_scans:
            for literal in literals:
                for table in selected:
                    for column in _text_columns(db, table):
                        if probes >= max_scans:
                            break
                        probes += 1
                        # Bound full scans on large public BIRD tables. No negative
                        # conclusions follow from a timeout or scan interruption.
                        counter = [0]
                        def watchdog():
                            counter[0] += 1
                            return int(counter[0] > 100)
                        db.set_progress_handler(watchdog, 1000)
                        try:
                            cursor = db.execute(
                                f"SELECT 1 FROM {_ident(table)} "
                                f"WHERE {_ident(column)} = ? COLLATE NOCASE LIMIT 1",
                                (literal,),
                            )
                            found = cursor.fetchone() is not None
                        except sqlite3.DatabaseError:
                            found = False
                        finally:
                            db.set_progress_handler(None, 0)
                        if found:
                            matches += 1
                            lines.append(f"Verified exact value {table}.{column} = {literal!r}")
                    if probes >= max_scans:
                        break
                if probes >= max_scans:
                    break
    return PhysicalFacts(
        text=("\n".join(lines) if lines else ""),
        value_matches=matches, probes=probes, fk_lines=fk_count, pk_lines=pk_count,
    )
