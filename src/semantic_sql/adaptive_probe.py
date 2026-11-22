"""Gold-blind, opt-in BIRD semantic-risk probes for the existing LangGraph.

Risk flags are *hypotheses* (not proof of wrong SQL). SQLite is opened read-only
with query_only; probes may see schema and small physical rows but never labels,
ground truth, or evaluator results.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path


_TABLE = re.compile(r'\b(?:from|join)\s+["\[]?([A-Za-z_][A-Za-z_0-9]*)', re.I)


def semantic_risk(question: str, sql: str) -> str | None:
    q, s = question.casefold(), sql.casefold()
    # No silent guesses on compound SQL / CTE / nested expressions.
    if not re.match(r"\s*select\b", s) or re.search(r"\b(union|intersect|except)\b", s):
        return None
    projection = re.search(r"\bselect\s+(.*?)\s+from\b", s, flags=re.S)
    head = projection.group(1) if projection else ""
    if re.search(r"\b(how many|number of)\b", q):
        if re.search(r"\bjoin\b", s) and re.search(r"\bcount\s*\(\s*\*\s*\)", head):
            return "COUNT_JOIN_FANOUT_RISK"
    if re.search(r"\b(percent(?:age)?|ratio|proportion|fraction)\b", q):
        if "/" not in head and not re.search(r"\b(printf|round)\s*\(\s*.*?/", head):
            return "RATIO_PROJECTION_RISK"
    if re.search(r"\b(highest|lowest|top \d+|most expensive|least expensive)\b", q):
        if not re.search(r"\border\s+by\b", s) and not re.search(r"\b(max|min)\s*\(", head):
            return "RANKING_ORDER_RISK"
    return None


def _referenced_physical_tables(con: sqlite3.Connection, sql: str) -> list[str]:
    physical = {str(n).casefold(): str(n) for (n,) in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )}
    names = []
    for raw in _TABLE.findall(sql):
        name = physical.get(raw.casefold())
        if name and name not in names:
            names.append(name)
    return names[:3]


def schema_probe(database: str | Path, sql: str) -> tuple[str, bool]:
    """Return bounded physical FK evidence plus whether a real relation was found."""
    p = Path(database).resolve()
    con = sqlite3.connect(p.as_uri() + "?mode=ro", uri=True, timeout=2)
    try:
        con.execute("PRAGMA query_only=ON")
        parts, found_fk = [], False
        for table in _referenced_physical_tables(con, sql):
            quoted = '"' + table.replace('"', '""') + '"'
            fks = [
                f"{row[3]}->{row[2]}.{row[4]}"
                for row in con.execute(f"PRAGMA foreign_key_list({quoted})")
            ]
            found_fk |= bool(fks)
            parts.append(f"{table}: physical foreign keys {fks[:6]}")
        return "\n".join(parts)[:1200], found_fk
    finally:
        con.close()


def data_probe(database: str | Path, sql: str) -> str:
    """Bounded read-only examples, not exhaustive data and not ground truth."""
    p = Path(database).resolve()
    con = sqlite3.connect(p.as_uri() + "?mode=ro", uri=True, timeout=2)
    try:
        con.execute("PRAGMA query_only=ON")
        parts = []
        for table in _referenced_physical_tables(con, sql):
            quoted = '"' + table.replace('"', '""') + '"'
            cols = [r[1] for r in con.execute(f"PRAGMA table_info({quoted})")][:4]
            examples = con.execute(f"SELECT * FROM {quoted} LIMIT 2").fetchall()
            values = [dict(zip(cols, (str(x)[:45] for x in row[:4]))) for row in examples]
            parts.append(f"{table}: sample {values}")
        return "\n".join(parts)[:1600]
    finally:
        con.close()


def preserves_filter_guard(original: str, proposed: str) -> bool:
    """Conservative guard: never auto-change an existing WHERE clause.

    Not a complete semantic equivalence checker; complex/nested SQL abstains.
    """
    if re.search(r"\b(with|union|intersect|except)\b", original, re.I):
        return False
    pattern = re.compile(
        r"\bwhere\b(.*?)(?=\b(group\s+by|having|order\s+by|limit)\b|$)",
        re.I | re.S,
    )
    orig, new = pattern.search(original), pattern.search(proposed)
    if orig is None:
        return True
    if new is None:
        return False
    norm = lambda value: re.sub(r"\s+", "", value).casefold().strip("()")
    return norm(orig.group(1)) == norm(new.group(1))
