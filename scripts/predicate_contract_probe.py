"""Experimental fail-closed predicate contract; never modifies SQL or reads labels."""
from __future__ import annotations
import sqlite3
from pathlib import Path

def inspect_predicate(question: str, sql: str, database: str | Path) -> dict:
    """Return physical observations, NOT a semantic correctness verdict.

    Only unambiguous single-table, qualified equality literals are eligible.
    Compound predicates, nested queries and ambiguous intent abstain.
    """
    from sqlglot import exp, parse_one
    result = {"status": "ABSTAIN", "reason": "UNSUPPORTED", "observations": []}
    try:
        root = parse_one(sql, read="sqlite")
    except Exception:
        return {**result, "reason": "AST_PARSE_ERROR"}
    if not isinstance(root, exp.Select) or root.find(exp.Subquery) or root.find(exp.Join) or root.find(exp.With) or root.find(exp.Union):
        return {**result, "reason": "COMPLEX_SCOPE"}
    tables = list(root.find_all(exp.Table))
    if len(tables) != 1:
        return {**result, "reason": "AMBIGUOUS_TABLE"}
    table = tables[0]
    alias = table.alias_or_name
    where = root.args.get("where")
    if not where or not isinstance(where.this, exp.EQ):
        return {**result, "reason": "COMPOUND_OR_MISSING_PREDICATE"}
    eq = where.this
    col, literal = eq.left, eq.right
    if not isinstance(col, exp.Column) or not isinstance(literal, exp.Literal) or not literal.is_string:
        return {**result, "reason": "NON_LITERAL_PREDICATE"}
    if col.table and col.table.casefold() != alias.casefold():
        return {**result, "reason": "UNRESOLVED_ALIAS"}
    value = literal.this
    # Independent grounding: only explicit quoted strings, not inferred synonyms.
    if not (f'"{value}"' in question or f"'{value}'" in question):
        return {**result, "reason": "NO_EXPLICIT_QUESTION_BINDING"}
    if len(value) < 2 or len(value) > 100:
        return {**result, "reason": "VALUE_BOUNDS"}
    def quote(s: str) -> str:
        return '"' + s.replace('"', '""') + '"'
    try:
        uri = Path(database).resolve().as_uri() + "?mode=ro&immutable=1"
        with sqlite3.connect(uri, uri=True) as conn:
            conn.execute("PRAGMA query_only=ON")
            columns = {r[1].casefold() for r in conn.execute(f"PRAGMA table_info({quote(table.name)})")}
            if col.name.casefold() not in columns:
                return {**result, "reason": "UNKNOWN_COLUMN"}
            # Validate table via sqlite_master before interpolating quoted identifier.
            if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table.name,)).fetchone():
                return {**result, "reason": "UNKNOWN_TABLE"}
            count = [0]
            def budget():
                count[0] += 1
                return int(count[0] > 2000)
            conn.set_progress_handler(budget, 1000)
            try:
                query = f"SELECT 1 FROM {quote(table.name)} WHERE {quote(col.name)} COLLATE BINARY = ? LIMIT 1"
                exists = conn.execute(query, (value,)).fetchone() is not None
            finally:
                conn.set_progress_handler(None, 0)
    except (sqlite3.Error, OSError):
        return {**result, "reason": "PHYSICAL_PROBE_ERROR"}
    return {"status": "OBSERVED", "reason": "EXACT_VALUE_PRESENT" if exists else "EXACT_VALUE_ABSENT",
            "observations": [{"table": table.name, "column": col.name, "exact_exists": exists,
                              "semantic_correctness": "UNKNOWN"}]}
