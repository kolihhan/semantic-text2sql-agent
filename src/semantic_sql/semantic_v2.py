"""Experimental Semantic Layer v2: evidence-gated literal correction.

Narrow zero-training intervention, NOT a general SQL semantic verifier.
A single wrong equality string literal may be replaced only when an explicit
quoted question value occurs in the same physical database column, the AST
structure otherwise remains unchanged, the result passes SQL preflight and an
isolated synthetic SQLite counterexample demonstrates a behavioral difference.

No LLM reviewer, no gold SQL, no changes to the user's database.
Opt-in only; performance must be independently measured before promotion.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3

from .execution import _open_readonly_connection
from .verifier import verify_sql_preflight


@dataclass(frozen=True)
class SemanticV2Decision:
    sql: str
    changed: bool
    reason: str
    evidence: tuple[str, ...] = ()


_QUOTED = re.compile(r"""(?<!\w)(?P<mark>['"\x60])(?P<value>[^'"\x60]{2,80})(?P=mark)""")
_SAFE_VALUE = re.compile(r"^[\w \t.,&()/+\-]{2,80}$", flags=re.UNICODE)


def _quoted_request(question: str) -> str | None:
    matches = [m.group("value").strip() for m in _QUOTED.finditer(question)]
    if len(matches) != 1 or not _SAFE_VALUE.fullmatch(matches[0]):
        return None
    return matches[0]


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _physical_value_readonly(database: str | Path, table: str, column: str, value: str):
    """Physical-column validation via SQLite mode=ro, never a writable connection."""
    try:
        con, _ = _open_readonly_connection(database)
    except (sqlite3.Error, OSError):
        return None, (), None, "physical_database_unavailable"
    try:
        names = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )]
        matched_table = next((t for t in names if t.lower() == table.lower()), None)
        if matched_table is None:
            return None, (), None, "table_not_in_physical_catalog"
        cols = tuple(r[1] for r in con.execute(
            f"PRAGMA table_info({_quote_identifier(matched_table)})"
        ))
        matched_col = next((c for c in cols if c.lower() == column.lower()), None)
        if matched_col is None:
            return None, (), None, "column_not_in_physical_catalog"
        quoted_col = _quote_identifier(matched_col)
        row = con.execute(
            f"SELECT {quoted_col} FROM {_quote_identifier(matched_table)} "
            f"WHERE CAST({quoted_col} AS TEXT) = ? COLLATE NOCASE LIMIT 1",
            (value,),
        ).fetchone()
        actual = row[0] if row is not None else None
        if not isinstance(actual, str) or actual.casefold() != value.casefold():
            return None, (), None, "question_value_not_verified_in_same_column"
        return matched_table, cols, actual, "verified"
    except sqlite3.Error:
        return None, (), None, "physical_evidence_query_failed"
    finally:
        con.close()


def _counterexample(
    original_sql: str, proposed_sql: str, *,
    table: str, columns: tuple[str, ...], column: str, requested: str,
) -> bool:
    """Use synthetic in-memory data; a difference alone is not a correctness proof."""
    if len(columns) > 200 or len({c.lower() for c in columns}) != len(columns):
        return False
    con = sqlite3.connect(":memory:")
    try:
        quoted_table = _quote_identifier(table)
        fields = ", ".join(_quote_identifier(c) + " TEXT" for c in columns)
        con.execute(f"CREATE TABLE {quoted_table} ({fields})")
        values = [requested if c.lower() == column.lower() else "1" for c in columns]
        cols = ", ".join(_quote_identifier(c) for c in columns)
        placeholders = ", ".join("?" for _ in values)
        con.execute(f"INSERT INTO {quoted_table} ({cols}) VALUES ({placeholders})", values)
        con.execute("PRAGMA query_only = ON")
        before = con.execute(original_sql).fetchmany(3)
        after = con.execute(proposed_sql).fetchmany(3)
        return before != after
    except sqlite3.Error:
        return False
    finally:
        con.close()


def evidence_gated_literal_revision(
    *, database: str | Path, question: str, sql: str,
) -> SemanticV2Decision:
    """Return a deterministic, narrow correction, or preserve the input SQL."""
    def keep(reason: str) -> SemanticV2Decision:
        return SemanticV2Decision(sql=sql, changed=False, reason=reason)

    value = _quoted_request(question)
    if value is None:
        return keep("no_unambiguous_quoted_entity")
    try:
        import sqlglot
        from sqlglot import exp
    except ImportError:
        return keep("sqlglot_not_installed")
    try:
        parsed = sqlglot.parse_one(sql, read="sqlite")
    except (sqlglot.errors.ParseError, TypeError, ValueError):
        return keep("unparseable_sql")
    if not isinstance(parsed, exp.Select):
        return keep("non_simple_select")
    if (parsed.args.get("joins") or parsed.args.get("with_")
        or parsed.find(exp.Subquery) or parsed.find(exp.Union)
        or parsed.find(exp.CTE)):
        return keep("join_cte_or_nested_query")

    source = parsed.args.get("from_")
    table_ast = source.this if source is not None else None
    if not isinstance(table_ast, exp.Table):
        return keep("not_one_physical_table")
    if table_ast.args.get("db") or table_ast.args.get("catalog"):
        return keep("qualified_database")
    where = parsed.args.get("where")
    predicate = where.this if where is not None else None
    if not isinstance(predicate, exp.EQ):
        return keep("no_single_equality_filter")

    lhs, rhs = predicate.this, predicate.expression
    if isinstance(lhs, exp.Literal) and lhs.is_string and isinstance(rhs, exp.Column):
        old_literal, field = lhs, rhs
    elif isinstance(rhs, exp.Literal) and rhs.is_string and isinstance(lhs, exp.Column):
        old_literal, field = rhs, lhs
    else:
        return keep("not_column_equals_string")

    if len([node for node in parsed.find_all(exp.Literal) if node.is_string]) != 1:
        return keep("multiple_sql_string_literals")
    if field.is_star or (field.table and field.table.lower() != table_ast.alias_or_name.lower()):
        return keep("ambiguous_column")
    if old_literal.this.casefold() == value.casefold():
        return keep("entity_already_matches_question")

    # Never adjust complex grouping, windows, arbitrary functions or ranking.
    if (parsed.args.get("group") or parsed.args.get("having")
        or parsed.args.get("order") or parsed.args.get("limit")
        or parsed.find(exp.Window) or parsed.find(exp.Anonymous)):
        return keep("complex_projection_or_order")
    for selection in parsed.expressions:
        expression = selection.this if isinstance(selection, exp.Alias) else selection
        if not isinstance(expression, (exp.Column, exp.Star, exp.Count)):
            return keep("unsupported_projection")

    matched_table, physical_columns, actual, evidence_status = _physical_value_readonly(
        database, table_ast.name, field.name, value
    )
    if evidence_status != "verified":
        return keep(evidence_status)
    matched_col = next(c for c in physical_columns if c.lower() == field.name.lower())

    candidate_tree = parsed.copy()
    cand_where = candidate_tree.args["where"].this
    cand_lhs, cand_rhs = cand_where.this, cand_where.expression
    replace_node = cand_lhs if isinstance(cand_lhs, exp.Literal) else cand_rhs
    replace_node.replace(exp.Literal.string(actual))
    proposed_sql = candidate_tree.sql(dialect="sqlite")
    if proposed_sql.strip() == sql.strip():
        return keep("no_semantic_delta")
    if not verify_sql_preflight(database, proposed_sql).ok:
        return keep("revised_preflight_failed")
    if not _counterexample(
        sql, proposed_sql, table=matched_table,
        columns=physical_columns, column=matched_col, requested=actual,
    ):
        return keep("synthetic_counterexample_not_discriminating")
    return SemanticV2Decision(
        sql=proposed_sql,
        changed=True,
        reason="single_literal_question_and_physical_value_evidence",
        evidence=(
            "question_contains_exactly_one_quoted_value",
            "physical_column_contains_requested_value",
            "single_ast_equality_literal_changed",
            "synthetic_in_memory_counterexample_discriminates_outputs",
            "revised_sql_passes_readonly_preflight",
        ),
    )
