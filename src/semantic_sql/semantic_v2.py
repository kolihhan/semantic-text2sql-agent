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


def _physical_columns_readonly(database: str | Path, table: str) -> tuple[str, tuple[str, ...]] | None:
    """Read physical column metadata without ever opening a writable DB handle."""
    try:
        con, _ = _open_readonly_connection(database)
    except (sqlite3.Error, OSError):
        return None
    try:
        names = [row[0] for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )]
        name = next((name for name in names if name.lower() == table.lower()), None)
        if name is None:
            return None
        cols = tuple(row[1] for row in con.execute(
            f"PRAGMA table_info({_quote_identifier(name)})"
        ))
        return (name, cols) if cols else None
    except sqlite3.Error:
        return None
    finally:
        con.close()


def _counterexample(
    original_sql: str, proposed_sql: str, *,
    tables: tuple[tuple[str, tuple[str, ...]], ...],
    target_table: str, column: str, requested: str,
) -> bool:
    """Compare SELECT results on in-memory one-row-per-table fixtures.

    JOIN equality keys use synthetic value '1'; the requested column is
    guaranteed not to be a join key by the caller. This is a behavioral
    counterexample, not a proof that the resulting query is fully correct.
    """
    if len(tables) not in (1, 2, 3):
        return False
    if any(len(cols) > 200 or len({c.lower() for c in cols}) != len(cols)
           for _, cols in tables):
        return False
    con = sqlite3.connect(":memory:")
    try:
        for table, cols in tables:
            quoted_table = _quote_identifier(table)
            fields = ", ".join(_quote_identifier(c) + " TEXT" for c in cols)
            con.execute(f"CREATE TABLE {quoted_table} ({fields})")
        for table, cols in tables:
            values = [
                requested if table.lower() == target_table.lower()
                    and c.lower() == column.lower() else "1"
                for c in cols
            ]
            quoted_cols = ", ".join(_quote_identifier(c) for c in cols)
            placeholders = ", ".join("?" for _ in values)
            con.execute(
                f"INSERT INTO {_quote_identifier(table)} ({quoted_cols}) VALUES ({placeholders})",
                values,
            )
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
    if (parsed.args.get("with_") or parsed.find(exp.Subquery)
        or parsed.find(exp.Union) or parsed.find(exp.CTE)):
        return keep("join_cte_or_nested_query")

    source = parsed.args.get("from_")
    table_ast = source.this if source is not None else None
    if not isinstance(table_ast, exp.Table):
        return keep("not_one_physical_table")
    joins = parsed.args.get("joins") or []
    if len(joins) > 2:
        return keep("complex_or_unsafe_join")
    table_exprs = [table_ast]
    join_keys: set[tuple[str, str]] = set()
    for join in joins:
        joined = join.this
        on = join.args.get("on")
        if (not isinstance(joined, exp.Table)
            or join.args.get("side")
            or join.args.get("kind") not in (None, "", "INNER")
            or not isinstance(on, exp.EQ)
            or not isinstance(on.this, exp.Column)
            or not isinstance(on.expression, exp.Column)):
            return keep("complex_or_unsafe_join")
        if not on.this.table or not on.expression.table:
            return keep("complex_or_unsafe_join")
        join_keys.add((on.this.table.lower(), on.this.name.lower()))
        join_keys.add((on.expression.table.lower(), on.expression.name.lower()))
        table_exprs.append(joined)
    if any(t.args.get("db") or t.args.get("catalog") for t in table_exprs):
        return keep("qualified_database")
    aliases = {t.alias_or_name.lower(): t.name for t in table_exprs}
    if (len(aliases) != len(table_exprs)
        or len({t.name.lower() for t in table_exprs}) != len(table_exprs)):
        return keep("ambiguous_or_self_join")
    if any(a not in aliases for a, _ in join_keys):
        return keep("unknown_join_alias")
    where = parsed.args.get("where")
    predicate = where.this if where is not None else None
    if where is None and not joins:
        # A missing filter is a different semantic failure from a wrong value.
        # Only the evidence-gated independent single-table path may propose one.
        return _evidence_gated_missing_filter(
            database=database, question=question, original_sql=sql,
            parsed=parsed, quoted_value=value,
        )
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
    if field.is_star:
        return keep("ambiguous_column")
    if len(aliases) > 1 and not field.table:
        return keep("ambiguous_join_filter_column")
    target_alias = (field.table or table_ast.alias_or_name).lower()
    if target_alias not in aliases:
        return keep("ambiguous_column")
    if (target_alias, field.name.lower()) in join_keys:
        return keep("target_is_join_key")
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
        database, aliases[target_alias], field.name, value
    )
    if evidence_status != "verified":
        return keep(evidence_status)
    matched_col = next(c for c in physical_columns if c.lower() == field.name.lower())
    physical_tables = []
    for alias, raw_table in aliases.items():
        if alias == target_alias:
            physical_tables.append((matched_table, physical_columns))
        else:
            found = _physical_columns_readonly(database, raw_table)
            if found is None:
                return keep("joined_table_not_in_physical_catalog")
            physical_tables.append(found)
    # Join equality keys must refer to physically declared columns in their
    # associated base tables, never generated aliases or guessed fields.
    physical_by_alias = {
        alias: cols for alias, (_, cols) in zip(aliases, physical_tables)
    }
    if any(c not in {name.lower() for name in physical_by_alias[a]}
           for a, c in join_keys):
        return keep("join_key_not_in_physical_catalog")

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
        sql, proposed_sql, tables=tuple(physical_tables),
        target_table=matched_table, column=matched_col, requested=actual,
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


def _evidence_gated_missing_filter(
    *, database: str | Path, question: str, original_sql: str, parsed, quoted_value: str,
) -> SemanticV2Decision:
    """Conservative missing WHERE candidate on one real table (no LLM, no gold).

    The user must request a named/quoted entity using an explicit predicate-
    like phrase. That value must exist as an exact match in exactly ONE
    physical text column; no other question value or SQL predicate may exist.
    A two-row synthetic database confirms the appended WHERE changes behavior.
    """
    import sqlglot
    from sqlglot import exp

    def keep(reason: str) -> SemanticV2Decision:
        return SemanticV2Decision(sql=original_sql, changed=False, reason=reason)

    # Quotations alone can denote examples, not filtering requests.
    explicit_filter = re.search(
        r"\b(?:for|of|in|at|named|called|with|whose|where|regarding|"
        r"about|from|labeled|labelled|containing|matching)\b"
        r"[^\x27\x22\x60]{0,55}[\x27\x22\x60]",
        question, flags=re.I,
    )
    if not explicit_filter:
        return keep("quoted_entity_not_explicit_filter")
    if (parsed.args.get("joins") or parsed.args.get("with_") or
        any(parsed.args.get(k) is not None for k in
            ("group","having","order","limit","offset","distinct","qualify")) or
        parsed.find(exp.Union) or parsed.find(exp.Subquery) or
        parsed.find(exp.Window) or parsed.find(exp.Anonymous) or
        parsed.find(exp.CTE)):
        return keep("omitted_filter_complex_query")
    if any(x.is_string for x in parsed.find_all(exp.Literal)):
        return keep("omitted_filter_sql_already_has_string_literals")
    for selection in parsed.expressions:
        item = selection.this if isinstance(selection, exp.Alias) else selection
        if not isinstance(item, (exp.Column, exp.Star, exp.Count)):
            return keep("omitted_filter_unsupported_projection")
        if isinstance(item, exp.Count) and item.find(exp.Distinct):
            return keep("omitted_filter_distinct_aggregate")

    source=parsed.args.get("from_")
    table=source.this if source is not None else None
    if not isinstance(table, exp.Table) or table.args.get("db") or table.args.get("catalog"):
        return keep("omitted_filter_not_one_physical_table")
    found=_physical_columns_readonly(database,table.name)
    if found is None:
        return keep("omitted_filter_table_not_in_catalog")
    physical_table, columns=found
    if len(columns)>24:
        return keep("omitted_filter_too_many_columns")
    try:
        con,_=_open_readonly_connection(database)
    except (sqlite3.Error,OSError):
        return keep("omitted_filter_database_unavailable")
    matches=[]
    try:
        # A bounded instruction budget avoids unbounded scans on large BIRD DBs.
        calls=[0]
        def progress():
            calls[0]+=1
            return int(calls[0]>1000)
        con.set_progress_handler(progress,1000)
        metadata=con.execute(f"PRAGMA table_info({_quote_identifier(physical_table)})").fetchall()
        eligible=[
            col for _,col,kind,*_ in metadata
            if col in columns and any(token in str(kind).upper()
                                      for token in ("TEXT","CHAR","CLOB","VARCHAR"))
        ]
        if len(eligible)>16:
            return keep("omitted_filter_too_many_text_columns")
        for col in eligible:
            # Exact equality with ASCII-insensitive matching, not LIKE.
            row=con.execute(
                f"SELECT {_quote_identifier(col)} FROM {_quote_identifier(physical_table)} "
                f"WHERE {_quote_identifier(col)} = ? COLLATE NOCASE LIMIT 1",
                (quoted_value,),
            ).fetchone()
            if row is not None and isinstance(row[0],str) and row[0].casefold()==quoted_value.casefold():
                matches.append((col,row[0]))
    except sqlite3.Error:
        return keep("omitted_filter_value_lookup_failed_or_too_large")
    finally:
        con.close()
    if len(matches)!=1:
        return keep("omitted_filter_no_unique_physical_column")
    col,actual=matches[0]
    qualifier=table.alias_or_name if table.alias else None
    alternate=parsed.copy()
    alternate.set("where",exp.Where(this=exp.EQ(
        this=exp.column(col,table=qualifier),
        expression=exp.Literal.string(actual),
    )))
    proposed=alternate.sql(dialect="sqlite")
    if not verify_sql_preflight(database,proposed).ok:
        return keep("omitted_filter_revised_sql_failed_preflight")

    # A contrast row is needed: without it, SELECT without WHERE and SELECT
    # with WHERE would agree on a one-row fixture and prove nothing.
    if len(quoted_value)>80 or actual=="__semantic_v2_decoy__":
        return keep("omitted_filter_counterexample_unsupported")
    in_memory=sqlite3.connect(":memory:")
    try:
        fields=", ".join(_quote_identifier(c)+" TEXT" for c in columns)
        in_memory.execute(f"CREATE TABLE {_quote_identifier(physical_table)} ({fields})")
        names=", ".join(_quote_identifier(c) for c in columns)
        params=", ".join("?" for _ in columns)
        for entity in (actual,"__semantic_v2_decoy__"):
            row=[entity if c==col else str(i+1) for i,c in enumerate(columns)]
            in_memory.execute(
                f"INSERT INTO {_quote_identifier(physical_table)} ({names}) VALUES ({params})",row)
        in_memory.execute("PRAGMA query_only = ON")
        before=in_memory.execute(original_sql).fetchmany(4)
        after=in_memory.execute(proposed).fetchmany(4)
        if before==after:
            return keep("omitted_filter_counterexample_not_discriminating")
    except sqlite3.Error:
        return keep("omitted_filter_counterexample_invalid")
    finally:
        in_memory.close()
    return SemanticV2Decision(
        sql=proposed, changed=True,
        reason="missing_where_explicit_entity_and_single_column_evidence",
        evidence=(
            "one_explicitly_filtering_quoted_entity",
            "only_one_physical_text_column_matches",
            "only_where_clause_appended",
            "two_row_in_memory_counterexample_discriminates",
            "revised_sql_passes_readonly_preflight",
        ),
    )
