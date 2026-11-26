# MIT-licensed upstream-derived component; copyright notice in this directory's LICENSE.
"""Read-only SQLite access with layered SQL safety."""

from __future__ import annotations

import difflib
import hashlib
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlglot import exp, parse
from sqlglot.errors import ParseError

from .schema import DEFAULT_DB_PATH, get_identifier_allowlist, get_table_columns

FORBIDDEN_NODES = (
    exp.Alter,
    exp.Attach,
    exp.Command,
    exp.Create,
    exp.Delete,
    exp.Detach,
    exp.Drop,
    exp.Insert,
    exp.Merge,
    exp.Pragma,
    exp.Transaction,
    exp.Update,
)
FORBIDDEN_FUNCTIONS = {"load_extension", "readfile", "writefile"}
READ_ROOTS = (exp.Select, exp.Union, exp.Intersect, exp.Except)
_TIMEOUTS: dict[int, float] = {}


def connect_readonly(
    path: str | Path = DEFAULT_DB_PATH,
    *,
    timeout_seconds: float = 2.0,
) -> sqlite3.Connection:
    """Open SQLite read-only and install engine-level defenses."""
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Database not found: {resolved}. Run ./scripts/setup_chinook.sh."
        )
    connection = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only = ON")
    connection.enable_load_extension(False)

    denied = {
        value
        for name in (
            "SQLITE_INSERT",
            "SQLITE_UPDATE",
            "SQLITE_DELETE",
            "SQLITE_CREATE_INDEX",
            "SQLITE_CREATE_TABLE",
            "SQLITE_CREATE_TEMP_INDEX",
            "SQLITE_CREATE_TEMP_TABLE",
            "SQLITE_CREATE_TEMP_TRIGGER",
            "SQLITE_CREATE_TEMP_VIEW",
            "SQLITE_CREATE_TRIGGER",
            "SQLITE_CREATE_VIEW",
            "SQLITE_DROP_INDEX",
            "SQLITE_DROP_TABLE",
            "SQLITE_DROP_TEMP_INDEX",
            "SQLITE_DROP_TEMP_TABLE",
            "SQLITE_DROP_TEMP_TRIGGER",
            "SQLITE_DROP_TEMP_VIEW",
            "SQLITE_DROP_TRIGGER",
            "SQLITE_DROP_VIEW",
            "SQLITE_ALTER_TABLE",
            "SQLITE_ATTACH",
            "SQLITE_DETACH",
            "SQLITE_PRAGMA",
            "SQLITE_REINDEX",
        )
        if (value := getattr(sqlite3, name, None)) is not None
    }

    def authorizer(
        action: int,
        _arg1: str | None,
        _arg2: str | None,
        _database: str | None,
        _trigger: str | None,
    ) -> int:
        return sqlite3.SQLITE_DENY if action in denied else sqlite3.SQLITE_OK

    connection.set_authorizer(authorizer)
    _TIMEOUTS[id(connection)] = timeout_seconds
    return connection


def _arm_timeout(conn: sqlite3.Connection) -> None:
    deadline = time.monotonic() + _TIMEOUTS.get(id(conn), 2.0)
    conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 1_000)


def _cte_names(statement: exp.Expression) -> set[str]:
    return {
        cte.alias_or_name.casefold()
        for cte in statement.find_all(exp.CTE)
        if cte.alias_or_name
    }


def _select_scopes(statement: exp.Expression) -> list[exp.Select]:
    """Return every SELECT scope, including CTE and derived-table bodies."""
    return [node for node in statement.find_all(exp.Select) if isinstance(node, exp.Select)]


def _from_sources(select: exp.Select) -> list[exp.Expression]:
    """Direct FROM/JOIN sources for one SELECT (not nested inside those sources)."""
    sources: list[exp.Expression] = []
    from_clause = select.args.get("from_")
    if from_clause is not None and from_clause.this is not None:
        sources.append(from_clause.this)
    for join in select.args.get("joins") or []:
        if join.this is not None:
            sources.append(join.this)
    return sources


def _scope_aliases(
    select: exp.Select,
    *,
    allowlist: dict[str, frozenset[str]],
    ctes: set[str],
) -> tuple[dict[str, str], set[str], str | None]:
    """Map aliases visible in this SELECT; derived sources skip physical column checks."""
    aliases: dict[str, str] = {}
    derived = set(ctes)
    for source in _from_sources(select):
        if isinstance(source, exp.Table):
            name = source.name.casefold()
            if name not in allowlist and name not in ctes:
                return {}, derived, f"Unknown table: {source.name}"
            aliases[(source.alias_or_name or source.name).casefold()] = name
        elif isinstance(source, exp.Subquery):
            alias = (source.alias_or_name or "").casefold()
            if alias:
                aliases[alias] = alias
                derived.add(alias)
    return aliases, derived, None


def _resolve_qualifier(
    qualifier: str,
    select_scope: exp.Select | None,
    scope_maps: dict[int, tuple[dict[str, str], set[str]]],
    ctes: set[str],
) -> tuple[str, set[str]]:
    """Resolve a column qualifier through nested SELECT scopes (correlated refs)."""
    current = select_scope
    while current is not None:
        aliases, derived = scope_maps.get(id(current), ({}, set(ctes)))
        if qualifier in aliases:
            return aliases[qualifier], derived
        parent = current.parent
        current = None
        while parent is not None:
            if isinstance(parent, exp.Select):
                current = parent
                break
            parent = parent.parent
    return qualifier, set(ctes)


def _nearest_select(node: exp.Expression) -> exp.Select | None:
    parent = node.parent
    while parent is not None:
        if isinstance(parent, exp.Select):
            return parent
        parent = parent.parent
    return None


def _output_aliases_for_select(select: exp.Select) -> set[str]:
    return {
        alias.alias.casefold()
        for alias in select.find_all(exp.Alias)
        if alias.alias and _nearest_select(alias) is select
    }


def is_safe(
    sql: str,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> tuple[bool, str]:
    """Validate one read-only statement using a SQLite-dialect AST."""
    try:
        statements = [statement for statement in parse(sql, read="sqlite") if statement]
    except ParseError as exc:
        return False, f"SQL parse failed: {exc}"
    if len(statements) != 1:
        return False, "Exactly one SQL statement is required"

    statement = statements[0]
    if not isinstance(statement, READ_ROOTS):
        return False, "Only SELECT or WITH queries are allowed"
    forbidden = next(
        (node for node in statement.walk() if isinstance(node, FORBIDDEN_NODES)),
        None,
    )
    if forbidden is not None:
        return False, f"Forbidden SQL operation: {forbidden.key.upper()}"

    for function in statement.find_all(exp.Func):
        function_name = (
            function.name if isinstance(function, exp.Anonymous) else function.sql_name()
        )
        if function_name.casefold() in FORBIDDEN_FUNCTIONS:
            return False, f"Forbidden SQL function: {function_name}"

    allowlist = get_identifier_allowlist(db_path)
    ctes = _cte_names(statement)
    all_columns = set().union(*allowlist.values()) if allowlist else set()
    has_derived = bool(ctes) or any(statement.find_all(exp.Subquery))

    scope_maps: dict[int, tuple[dict[str, str], set[str]]] = {}
    for select in _select_scopes(statement):
        aliases, derived, error = _scope_aliases(select, allowlist=allowlist, ctes=ctes)
        if error:
            return False, error
        scope_maps[id(select)] = (aliases, derived)

    for column in statement.find_all(exp.Column):
        name = column.name.casefold()
        qualifier = column.table.casefold() if column.table else ""
        if name == "*":
            continue
        select_scope = _nearest_select(column)
        aliases, derived = (
            scope_maps.get(id(select_scope), ({}, set(ctes)))
            if select_scope is not None
            else ({}, set(ctes))
        )
        output_aliases = (
            _output_aliases_for_select(select_scope) if select_scope is not None else set()
        )
        if qualifier:
            table_name, derived = _resolve_qualifier(
                qualifier, select_scope, scope_maps, ctes
            )
            if table_name in derived or table_name in ctes:
                continue
            if table_name not in allowlist:
                return False, f"Unknown table or alias: {column.table}"
            if name not in allowlist[table_name]:
                return False, f"Unknown column: {column.sql(dialect='sqlite')}"
        elif name not in all_columns and name not in output_aliases:
            # Derived-table/CTE outputs are not present in the physical schema.
            if not has_derived and not derived:
                return False, f"Unknown column: {column.name}"
    return True, ""


def validate(conn: sqlite3.Connection, sql: str) -> tuple[bool, str]:
    """Compile a query without executing it."""
    try:
        _arm_timeout(conn)
        conn.execute(f"EXPLAIN QUERY PLAN {sql}")
        return True, ""
    except sqlite3.Error as exc:
        return False, str(exc)


def execute(
    conn: sqlite3.Connection,
    sql: str,
    *,
    limit: int | None = 200,
) -> tuple[list[str], list[tuple[Any, ...]], bool]:
    """Execute a validated query, optionally retaining one row to detect truncation."""
    _arm_timeout(conn)
    cursor = conn.execute(sql)
    columns = [description[0] for description in cursor.description or ()]
    if limit is None:
        rows = cursor.fetchall()
        return columns, [tuple(row) for row in rows], False
    rows = cursor.fetchmany(limit + 1)
    truncated = len(rows) > limit
    return columns, [tuple(row) for row in rows[:limit]], truncated


def result_signature(rows: list[tuple[Any, ...]]) -> str:
    """Order- and duplicate-insensitive fingerprint of a full result set.

    Equal signatures mean equal results under BIRD's official ``set(rows)`` comparison, so
    this is the key for clustering candidates by execution result.
    """
    canonical = sorted(repr(row) for row in set(rows))
    return hashlib.sha256("\n".join(canonical).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class CandidateResult:
    """One candidate query's full execution outcome, independent of any gold answer."""

    sql: str
    error: str | None
    columns: list[str]
    rows: list[tuple[Any, ...]]
    signature: str | None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def row_count(self) -> int:
        return len(self.rows)


def execute_candidate(
    conn: sqlite3.Connection, sql: str, *, db_path: str | Path = DEFAULT_DB_PATH
) -> CandidateResult:
    """Safety-check then fully execute one candidate, with no row cap and no gold query.

    This is the executor for multi-candidate generation, cascades, and hidden-test runs:
    the AST safety layer gates every candidate exactly as it gates the agent's own SQL, and
    the connection's read-only/authorizer/timeout defenses still apply underneath.
    """
    safe, reason = is_safe(sql, db_path=db_path)
    if not safe:
        return CandidateResult(sql, f"safety-rejected: {reason}", [], [], None)
    try:
        columns, rows, _ = execute(conn, sql, limit=None)
    except sqlite3.Error as exc:
        return CandidateResult(sql, f"execute-failed: {exc}", [], [], None)
    return CandidateResult(sql, None, columns, rows, result_signature(rows))


# Databases above this size get a longer execution budget (BIRD's largest are 0.4-0.6GB and
# legitimately need more than the interactive 2s ceiling).
LARGE_DATABASE_BYTES = 100 * 1024 * 1024
LARGE_DATABASE_TIMEOUT_S = 10.0


def timeout_for_database(db_path: str | Path, default: float = 2.0) -> float:
    """Execution timeout scaled to database size (see LARGE_DATABASE_BYTES)."""
    return LARGE_DATABASE_TIMEOUT_S if Path(db_path).stat().st_size > LARGE_DATABASE_BYTES else default


RECOVERABLE_SAFETY_PREFIXES = ("Unknown column", "Unknown table", "SQL parse failed")


def is_recoverable_rejection(reason: str) -> bool:
    """Unknown identifiers and parse failures are repairable mistakes (often an unquoted
    ``T-BIL``); forbidden operations, multiple statements, and non-SELECTs never are."""
    return reason.startswith(RECOVERABLE_SAFETY_PREFIXES)


def identifier_hint(reason: str, db_path: str | Path = DEFAULT_DB_PATH) -> str:
    """Suggest real schema names for an unknown-identifier rejection."""
    token = reason.split(":", 1)[-1].strip().split(".")[-1].strip("`\"'[] ").casefold()
    names = [
        (table, column)
        for table, columns in get_table_columns(db_path).items()
        for column in [table, *columns]
    ]
    lowered = {name.casefold(): (table, name) for table, name in names}
    # A bare prefix of a punctuated name (``T`` from an unquoted ``T-BIL``, ``Character``
    # from ``Character Name``) is the most likely intent, so those candidates rank first.
    matches = [
        key
        for key in lowered
        if token and key.startswith(token) and len(key) > len(token) and not key[len(token)].isalnum()
    ]
    matches += [
        key
        for key in (difflib.get_close_matches(token, list(lowered), n=5, cutoff=0.6) if token else [])
        if key not in matches
    ]
    matches += [
        key
        for key in lowered
        if token and len(token) >= 3 and token in key.replace("_", " ").split()
        and key not in matches
    ][:5]
    rendered = []
    for key in matches[:6]:
        table, name = lowered[key]
        quoted = name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) else f"`{name}`"
        rendered.append(quoted if name == table else f"{quoted} (in {table})")
    suggestion = f" Did you mean: {', '.join(rendered)}?" if rendered else ""
    return (
        suggestion
        + " Names containing spaces or punctuation must be written in backticks exactly as"
        " in the schema."
    )
