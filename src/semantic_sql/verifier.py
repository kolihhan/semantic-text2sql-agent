from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from .contracts import VerificationIssue, VerificationResult
from .execution import SQLExecutionError, _open_readonly_connection, split_sql_segments, validate_readonly_sql


def verify_sql_preflight(database: str | Path, sql: str) -> VerificationResult:
    issues: list[VerificationIssue] = []
    seen: set[str] = set()

    def add_issue(code: str, message: str) -> None:
        if code not in seen:
            issues.append(VerificationIssue(code=code, message=message))
            seen.add(code)

    segments, saw_comment = split_sql_segments(sql)
    if not segments:
        add_issue("empty_sql", "candidate SQL is empty")
        return VerificationResult(ok=False, issues=tuple(issues))
    if len(segments) > 1:
        add_issue("multiple_statements", "candidate must contain exactly one SQL statement")
    if saw_comment:
        add_issue("non_sql_text", "candidate contains commentary instead of SQL only")
    readonly_count = 0
    for segment in segments:
        first = re.match(r"\s*([A-Za-z]+)\b", segment)
        first_word = first.group(1).lower() if first else ""
        if first_word in {"select", "with"}:
            readonly_count += 1
        elif first_word in {"alter", "attach", "create", "delete", "detach", "drop", "insert", "pragma", "reindex", "replace", "update", "vacuum"}:
            add_issue("not_readonly", "candidate is not read-only")
        else:
            add_issue("non_sql_text", "candidate contains text that is not a SELECT or CTE")
    if readonly_count > 1:
        add_issue("multiple_statements", "candidate contains more than one SELECT/CTE")
    if issues:
        return VerificationResult(ok=False, issues=tuple(issues))
    try:
        normalized = validate_readonly_sql(segments[0])
    except SQLExecutionError as exc:
        add_issue("not_readonly", str(exc))
        return VerificationResult(ok=False, issues=tuple(issues))
    connection: sqlite3.Connection | None = None
    blocked_actions: list[int] = []
    try:
        connection, blocked_actions = _open_readonly_connection(database)
        connection.execute(f"EXPLAIN QUERY PLAN {normalized}").fetchall()
    except sqlite3.Error as exc:
        if blocked_actions:
            add_issue("not_readonly", "candidate contains a write operation")
        else:
            add_issue("sqlite_compile_error", str(exc))
    finally:
        if connection is not None:
            connection.close()
    return VerificationResult(ok=not issues, issues=tuple(issues))
