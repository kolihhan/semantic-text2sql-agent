from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3


class SQLExecutionError(RuntimeError):
    pass


def split_sql_segments(sql: str) -> tuple[tuple[str, ...], bool]:
    segments: list[str] = []
    current: list[str] = []
    quote: str | None = None
    saw_comment = False
    index = 0
    while index < len(sql):
        char = sql[index]
        following = sql[index + 1] if index + 1 < len(sql) else ""
        if quote is not None:
            current.append(char)
            if char == quote:
                if following == quote and quote != "]":
                    current.append(following)
                    index += 2
                    continue
                quote = None
            index += 1
            continue
        if char in {"'", '"', "`"}:
            quote = char
            current.append(char)
            index += 1
            continue
        if char == "[":
            quote = "]"
            current.append(char)
            index += 1
            continue
        if char == "-" and following == "-":
            saw_comment = True
            index += 2
            while index < len(sql) and sql[index] not in "\r\n":
                index += 1
            current.append("\n")
            continue
        if char == "/" and following == "*":
            saw_comment = True
            index += 2
            while index + 1 < len(sql) and sql[index : index + 2] != "*/":
                index += 1
            index = min(index + 2, len(sql))
            current.append(" ")
            continue
        if char == ";":
            segment = "".join(current).strip()
            if segment:
                segments.append(segment)
            current.clear()
            index += 1
            continue
        current.append(char)
        index += 1
    trailing = "".join(current).strip()
    if trailing:
        segments.append(trailing)
    return tuple(segments), saw_comment


_WRITE_ACTIONS = frozenset(action for action in (getattr(sqlite3, "SQLITE_INSERT", None), getattr(sqlite3, "SQLITE_UPDATE", None), getattr(sqlite3, "SQLITE_DELETE", None), getattr(sqlite3, "SQLITE_ALTER_TABLE", None), getattr(sqlite3, "SQLITE_ATTACH", None), getattr(sqlite3, "SQLITE_DETACH", None), getattr(sqlite3, "SQLITE_CREATE_INDEX", None), getattr(sqlite3, "SQLITE_CREATE_TABLE", None), getattr(sqlite3, "SQLITE_CREATE_TEMP_INDEX", None), getattr(sqlite3, "SQLITE_CREATE_TEMP_TABLE", None), getattr(sqlite3, "SQLITE_CREATE_TEMP_TRIGGER", None), getattr(sqlite3, "SQLITE_CREATE_TEMP_VIEW", None), getattr(sqlite3, "SQLITE_CREATE_TRIGGER", None), getattr(sqlite3, "SQLITE_CREATE_VIEW", None), getattr(sqlite3, "SQLITE_CREATE_VTABLE", None), getattr(sqlite3, "SQLITE_DROP_INDEX", None), getattr(sqlite3, "SQLITE_DROP_TABLE", None), getattr(sqlite3, "SQLITE_DROP_TEMP_INDEX", None), getattr(sqlite3, "SQLITE_DROP_TEMP_TABLE", None), getattr(sqlite3, "SQLITE_DROP_TEMP_TRIGGER", None), getattr(sqlite3, "SQLITE_DROP_TEMP_VIEW", None), getattr(sqlite3, "SQLITE_DROP_TRIGGER", None), getattr(sqlite3, "SQLITE_DROP_VIEW", None), getattr(sqlite3, "SQLITE_DROP_VTABLE", None), getattr(sqlite3, "SQLITE_REINDEX", None), getattr(sqlite3, "SQLITE_ANALYZE", None)) if action is not None)


def validate_readonly_sql(sql: str) -> str:
    normalized = sql.strip()
    segments, _ = split_sql_segments(normalized)
    if len(segments) != 1:
        raise SQLExecutionError("multiple statements are not allowed")
    lowered = normalized.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise SQLExecutionError("only read-only SELECT/CTE queries are allowed")
    return normalized


def _open_readonly_connection(database: str | Path) -> tuple[sqlite3.Connection, list[int]]:
    path = Path(database).resolve()
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only = ON")
    blocked_actions: list[int] = []
    def authorize(action: int, argument_one: str | None, argument_two: str | None, database_name: str | None, trigger_name: str | None) -> int:
        if action in _WRITE_ACTIONS:
            blocked_actions.append(action)
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    connection.set_authorizer(authorize)
    return connection, blocked_actions


@dataclass(frozen=True)
class ExecutionResult:
    columns: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]
    truncated: bool = False



def execute_readonly(database: str | Path, sql: str, *, max_rows: int = 100) -> ExecutionResult:
    normalized = validate_readonly_sql(sql)
    connection, _blocked_actions = _open_readonly_connection(database)
    try:
        cur = connection.execute(normalized)
        columns = tuple(item[0] for item in (cur.description or ()))
        rows = cur.fetchmany(max_rows + 1)
        return ExecutionResult(columns=columns, rows=tuple(tuple(row) for row in rows[:max_rows]), truncated=len(rows) > max_rows)
    except sqlite3.Error as exc:
        raise SQLExecutionError(str(exc)) from exc
    finally:
        connection.close()
