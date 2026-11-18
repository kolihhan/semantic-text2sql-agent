"""Gold-blind read-only SQLite counterexample for a single missing string literal.

No publisher reference SQL, gold answers, model calls or trained models.
Only the natural-language question, its explicit external evidence, a fixed
model's generated SQL, and the original physical SQLite database are used.
Replace one equality literal ONLY when that exact literal does not exist in
the mapped table/column but a unique conservative NFKC/case/punctuation
equivalent DOES exist. SQL with nested SELECTs, OR, CTEs, ambiguous aliases,
multiple viable rewrites, or unsupported syntax abstains.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3
import time
import unicodedata


@dataclass(frozen=True)
class Token:
    typ: str
    val: str
    start: int
    end: int
    depth: int


@dataclass(frozen=True)
class Proposal:
    sql: str
    changed: bool
    reason: str
    base_table: str | None = None
    column: str | None = None
    source_literal: str | None = None
    replacement_literal: str | None = None


def _canonical(value: str) -> str:
    if not isinstance(value, str):
        return ""
    return "".join(c for c in unicodedata.normalize("NFKC", value).casefold() if c.isalnum())


def _tokens(sql: str) -> list[Token]:
    result = []
    i, depth, n = 0, 0, len(sql)
    while i < n:
        begin = i
        char = sql[i]
        if char.isspace():
            i += 1
            continue
        if sql.startswith("--", i):
            end = sql.find("\n", i)
            i = n if end < 0 else end
            continue
        if sql.startswith("/*", i):
            end = sql.find("*/", i + 2)
            if end < 0:
                raise ValueError("unterminated comment")
            i = end + 2
            continue
        if char == "'":
            i += 1
            value = []
            while i < n:
                if sql[i] == "'":
                    if i + 1 < n and sql[i + 1] == "'":
                        value.append("'")
                        i += 2
                        continue
                    i += 1
                    break
                value.append(sql[i])
                i += 1
            else:
                raise ValueError("unterminated literal")
            result.append(Token("string", "".join(value), begin, i, depth))
            continue
        if char in ('"', chr(96), '['):
            stop = ']' if char == '[' else char
            i += 1
            value = []
            while i < n:
                if sql[i] == stop:
                    if stop != ']' and i + 1 < n and sql[i + 1] == stop:
                        value.append(stop)
                        i += 2
                        continue
                    i += 1
                    break
                value.append(sql[i])
                i += 1
            else:
                raise ValueError("unterminated quoted identifier")
            result.append(Token("identifier", "".join(value), begin, i, depth))
            continue
        if char == '(':
            result.append(Token("symbol", "(", begin, i + 1, depth))
            depth += 1
            i += 1
            continue
        if char == ')':
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced")
            result.append(Token("symbol", ")", begin, i + 1, depth))
            i += 1
            continue
        match = re.match(r"[A-Za-z_][\w]*", sql[i:])
        if match:
            i += len(match.group())
            result.append(Token("word", match.group(), begin, i, depth))
            continue
        if char in ".=,;<>!-+*/%":
            i += 1
            result.append(Token("symbol", char, begin, i, depth))
            continue
        match = re.match(r"\d+(?:\.\d+)?", sql[i:])
        if match:
            i += len(match.group())
            result.append(Token("number", match.group(), begin, i, depth))
            continue
        raise ValueError("unsupported SQL character")
    if depth != 0:
        raise ValueError("unbalanced parentheses")
    return result


_RESERVED = {
    "ON", "WHERE", "GROUP", "ORDER", "HAVING", "LIMIT", "JOIN", "INNER", "LEFT",
    "RIGHT", "OUTER", "CROSS", "USING", "UNION", "INTERSECT", "EXCEPT", "AS",
    "INDEXED", "NOT",
}


def _analyze(sql: str, tokens: list[Token]) -> tuple[dict[str, str], int, int]:
    if not tokens or tokens[0].val.upper() != "SELECT":
        raise ValueError("not a simple select")
    if sum(t.typ == "word" and t.val.upper() == "SELECT" for t in tokens) != 1:
        raise ValueError("nested subquery unsupported")
    semi = [i for i, t in enumerate(tokens) if t.val == ";"]
    if len(semi) > 1 or (semi and semi[-1] != len(tokens) - 1):
        raise ValueError("multiple SQL statements")
    for t in tokens:
        if t.typ == "word" and t.val.upper() in {
            "UNION", "INTERSECT", "EXCEPT", "WITH", "RETURNING", "UPDATE",
            "DELETE", "INSERT", "REPLACE",
        }:
            raise ValueError("compound SQL unsupported")
    aliases = {}
    for i, t in enumerate(tokens):
        if t.depth != 0 or t.typ != "word" or t.val.upper() not in {"FROM", "JOIN"}:
            continue
        if i + 1 >= len(tokens) or tokens[i + 1].typ not in {"identifier", "word"}:
            raise ValueError("derived table unsupported")
        table = tokens[i + 1].val
        if i + 2 < len(tokens) and tokens[i + 2].val == ".":
            raise ValueError("schema-qualified tables unsupported")
        aliases[table.casefold()] = table
        j = i + 2
        if j < len(tokens) and tokens[j].typ == "word" and tokens[j].val.upper() == "AS":
            j += 1
        if j < len(tokens) and tokens[j].typ in {"identifier", "word"} and tokens[j].val.upper() not in _RESERVED:
            aliases[tokens[j].val.casefold()] = table
    if not aliases:
        raise ValueError("missing physical table")
    where = next((i for i, t in enumerate(tokens)
                  if t.typ == "word" and t.val.upper() == "WHERE" and t.depth == 0), None)
    if where is None:
        raise ValueError("no WHERE")
    end = next((i for i in range(where + 1, len(tokens))
                if tokens[i].typ == "word" and tokens[i].depth == 0
                and tokens[i].val.upper() in {"GROUP", "ORDER", "LIMIT", "HAVING", "UNION"}),
               len(tokens))
    if any(t.typ == "word" and t.val.upper() == "OR" for t in tokens[where + 1:end]):
        raise ValueError("OR predicate unsupported")
    return aliases, where, end


def _id(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _exec_bounded(con: sqlite3.Connection, sql: str, params: tuple,
                  seconds: float = 1.5) -> list[tuple]:
    stop = time.monotonic() + seconds
    con.set_progress_handler(lambda: 1 if time.monotonic() > stop else 0, 2000)
    try:
        return con.execute(sql, params).fetchall()
    finally:
        con.set_progress_handler(None, 0)


def _matching_value(con: sqlite3.Connection, table: str, column: str,
                    search: str) -> tuple[str | None, str]:
    if _exec_bounded(
        con, f'SELECT 1 FROM {_id(table)} WHERE {_id(column)} = ? LIMIT 1',
        (search,)
    ):
        return None, "literal_exists"
    target = _canonical(search)
    if len(target) < 5:
        return None, "too_short"
    matches = _exec_bounded(
        con,
        f'SELECT DISTINCT CAST({_id(column)} AS TEXT) FROM {_id(table)} '
        f"WHERE typeof({_id(column)}) = 'text' AND canonical_text({_id(column)}) = ? LIMIT 3",
        (target,),
    )
    values = [r[0] for r in matches]
    if len(values) != 1:
        return None, "ambiguous_or_missing"
    if values[0] == search:
        return None, "literal_exists"
    return values[0], "unique_physical_equivalent"


def propose_literal_counterexample(database: Path, user_prompt: str,
                                    sql: str) -> Proposal:
    """Propose an alternative only from physical literal evidence, never gold."""
    unchanged = lambda reason: Proposal(sql, False, reason)
    try:
        toks = _tokens(sql)
        aliases, where, end = _analyze(sql, toks)
        tables = sorted(set(aliases.values()))
        question = user_prompt.split("\nSchema context:\n", 1)[0].removeprefix("Question: ")
        evidence = (user_prompt.split("\nExternal evidence:\n", 1)[1]
                    if "\nExternal evidence:\n" in user_prompt else "")
        source_canonical = _canonical(question + " " + evidence)
        filters = []
        for i in range(where + 1, end - 1):
            if toks[i].val != "=" or toks[i + 1].typ != "string":
                continue
            position = i - 1
            if position <= where or toks[position].typ not in {"word", "identifier"}:
                continue
            qualifier = None
            if (position >= where + 3 and toks[position - 1].val == "."
                    and toks[position - 2].typ in {"word", "identifier"}):
                qualifier = toks[position - 2].val
            elif position >= where + 2 and toks[position - 1].val == ".":
                continue
            literal = toks[i + 1]
            key = _canonical(literal.val)
            if len(key) < 5 or key not in source_canonical:
                continue
            filters.append((qualifier, toks[position].val, literal))
        if not filters:
            return unchanged("no_explicit_question_literal_equality")
        if len(filters) > 4:
            return unchanged("too_many_literal_filters")
        uri = "file:" + database.resolve().as_posix() + "?mode=ro"
        with sqlite3.connect(uri, uri=True) as con:
            con.execute("PRAGMA query_only=ON")
            con.create_function("canonical_text", 1, _canonical, deterministic=True)
            catalog = {}
            for table in tables:
                columns = [r[1] for r in con.execute(f'PRAGMA table_info({_id(table)})')]
                if not columns:
                    return unchanged("unknown_table")
                catalog[table.casefold()] = {s.casefold(): s for s in columns}
            try:
                con.execute("EXPLAIN QUERY PLAN " + sql).fetchall()
            except sqlite3.Error:
                return unchanged("original_SQL_not_compilable")
            proposals = []
            for alias, column, literal in filters:
                if alias:
                    table = aliases.get(alias.casefold())
                    if table is None:
                        return unchanged("unknown_alias")
                    physical_column = catalog[table.casefold()].get(column.casefold())
                    if not physical_column:
                        return unchanged("unknown_qualified_column")
                else:
                    matches = [(table, cols[column.casefold()])
                               for table, cols in ((t, catalog[t.casefold()]) for t in tables)
                               if column.casefold() in cols]
                    if len(matches) != 1:
                        return unchanged("ambiguous_unqualified_column")
                    table, physical_column = matches[0]
                replacement, why = _matching_value(
                    con, table, physical_column, literal.val
                )
                if why == "literal_exists":
                    continue
                if replacement is None:
                    return unchanged(why)
                proposals.append((table, physical_column, literal, replacement))
            if len(proposals) != 1:
                return unchanged("no_or_multiple_unique_counterexamples")
            table, col, literal, replacement = proposals[0]
            rewritten = (sql[:literal.start] + "'" + replacement.replace("'", "''")
                         + "'" + sql[literal.end:])
            try:
                con.execute("EXPLAIN QUERY PLAN " + rewritten).fetchall()
            except sqlite3.Error:
                return unchanged("rewritten_SQL_not_compilable")
            return Proposal(
                rewritten, True, "unique_physical_normalized_equality_counterexample",
                table, col, literal.val, replacement,
            )
    except (ValueError, sqlite3.Error, TimeoutError, OSError) as exc:
        return unchanged("fail_closed_" + type(exc).__name__)
