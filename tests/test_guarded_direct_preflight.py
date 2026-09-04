from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import semantic_sql.inference as inference
import semantic_sql.verifier as verifier


MULTI_SQL_WITH_PROSE = (
    'SELECT id FROM schools;\n'
    'SELECT "County Name" FROM schools;\n'
    "The first query returns identifiers."
)


class SequenceProvider:
    def __init__(self, *responses: str) -> None:
        self._responses = list(responses)
        self.text_calls = 0
    def complete_text(self, system: str, user: str) -> str:
        self.text_calls += 1
        if not self._responses:
            raise AssertionError("unexpected additional text-model call")
        return self._responses.pop(0)


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "catalog.sqlite"
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            'CREATE TABLE schools (id INTEGER PRIMARY KEY, "County Name" TEXT NOT NULL)'
        )
        connection.execute(
            'INSERT INTO schools (id, "County Name") VALUES (1, "Alpha")'
        )
        connection.commit()
    finally:
        connection.close()
    return path


def issue_codes(result) -> set[str]:
    return {issue.code for issue in result.issues}


def test_preflight_rejects_multiple_sql_statements_and_prose(database: Path) -> None:
    result = verifier.verify_sql_preflight(database, MULTI_SQL_WITH_PROSE)
    assert result.ok is False
    assert {"multiple_statements", "non_sql_text"} <= issue_codes(result)


def test_preflight_rejects_invalid_alias_and_unquoted_identifier(database: Path) -> None:
    result = verifier.verify_sql_preflight(
        database,
        "SELECT s.id, County Name FROM schools",
    )
    assert result.ok is False
    assert "sqlite_compile_error" in issue_codes(result)


def test_preflight_rejects_data_modifying_cte_without_side_effects(database: Path) -> None:
    result = verifier.verify_sql_preflight(
        database,
        (
            "WITH selected AS (SELECT id FROM schools) "
            "DELETE FROM schools "
            "WHERE id IN (SELECT id FROM selected)"
        ),
    )
    assert result.ok is False
    assert "not_readonly" in issue_codes(result)
    with sqlite3.connect(database) as connection:
        remaining = connection.execute("SELECT COUNT(*) FROM schools").fetchone()[0]
    assert remaining == 1


@pytest.mark.parametrize(
    "sql",
    [
        (
            'WITH selected AS (SELECT "County Name" FROM schools) '
            'SELECT "County Name" FROM selected'
        ),
        "SELECT s.id FROM schools AS s",
        'SELECT "County Name" FROM schools',
    ],
)
def test_preflight_accepts_valid_sql(database: Path, sql: str) -> None:
    result = verifier.verify_sql_preflight(database, sql)
    assert result.ok is True
    assert result.issues == ()


def test_guarded_graph_has_only_reliability_nodes() -> None:
    rendered = inference._build_guarded_graph().get_graph()
    assert set(rendered.nodes) == {
        "__start__", "generate", "verify", "repair",
        "execute", "refuse", "__end__",
    }
    assert {(edge.source, edge.target) for edge in rendered.edges} == {
        ("__start__", "generate"),
        ("generate", "verify"),
        ("verify", "execute"),
        ("verify", "repair"),
        ("verify", "refuse"),
        ("repair", "verify"),
        ("execute", "__end__"),
        ("refuse", "__end__"),
    }


def test_guarded_direct_repairs_before_executing(
    database: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rejected = "SELECT s.id, County Name FROM schools"
    repaired = 'SELECT "County Name" FROM schools'
    provider = SequenceProvider(rejected, repaired)
    executed: list[str] = []
    real_execute = inference.execute_readonly

    def observe_execute(database, sql: str, *, max_rows: int):
        executed.append(sql)
        return real_execute(database, sql, max_rows=max_rows)

    monkeypatch.setattr(inference, "execute_readonly", observe_execute)
    result = inference.run_guarded(
        database=database,
        provider=provider,
        question="Return the county name.",
        schema_context='schools(id, "County Name")',
        max_repairs=1,
        max_rows=10,
    )
    assert result.status == "ok"
    assert result.candidate is not None
    assert result.candidate.sql == repaired
    assert result.candidate.attempt == 1
    assert result.verification is not None
    assert result.verification.ok is True
    assert result.rows == (("Alpha",),)
    assert tuple(stage.name for stage in result.stages) == (
        "sql", "verify", "repair", "verify", "execute",
    )
    assert provider.text_calls == 2
    assert executed == [repaired]


def test_guarded_direct_refuses_rejected_sql_at_zero_budget(
    database: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = SequenceProvider(MULTI_SQL_WITH_PROSE)

    def unexpected_execute(*args, **kwargs):
        raise AssertionError("verifier-rejected SQL reached execution")

    monkeypatch.setattr(inference, "execute_readonly", unexpected_execute)
    result = inference.run_guarded(
        database=database,
        provider=provider,
        question="Return school information.",
        schema_context='schools(id, "County Name")',
        max_repairs=0,
        max_rows=10,
    )
    assert result.status == "verification_failed"
    assert result.candidate is not None
    assert result.candidate.attempt == 0
    assert result.verification is not None
    assert result.verification.ok is False
    assert result.rows == ()
    assert tuple(stage.name for stage in result.stages) == ("sql", "verify")
    assert provider.text_calls == 1


def test_readonly_execution_allows_semicolon_inside_string_literal(tmp_path):
    import sqlite3
    from semantic_sql.execution import execute_readonly

    database = tmp_path / "db.sqlite"
    with sqlite3.connect(database) as con:
        con.execute("CREATE TABLE items (value TEXT)")
        con.execute("INSERT INTO items VALUES (';')")
    result = execute_readonly(database, "SELECT value FROM items WHERE value = ';'")
    assert result.rows == ((";",),)


def test_readonly_execution_still_rejects_multiple_statements(tmp_path):
    import sqlite3
    import pytest
    from semantic_sql.execution import SQLExecutionError, execute_readonly

    database = tmp_path / "db.sqlite"
    with sqlite3.connect(database) as con:
        con.execute("CREATE TABLE items (value TEXT)")
    with pytest.raises(SQLExecutionError, match="multiple statements"):
        execute_readonly(database, "SELECT 1; SELECT 2")
