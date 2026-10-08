from __future__ import annotations

import sqlite3
from pathlib import Path

from semantic_sql.inference import run_guarded


class SequenceProvider:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.calls = 0

    def complete_text(self, system: str, user: str) -> str:
        self.calls += 1
        if not self.responses:
            raise AssertionError("unexpected model call")
        return self.responses.pop(0)


def _db(tmp_path: Path) -> Path:
    path = tmp_path / "races.sqlite"
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE races (name TEXT, year INTEGER)")
        con.execute("INSERT INTO races VALUES ('Singapore Grand Prix', 2010)")
    return path


def test_zero_row_repair_runs_once_only_after_successful_empty_execution(tmp_path: Path) -> None:
    database = _db(tmp_path)
    initial = "SELECT name FROM races WHERE name = 'Singapore' AND year = 2010"
    repaired = "SELECT name FROM races WHERE name = 'Singapore Grand Prix' AND year = 2010"
    provider = SequenceProvider(initial, repaired)

    result = run_guarded(
        database=database,
        provider=provider,
        question="Name the 2010 Singapore Grand Prix.",
        schema_context="races(name, year)",
        zero_row_repair=True,
        zero_row_evidence="Nearby database value: races.name = Singapore Grand Prix",
        max_repairs=1,
        max_rows=10,
    )

    assert result.status == "ok"
    assert result.candidate is not None
    assert result.candidate.sql == repaired
    assert result.rows == (("Singapore Grand Prix",),)
    assert provider.calls == 2
    assert [stage.name for stage in result.stages] == [
        "sql", "verify", "execute", "zero_row_repair", "verify", "execute"
    ]


def test_zero_row_repair_does_not_touch_nonempty_result(tmp_path: Path) -> None:
    database = _db(tmp_path)
    initial = "SELECT name FROM races WHERE name = 'Singapore Grand Prix' AND year = 2010"
    provider = SequenceProvider(initial)

    result = run_guarded(
        database=database,
        provider=provider,
        question="Name the 2010 Singapore Grand Prix.",
        schema_context="races(name, year)",
        zero_row_repair=True,
        zero_row_evidence="Nearby database value: races.name = Singapore Grand Prix",
        max_repairs=1,
        max_rows=10,
    )

    assert result.rows == (("Singapore Grand Prix",),)
    assert provider.calls == 1
    assert [stage.name for stage in result.stages] == ["sql", "verify", "execute"]


def test_zero_row_repair_is_off_by_default(tmp_path: Path) -> None:
    database = _db(tmp_path)
    initial = "SELECT name FROM races WHERE name = 'Singapore' AND year = 2010"
    provider = SequenceProvider(initial)

    result = run_guarded(
        database=database,
        provider=provider,
        question="Name the 2010 Singapore Grand Prix.",
        schema_context="races(name, year)",
        max_repairs=1,
        max_rows=10,
    )

    assert result.rows == ()
    assert result.candidate is not None
    assert result.candidate.sql == initial
    assert provider.calls == 1
