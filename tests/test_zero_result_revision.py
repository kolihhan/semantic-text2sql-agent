from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from semantic_sql.contracts import SQLCandidate
from semantic_sql.inference import run_guarded_zero_result_revision


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
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE schools (id INTEGER PRIMARY KEY, county TEXT NOT NULL)')
        connection.execute('INSERT INTO schools (id, county) VALUES (1, "Alpha")')
    return path


def test_zero_result_filter_revision_accepts_one_nonempty_filter_fix(database: Path) -> None:
    initial = "SELECT county FROM schools WHERE county = 'Beta'"
    revised = "SELECT county FROM schools WHERE county = 'Alpha'"
    provider = SequenceProvider(
        '{"changed": true, "issue_type": "FILTER_VALUE", '
        '"sql": "SELECT county FROM schools WHERE county = \'Alpha\'"}'
    )

    result = run_guarded_zero_result_revision(
        database=database,
        provider=provider,
        question="Return the county named Alpha.",
        schema_context="schools(id, county)",
        initial_candidate=SQLCandidate(initial),
        max_rows=10,
    )

    assert result.status == "ok"
    assert result.candidate is not None
    assert result.candidate.sql == revised
    assert result.rows == (("Alpha",),)
    assert provider.text_calls == 1
    assert any(stage.name == "empty_result_revision" and "accepted" in stage.summary for stage in result.stages)


def test_nonempty_result_never_calls_semantic_reviewer(database: Path) -> None:
    initial = "SELECT county FROM schools"
    provider = SequenceProvider()

    result = run_guarded_zero_result_revision(
        database=database,
        provider=provider,
        question="Return the counties.",
        schema_context="schools(id, county)",
        initial_candidate=SQLCandidate(initial),
        max_rows=10,
    )

    assert result.rows == (("Alpha",),)
    assert result.candidate is not None and result.candidate.sql == initial
    assert provider.text_calls == 0


def test_zero_result_rejects_non_filter_semantic_change(database: Path) -> None:
    initial = "SELECT county FROM schools WHERE county = 'Beta'"
    provider = SequenceProvider(
        '{"changed": true, "issue_type": "PROJECTION", "sql": "SELECT id FROM schools"}'
    )

    result = run_guarded_zero_result_revision(
        database=database,
        provider=provider,
        question="Return the county named Alpha.",
        schema_context="schools(id, county)",
        initial_candidate=SQLCandidate(initial),
        max_rows=10,
    )

    assert result.rows == ()
    assert result.candidate is not None and result.candidate.sql == initial
    assert provider.text_calls == 1


def test_zero_result_rejects_filter_revision_that_stays_empty(database: Path) -> None:
    initial = "SELECT county FROM schools WHERE county = 'Beta'"
    provider = SequenceProvider(
        '{"changed": true, "issue_type": "FILTER_VALUE", '
        '"sql": "SELECT county FROM schools WHERE county = \'Gamma\'"}'
    )

    result = run_guarded_zero_result_revision(
        database=database,
        provider=provider,
        question="Return the county named Alpha.",
        schema_context="schools(id, county)",
        initial_candidate=SQLCandidate(initial),
        max_rows=10,
    )

    assert result.rows == ()
    assert result.candidate is not None and result.candidate.sql == initial
    assert provider.text_calls == 1
