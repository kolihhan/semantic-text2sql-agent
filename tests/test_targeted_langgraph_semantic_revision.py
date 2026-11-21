"""Minimum CPU check for opt-in LangGraph targeted semantic revision."""
from __future__ import annotations

import sqlite3

import pytest

from semantic_sql.agent import SemanticSQLService
from semantic_sql.contracts import SQLCandidate
from semantic_sql.inference import run_guarded


ORIGINAL = "SELECT team FROM scores ORDER BY points DESC LIMIT 1"
FIXED = "SELECT team FROM scores GROUP BY team ORDER BY SUM(points) DESC LIMIT 1"
HINT = "The question asks for maximum total points per team, not one record's points."


class OneReply:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def complete_text(self, *, system, user):
        self.calls.append((system, user))
        if not self.replies:
            raise AssertionError("unexpected additional model call")
        return self.replies.pop(0)


@pytest.fixture
def database(tmp_path):
    db = tmp_path / "race.sqlite"
    with sqlite3.connect(db) as cx:
        cx.execute("CREATE TABLE scores(team TEXT, points INTEGER)")
        cx.executemany("INSERT INTO scores VALUES (?,?)", [
            ("A", 10), ("A", 10), ("B", 15),
        ])
    return db


def test_targeted_revision_reuses_existing_graph_and_corrects_grouped_total(database):
    p = OneReply(FIXED)
    result = run_guarded(
        database=database, provider=p,
        question="Which team has the maximum total points?",
        schema_context="scores(team, points)",
        initial_candidate=SQLCandidate(ORIGINAL), semantic_hint=HINT,
    )
    assert result.status == "ok"
    assert result.rows == (("A",),)
    assert result.candidate.sql == FIXED
    assert len(p.calls) == 1
    assert "Specific issue to check" in p.calls[0][1]
    assert [s.name for s in result.stages] == [
        "sql", "verify", "semantic_revision", "verify", "execute"
    ]


def test_invalid_targeted_fix_keeps_original_without_second_call(database):
    p = OneReply("UPDATE scores SET points=0")
    result = run_guarded(
        database=database, provider=p,
        question="Which team has the maximum total points?",
        schema_context="scores(team, points)",
        initial_candidate=SQLCandidate(ORIGINAL),
        semantic_hint=HINT, max_repairs=2,
    )
    assert result.status == "ok" and result.rows == (("B",),)
    assert result.candidate.sql == ORIGINAL
    assert len(p.calls) == 1
    assert any(s.summary == "targeted:invalid_fallback" for s in result.stages)


def test_correct_sql_preserved_by_model(database):
    p = OneReply(FIXED)
    result = SemanticSQLService(database, provider=p).ask(
        "Which team has the maximum total points?",
        initial_candidate=SQLCandidate(FIXED),
        semantic_hint=HINT,
    )
    assert result.rows == (("A",),)
    assert result.candidate.sql == FIXED
    assert len(p.calls) == 1
    assert any(s.summary == "targeted:same" for s in result.stages)


def test_no_hint_preserves_baseline_and_zero_model_calls(database):
    p = OneReply()
    result = SemanticSQLService(database, provider=p).ask(
        "Which team has the maximum total points?",
        initial_candidate=SQLCandidate(ORIGINAL),
    )
    assert result.rows == (("B",),)
    assert result.candidate.sql == ORIGINAL
    assert len(p.calls) == 0
    assert not any(s.name == "semantic_revision" for s in result.stages)


@pytest.mark.parametrize("bad_hint", ["", "   ", "x"*501])
def test_reject_empty_or_oversized_hints_before_model_call(database, bad_hint):
    p = OneReply()
    with pytest.raises(ValueError, match="semantic_hint"):
        run_guarded(
            database=database, provider=p, question="q", schema_context="scores",
            initial_candidate=SQLCandidate(ORIGINAL), semantic_hint=bad_hint,
        )
    assert not p.calls
