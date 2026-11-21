"""Research-only, owner-optional one-call semantic safety net tests.

No Qwen inference. Use synthetic SQLite data to test fallbacks, a zero-repair
budget, no extra calls for unflagged SQL and unchanged syntax repair.
"""
from __future__ import annotations

import sqlite3

import pytest
from semantic_sql.inference import run_guarded, _build_guarded_graph


QUESTION = "What is the growth rate of total loans between 1996 and 1997?"
INITIAL = (
    "SELECT SUM(CASE WHEN yr=1997 THEN amount ELSE 0 END) AS numerator, "
    "SUM(CASE WHEN yr=1996 THEN amount ELSE 0 END) AS denominator FROM loans"
)
GOOD = (
    "SELECT (SUM(CASE WHEN yr=1997 THEN amount ELSE 0 END) - "
    "SUM(CASE WHEN yr=1996 THEN amount ELSE 0 END)) * 1.0 "
    "/ NULLIF(SUM(CASE WHEN yr=1996 THEN amount ELSE 0 END),0) "
    "AS growth_rate FROM loans"
)
STILL_WRONG = (
    "SELECT SUM(CASE WHEN yr=1997 THEN amount ELSE 0 END) AS new_amount, "
    "SUM(CASE WHEN yr=1996 THEN amount ELSE 0 END) AS old_amount FROM loans"
)
MALFORMED = "SELECT nonexistent_column FROM loans"


class CountProvider:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0

    def complete_text(self, *, system, user):
        self.calls += 1
        if not self.replies:
            raise AssertionError("budget overrun: unexpected model call")
        return self.replies.pop(0)


@pytest.fixture
def database(tmp_path):
    db = tmp_path / "loans.sqlite"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE loans(yr INTEGER, amount REAL)")
        con.executemany(
            "INSERT INTO loans VALUES (?,?)", [(1996, 100.0), (1997, 150.0)]
        )
    return db


def run(db, provider, question=QUESTION, max_repairs=2, **kw):
    return run_guarded(
        database=db, provider=provider, question=question,
        schema_context="loans(yr, amount)", max_repairs=max_repairs,
        semantic_contract_checks=True, semantic_cost_guard=True, **kw,
    )


def test_one_successful_semantic_repair_is_one_extra_call(database):
    p = CountProvider(INITIAL, GOOD)
    r = run(database, p)
    assert p.calls == 2
    assert r.status == "ok" and r.rows == ((0.5,),)
    assert r.candidate.sql == GOOD
    assert "semantic_fallback" not in [x.name for x in r.stages]


@pytest.mark.parametrize("bad", [STILL_WRONG, MALFORMED])
def test_failed_or_rejected_semantic_repair_falls_back_original(database, bad):
    p = CountProvider(INITIAL, bad)
    r = run(database, p)
    assert p.calls == 2
    assert r.status == "ok" and r.candidate.sql == INITIAL
    assert r.rows == ((150.0, 100.0),)
    assert r.verification.ok
    assert "semantic_fallback" in [x.name for x in r.stages]
    assert r.candidate.attempt == 0


def test_zero_repair_budget_keeps_original_without_another_call(database):
    p = CountProvider(INITIAL)
    r = run(database, p, max_repairs=0)
    assert p.calls == 1
    assert r.status == "ok" and r.candidate.sql == INITIAL
    assert r.rows == ((150.0, 100.0),)
    assert "semantic_fallback" in [x.name for x in r.stages]


def test_correct_non_trigger_keeps_original_with_no_extra_calls(database):
    p = CountProvider(GOOD)
    r = run(database, p)
    assert p.calls == 1
    assert r.status == "ok" and r.rows == ((0.5,),)
    assert all(x.name != "semantic_fallback" for x in r.stages)


def test_syntax_failure_still_uses_original_bounded_repair(database):
    p = CountProvider("SELECT nonexistent FROM loans", "SELECT amount FROM loans")
    r = run(database, p, question="List amounts.", max_repairs=2)
    assert p.calls == 2
    assert r.status == "ok" and r.candidate.attempt == 1
    assert "semantic_fallback" not in [x.name for x in r.stages]


def test_legacy_default_graph_and_invalid_combination():
    nodes = set(_build_guarded_graph().get_graph().nodes)
    assert "semantic_fallback" not in nodes
    assert "semantic_fallback" in _build_guarded_graph(semantic_cost_guard=True).get_graph().nodes
    with pytest.raises(ValueError):
        run_guarded(database="unused.sqlite", provider=CountProvider(),
                    question="foo", schema_context="", semantic_cost_guard=True)
    with pytest.raises(ValueError):
        run_guarded(database="unused.sqlite", provider=CountProvider(),
                    question="foo", schema_context="", semantic_contract_checks=True,
                    semantic_cost_guard=True, semantic_revision=True)
