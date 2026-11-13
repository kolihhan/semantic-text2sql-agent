"""Semantic v2 is opt-in, read-only, and must preserve the original on uncertainty."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from semantic_sql.agent import SemanticSQLService
from semantic_sql.contracts import SQLCandidate
from semantic_sql.inference import _build_guarded_graph, run_guarded
from semantic_sql.semantic_v2 import evidence_gated_literal_revision


class NoCallsProvider:
    def complete_text(self, *, system: str, user: str) -> str:
        raise AssertionError("Semantic v2 should not invoke a model for frozen SQL")


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "events.sqlite"
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, name TEXT UNIQUE, city TEXT)")
        con.executemany(
            "INSERT INTO events (id, name, city) VALUES (?, ?, ?)",
            [(1, "October Meeting", "KL"), (2, "November Meeting", "SG")],
        )
    return path


def test_literal_mismatch_can_be_corrected_without_new_model_call(database: Path):
    original = "SELECT id FROM events WHERE name = 'November Meeting'"
    result = run_guarded(
        database=database, provider=NoCallsProvider(),
        question='List IDs for the event "October Meeting".',
        schema_context="events(id, name, city)",
        initial_candidate=SQLCandidate(original),
        semantic_v2=True,
        max_repairs=0,
    )
    assert result.status == "ok"
    assert result.rows == ((1,),)
    assert result.candidate is not None
    assert "October Meeting" in result.candidate.sql
    assert result.candidate.attempt == 0
    assert [s.name for s in result.stages] == [
        "sql", "verify", "semantic_v2", "verify", "execute"
    ]
    assert "changed=true" in result.stages[2].summary
    with sqlite3.connect(database) as con:
        assert con.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 2


def test_synthetic_counterexample_can_distinguish_same_count_on_real_db(database: Path):
    old = "SELECT COUNT(*) FROM events WHERE name = 'November Meeting'"
    result = run_guarded(
        database=database, provider=NoCallsProvider(),
        question="How many were in 'October Meeting'?",
        schema_context="events(id, name, city)",
        initial_candidate=SQLCandidate(old),
        semantic_v2=True, max_repairs=0,
    )
    assert result.status == "ok"
    assert result.rows == ((1,),)
    assert result.candidate is not None
    assert "October Meeting" in result.candidate.sql
    assert "changed=true" in result.stages[2].summary


@pytest.mark.parametrize("question,sql,reason", [
    ('List "October Meeting" records.',
     "SELECT id FROM events WHERE name = 'October Meeting'",
     "entity_already_matches_question"),
    ('List "October Meeting" and "November Meeting" records.',
     "SELECT id FROM events WHERE name = 'November Meeting'",
     "no_unambiguous_quoted_entity"),
    ('List "Unknown Event" records.',
     "SELECT id FROM events WHERE name = 'November Meeting'",
     "question_value_not_verified_in_same_column"),
    ('List event "October Meeting".',
     "SELECT id FROM events WHERE city = 'SG'",
     "question_value_not_verified_in_same_column"),
    ('List event "October Meeting".',
     "SELECT id FROM events WHERE name='November Meeting' AND id=2",
     "no_single_equality_filter"),
    ('List event "October Meeting".',
     "SELECT e.id FROM events AS e JOIN events e2 ON e.id=e2.id WHERE e.name='November Meeting'",
     "join_cte_or_nested_query"),
    ('List event "October Meeting".',
     "SELECT id FROM events WHERE name IN ('November Meeting')",
     "no_single_equality_filter"),
    ('List event "October Meeting".',
     "SELECT id FROM events WHERE name='November Meeting' ORDER BY id LIMIT 1",
     "complex_projection_or_order"),
    ('List event "October Meeting".',
     "SELECT id FROM events WHERE name='November Meeting' OR city='SG'",
     "no_single_equality_filter"),
])
def test_unsupported_or_unverifiable_semantics_keep_original(database: Path, question: str, sql: str, reason: str):
    decision = evidence_gated_literal_revision(database=database, question=question, sql=sql)
    assert decision.changed is False
    assert decision.sql == sql
    assert decision.reason == reason


def test_default_path_is_unchanged_and_does_not_depend_on_sqlglot(database: Path):
    original = "SELECT id FROM events WHERE name = 'November Meeting'"
    result = SemanticSQLService(
        database, provider=NoCallsProvider(), max_repairs=0,
    ).ask('List IDs for the event "October Meeting".', initial_candidate=SQLCandidate(original))
    assert result.candidate == SQLCandidate(original)
    assert result.rows == ((2,),)
    assert [s.name for s in result.stages] == ["sql", "verify", "execute"]


def test_service_opt_in_has_same_behavior_as_graph(database: Path):
    original = "SELECT id FROM events WHERE name = 'November Meeting'"
    result = SemanticSQLService(
        database, provider=NoCallsProvider(), semantic_v2=True, max_repairs=0,
    ).ask('List IDs for the event "October Meeting".', initial_candidate=SQLCandidate(original))
    assert result.rows == ((1,),)
    assert result.candidate is not None and "October Meeting" in result.candidate.sql


def test_no_combining_old_unproven_reviewer_and_new_gate():
    with pytest.raises(ValueError, match="cannot run together"):
        _build_guarded_graph(semantic_revision=True, semantic_v2=True)


def test_no_join_or_gold_labels_used_in_counterexample(database: Path):
    q='Which event was "October Meeting"?'
    old = "SELECT id FROM events WHERE name = 'November Meeting'"
    dec = evidence_gated_literal_revision(database=database, question=q, sql=old)
    assert dec.changed
    assert dec.evidence == (
        "question_contains_exactly_one_quoted_value",
        "physical_column_contains_requested_value",
        "single_ast_equality_literal_changed",
        "synthetic_in_memory_counterexample_discriminates_outputs",
        "revised_sql_passes_readonly_preflight",
    )
    assert dec.reason == "single_literal_question_and_physical_value_evidence"
