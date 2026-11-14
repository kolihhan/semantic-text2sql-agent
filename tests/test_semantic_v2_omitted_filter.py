"""Evidence-gated omitted-entity WHERE correction: no gold data or model calls."""
from __future__ import annotations

import sqlite3
from pathlib import Path
import pytest
from semantic_sql.agent import SemanticSQLService
from semantic_sql.contracts import SQLCandidate
from semantic_sql.semantic_v2 import evidence_gated_literal_revision


class NoModels:
    def complete_text(self, *, system: str, user: str) -> str:
        raise AssertionError("no language-model calls allowed for semantic evidence")


@pytest.fixture
def db(tmp_path: Path) -> Path:
    p=tmp_path/"events.sqlite"
    with sqlite3.connect(p) as con:
        con.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, name TEXT, city TEXT)")
        con.executemany(
            "INSERT INTO events VALUES (?, ?, ?)",
            [(1,"October Meeting","KL"),(2,"November Meeting","SG")],
        )
    return p


@pytest.mark.parametrize("original,question,expected",[
    ("SELECT id FROM events", 'List IDs for event "October Meeting".', ((1,),)),
    ("SELECT COUNT(*) FROM events", 'How many events for "October Meeting"?', ((1,),)),
    ("SELECT e.id FROM events AS e", 'Which event is named "October Meeting"?', ((1,),)),
])
def test_omitted_where_can_be_added_only_with_independent_database_evidence(db, original, question, expected):
    before=original
    result=SemanticSQLService(
        db,provider=NoModels(),semantic_v2=True,max_repairs=0
    ).ask(question,initial_candidate=SQLCandidate(original))
    assert result.status=="ok"
    assert result.rows==expected
    assert result.candidate is not None
    assert result.candidate.sql != before
    assert "WHERE" in result.candidate.sql
    assert "October Meeting" in result.candidate.sql
    assert "changed=true" in result.stages[2].summary
    assert result.candidate.attempt==0
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT COUNT(*) FROM events").fetchone()[0]==2


@pytest.mark.parametrize("original,question,reason",[
    ("SELECT id FROM events", 'List "October Meeting" records.',
     "quoted_entity_not_explicit_filter"),
    ("SELECT id FROM events", 'List IDs for "Missing Meeting".',
     "omitted_filter_no_unique_physical_column"),
    ("SELECT id FROM events WHERE city='SG'", 'List IDs for "October Meeting".',
     "question_value_not_verified_in_same_column"),
    ("SELECT id FROM events ORDER BY id LIMIT 1", 'List IDs for "October Meeting".',
     "omitted_filter_complex_query"),
    ("SELECT id FROM events GROUP BY id", 'List IDs for "October Meeting".',
     "omitted_filter_complex_query"),
    ("SELECT id FROM events WHERE id > 0", 'List IDs for "October Meeting".',
     "no_single_equality_filter"),
    ("SELECT id FROM events WHERE name IN ('November Meeting')",
     'List IDs for "October Meeting".',"no_single_equality_filter"),
    ("SELECT id FROM events", 'List IDs for "October Meeting" and "November Meeting".',
     "no_unambiguous_quoted_entity"),
])
def test_omitted_filter_does_not_override_ambiguous_or_existing_semantics(db,original,question,reason):
    choice=evidence_gated_literal_revision(database=db,question=question,sql=original)
    assert choice.changed is False
    assert choice.sql==original
    assert choice.reason==reason


def test_omitted_filter_ambiguous_duplicate_column_matches_fail_closed(db):
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO events VALUES (3, 'KL','KL')")
    choice=evidence_gated_literal_revision(
        database=db,question='Find events in "KL"',sql="SELECT id FROM events",
    )
    assert choice.changed is False
    assert choice.reason=="omitted_filter_no_unique_physical_column"


def test_original_default_service_unchanged_after_missing_filter_extension(db):
    result=SemanticSQLService(db,provider=NoModels(),max_repairs=0).ask(
        'List IDs for "October Meeting".', initial_candidate=SQLCandidate("SELECT id FROM events")
    )
    assert result.rows==((1,),(2,))
    assert [s.name for s in result.stages]==["sql","verify","execute"]

def test_filter_context_with_digits_does_not_break_quote_detection(db):
    sql="SELECT id FROM events"
    result=evidence_gated_literal_revision(
        database=db,
        question='List IDs for a 60-person event called "October Meeting".',
        sql=sql,
    )
    assert result.changed is True
    assert "WHERE" in result.sql
    assert "October Meeting" in result.sql
