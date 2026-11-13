"""Independent semantic count projection tests, no LLM or gold answers."""
from __future__ import annotations

import sqlite3
from pathlib import Path
import pytest

from semantic_sql.semantic_shape_v3 import evidence_gated_count_projection
from semantic_sql.agent import SemanticSQLService
from semantic_sql.contracts import SQLCandidate


class NoModels:
    def complete_text(self,*,system:str,user:str):
        raise AssertionError("v3 must not call an additional LLM")


@pytest.fixture
def database(tmp_path:Path)->Path:
    db=tmp_path/"events.sqlite"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE events(id INTEGER PRIMARY KEY, name TEXT, active INTEGER)")
        con.executemany("INSERT INTO events VALUES (?,?,?)",[
            (1,"ABC",1),(2,"DEF",1),(3,"GHI",0),
        ])
    return db


@pytest.mark.parametrize("question,original,expected",[
    ("How many events happened?", "SELECT name FROM events", ((3,),)),
    ("How many events have active = 1?","SELECT id FROM events WHERE active=1",((2,),)),
    ("What is the number of events?", "SELECT * FROM events",((3,),)),
    ("How many active events exist?", "SELECT id FROM events WHERE active = 1",((2,),)),
    ("How many events have active=1?", "SELECT e.name FROM events AS e WHERE e.active=1",((2,),)),
])
def test_count_grain_fix_changes_projections_only(database,question,original,expected):
    d=evidence_gated_count_projection(database=database,question=question,sql=original)
    assert d.changed is True, d.reason
    assert d.sql.upper().count("COUNT(*)")==1
    assert "WHERE" in d.sql if "WHERE" in original else "WHERE" not in d.sql
    with sqlite3.connect(database) as con:
        assert con.execute(d.sql).fetchall()==list(expected)
        assert con.execute("SELECT COUNT(*) FROM events").fetchone()[0]==3


@pytest.mark.parametrize("question,original,reason",[
    ("How many people attended events?","SELECT name FROM events","question_count_unit_not_physical_table"),
    ("How many unique events happened?","SELECT name FROM events","count_intent_requires_distinct_or_complex_grain"),
    ("How many events per person?","SELECT name FROM events","count_intent_requires_distinct_or_complex_grain"),
    ("How many events by year?","SELECT name FROM events","count_intent_requires_distinct_or_complex_grain"),
    ("List all events.","SELECT name FROM events","no_unambiguous_count_intent"),
    ("How many events happened?","SELECT COUNT(*) FROM events","sql_aggregation_grain_unsafe"),
    ("How many events happened?","SELECT name FROM events GROUP BY name","sql_aggregation_grain_unsafe"),
    ("How many events happened?","SELECT DISTINCT name FROM events","sql_aggregation_grain_unsafe"),
    ("How many events happened?","SELECT e.name FROM events e JOIN events a ON a.id=e.id","sql_aggregation_grain_unsafe"),
    ("How many events happened?","SELECT name FROM events ORDER BY id LIMIT 2","sql_aggregation_grain_unsafe"),
    ("How many events happened?","SELECT name,active FROM events","multiple_output_columns"),
    ("How many events happened?","SELECT MAX(id) FROM events","sql_aggregation_grain_unsafe"),
    ("How many events happened?","SELECT LOWER(name) FROM events","sql_aggregation_grain_unsafe"),
])
def test_count_grain_fail_closed(database,question,original,reason):
    d=evidence_gated_count_projection(database=database,question=question,sql=original)
    assert d.changed is False
    assert d.sql==original
    assert d.reason==reason


def test_no_model_needed_when_v3_enabled(database):
    result=SemanticSQLService(
        database,provider=NoModels(),max_repairs=0,
        semantic_shape_v3=True
    ).ask("How many events happened?",
          initial_candidate=SQLCandidate("SELECT name FROM events"))
    assert result.status=="ok"
    assert result.rows==((3,),)
    assert result.candidate is not None and "COUNT(*)" in result.candidate.sql
    assert "semantic_shape_v3" in [x.name for x in result.stages]
    assert "changed=true" in next(x.summary for x in result.stages if x.name=="semantic_shape_v3")


def test_default_system_keeps_original_sql_and_rows(database):
    original="SELECT name FROM events"
    result=SemanticSQLService(
        database,provider=NoModels(),max_repairs=0
    ).ask("How many events happened?",initial_candidate=SQLCandidate(original))
    assert result.candidate is not None and result.candidate.sql==original
    assert result.rows==(("ABC",),("DEF",),("GHI",))
    assert [x.name for x in result.stages]==["sql","verify","execute"]
