from __future__ import annotations

import sqlite3

import pytest

from semantic_sql.contracts import SQLCandidate
from semantic_sql.inference import _build_guarded_graph, run_guarded
from semantic_sql.execution import execute_readonly


SQL = ("SELECT COUNT(DISTINCT u.id) FROM users AS u JOIN badges AS b "
       "ON u.id = b.user_id WHERE b.role IN ('Teacher', 'Supporter')")


class NeverCall:
    def complete_text(self, system, user):
        raise AssertionError("no model calls allowed for matched replay")


@pytest.fixture()
def database(tmp_path):
    p=tmp_path/"fixture.sqlite"
    with sqlite3.connect(p) as con:
        con.executescript("""
        CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE badges(id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users(id), role TEXT);
        INSERT INTO users VALUES(1,'one'),(2,'two'),(3,'three');
        INSERT INTO badges VALUES(1,1,'Teacher'),(2,1,'Supporter'),
                                 (3,2,'Teacher'),(4,3,'Supporter');
        """)
    return p


def test_graph_has_real_stateful_conditional_edges():
    graph=_build_guarded_graph(allof_counterexample=True).get_graph()
    names=set(graph.nodes)
    assert {"allof_decide","allof_rewrite","allof_fallback"} <= names
    paths={(e.source,e.target) for e in graph.edges}
    assert ("verify","allof_decide") in paths
    assert ("allof_decide","allof_rewrite") in paths
    assert ("allof_decide","execute") in paths
    assert ("allof_rewrite","verify") in paths
    assert ("allof_fallback","execute") in paths


def test_genuine_physical_semantic_repair_without_extra_qwen(database):
    control=run_guarded(
        database=database,provider=NeverCall(),
        question="How many users have both Teacher and Supporter roles?",
        schema_context="users(id), badges(user_id,role)",
        initial_candidate=SQLCandidate(SQL),max_repairs=0,
    )
    treated=run_guarded(
        database=database,provider=NeverCall(),
        question="How many users have both Teacher and Supporter roles?",
        schema_context="users(id), badges(user_id,role)",
        initial_candidate=SQLCandidate(SQL),max_repairs=0,
        allof_counterexample=True,
    )
    assert control.rows==((3,),)
    assert treated.rows==((1,),)
    assert treated.status=="ok"
    assert treated.candidate.sql!=SQL
    assert tuple(s.name for s in treated.stages)==(
        "sql","verify","allof_decide","allof_rewrite","verify","execute")
    assert "union=3:intersection=1" in treated.stages[2].summary


def test_correct_original_preserved_and_no_extra_llm(database):
    candidate=SQL.replace("COUNT(DISTINCT u.id)","COUNT(DISTINCT b.user_id)")
    result=run_guarded(
        database=database,provider=NeverCall(),
        question="How many users have either Teacher or Supporter roles?",
        schema_context="users(id), badges(user_id,role)",
        initial_candidate=SQLCandidate(candidate),
        allof_counterexample=True,max_repairs=0,
    )
    assert result.candidate.sql==candidate
    assert result.rows==((3,),)
    assert "allof_rewrite" not in [x.name for x in result.stages]


def test_default_baseline_graph_unchanged():
    nodes=set(_build_guarded_graph().get_graph().nodes)
    assert "allof_decide" not in nodes and "allof_rewrite" not in nodes


def test_mutually_exclusive_experimental_modes(database):
    with pytest.raises(ValueError,match="mutually exclusive"):
        run_guarded(
            database=database,provider=NeverCall(),question="q",
            schema_context="schema",initial_candidate=SQLCandidate(SQL),
            allof_counterexample=True,semantic_revision=True,
        )
