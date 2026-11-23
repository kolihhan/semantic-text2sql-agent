from __future__ import annotations

import sqlite3

import pytest

from semantic_sql.inference import _build_guarded_graph,run_guarded
from semantic_sql.contracts import SQLCandidate

SQL=(
  "SELECT COUNT(DISTINCT o.orderNumber) FROM orders o "
  "JOIN orderdetails od ON o.orderNumber = od.orderNumber "
  "WHERE o.status = 'Cancelled' GROUP BY o.orderNumber "
  "HAVING SUM(od.quantityOrdered * od.priceEach) > 4000"
)
QUESTION="How many different orders with a total price greater than 4000 are cancelled?"


class NoModel:
    def complete_text(self, *, system, user):
        raise AssertionError("Matched experiment must not call Qwen for grouped witness")


@pytest.fixture()
def physical(tmp_path):
    p=tmp_path/"db.sqlite"
    with sqlite3.connect(p) as c:
        c.executescript("""
        CREATE TABLE orders(orderNumber INTEGER PRIMARY KEY,status TEXT);
        CREATE TABLE orderdetails(id INTEGER PRIMARY KEY,orderNumber INTEGER
          REFERENCES orders(orderNumber),quantityOrdered INTEGER,priceEach INTEGER);
        INSERT INTO orders VALUES(1,'Cancelled'),(2,'Cancelled'),(3,'Shipped');
        INSERT INTO orderdetails VALUES(1,1,20,250),(2,1,1,3),(3,2,30,200),
          (4,3,30,200);
        """)
    return p


def test_native_graph_real_conditional_state_routes():
    g=_build_guarded_graph(grouped_global=True).get_graph()
    assert {"grouped_decide","grouped_select","grouped_fallback"}<=set(g.nodes)
    edges={(e.source,e.target) for e in g.edges}
    assert ("verify","grouped_decide") in edges
    assert ("grouped_decide","execute") in edges
    assert ("grouped_decide","grouped_select") in edges
    assert ("grouped_select","verify") in edges
    assert ("verify","grouped_fallback") in edges
    assert ("grouped_fallback","execute") in edges


def test_actual_guarded_vs_global_graph_semantic_repair(physical):
    opts=dict(database=physical,provider=NoModel(),question=QUESTION,
              schema_context="orders(orderNumber,status),orderdetails(orderNumber,quantityOrdered,priceEach)",
              initial_candidate=SQLCandidate(SQL),max_repairs=0,max_rows=100)
    old=run_guarded(**opts)
    new=run_guarded(**opts,grouped_global=True)
    assert old.status==new.status=="ok"
    assert old.rows==((1,),(1,))
    assert new.rows==((2,),)
    assert new.candidate.sql!=old.candidate.sql
    assert tuple(x.name for x in new.stages)==(
      "sql","verify","grouped_decide","grouped_select","verify","execute",
    )


def test_right_count_control_untouched(physical):
    right="SELECT COUNT(*) FROM ("+SQL+") AS qualified_entities"
    result=run_guarded(
      database=physical,provider=NoModel(),question=QUESTION,
      schema_context="",initial_candidate=SQLCandidate(right),
      max_repairs=0,grouped_global=True)
    assert result.rows==((2,),)
    assert result.candidate.sql==right
    assert "grouped_select" not in [s.name for s in result.stages]


def test_failed_revision_restores_original_sql(physical,monkeypatch):
    from semantic_sql import inference
    from semantic_sql.grouped_global_cardinality import GroupedScalarWitness

    def unsafe(*args):
        return GroupedScalarWitness(
          original_sql=SQL,proposed_sql="DELETE FROM orders",
          count_per_group=(1,1),question_contract="unsafe simulated proposal")
    monkeypatch.setattr(inference,"find_global_cardinality_contradiction",unsafe)
    result=run_guarded(
      database=physical,provider=NoModel(),question=QUESTION,
      schema_context="",initial_candidate=SQLCandidate(SQL),
      grouped_global=True,max_repairs=0)
    assert result.status=="ok"
    assert result.candidate.sql==SQL
    assert result.rows==((1,),(1,))
    assert "grouped_fallback" in [s.name for s in result.stages]
    with sqlite3.connect(physical) as con:
        assert con.execute("SELECT COUNT(*) FROM orders").fetchone()[0]==3


def test_mutually_exclusive_semantic_revision_and_grouped_global(physical):
    with pytest.raises(ValueError,match="mutually exclusive"):
        run_guarded(database=physical,provider=NoModel(),question=QUESTION,
                    schema_context="",initial_candidate=SQLCandidate(SQL),
                    grouped_global=True,semantic_revision=True)
