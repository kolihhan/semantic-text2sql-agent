from __future__ import annotations

import sqlite3

import pytest

from semantic_sql.agent import SemanticSQLService
from semantic_sql.inference import _build_guarded_graph, run_guarded
from semantic_sql.adaptive_probe import semantic_risk, schema_probe, data_probe


class SequenceQwenStub:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
    def complete_text(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if not self.responses:
            raise AssertionError("unexpected Qwen call")
        return self.responses.pop(0)


@pytest.fixture()
def physical_db(tmp_path):
    path = tmp_path / "physical-bird-like.sqlite"
    with sqlite3.connect(path) as con:
        con.executescript("""
        PRAGMA foreign_keys=ON;
        CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE orders(id INTEGER PRIMARY KEY,
            customer_id INTEGER REFERENCES customers(id), amount INTEGER);
        INSERT INTO customers VALUES(1,'one'),(2,'two'),(3,'three');
        INSERT INTO orders VALUES(1,1,4),(2,1,7),(3,2,10);
        """)
    return path


BAD = "SELECT COUNT(*) FROM customers JOIN orders ON customers.id=orders.customer_id"
FIXED = "SELECT COUNT(DISTINCT customers.id) FROM customers JOIN orders ON customers.id=orders.customer_id"
SCHEMA = "customers(id, name)\norders(id, customer_id, amount)\nFK orders.customer_id -> customers.id"


def test_graph_has_real_conditional_data_and_schema_routes():
    g = _build_guarded_graph(adaptive_physical=True).get_graph()
    names = set(g.nodes)
    assert {"adaptive_decide", "adaptive_schema_probe", "adaptive_data_probe",
            "adaptive_repair", "adaptive_validate", "adaptive_fallback"} <= names
    edges = {(e.source, e.target) for e in g.edges}
    assert ("adaptive_decide", "execute") in edges
    assert ("adaptive_decide", "adaptive_schema_probe") in edges
    assert ("adaptive_schema_probe", "execute") in edges
    assert ("adaptive_schema_probe", "adaptive_data_probe") in edges
    assert ("adaptive_data_probe", "adaptive_repair") in edges
    assert ("adaptive_validate", "adaptive_fallback") in edges


def test_count_join_risk_routes_schema_then_data_then_one_repair(physical_db):
    qwen = SequenceQwenStub(BAD, FIXED)
    result = run_guarded(
        database=physical_db, provider=qwen,
        question="How many customers have orders?", schema_context=SCHEMA,
        adaptive_physical=True,
    )
    assert result.status == "ok"
    assert result.rows == ((2,),)
    assert result.candidate.sql == FIXED
    stages = [s.name for s in result.stages]
    assert stages == ["sql", "verify", "adaptive_decide", "adaptive_schema_probe",
                      "adaptive_data_probe", "adaptive_repair", "adaptive_validate", "execute"]
    assert len(qwen.calls) == 2
    assert "orders: physical foreign keys" in qwen.calls[1][1]
    assert "orders: sample" in qwen.calls[1][1]
    assert "Final answer" not in qwen.calls[1][1]


def test_no_risk_uses_exactly_one_model_call(physical_db):
    qwen = SequenceQwenStub(FIXED)
    result = run_guarded(
        database=physical_db, provider=qwen,
        question="How many customers have orders?", schema_context=SCHEMA,
        adaptive_physical=True,
    )
    assert result.rows == ((2,),)
    assert len(qwen.calls) == 1
    assert "adaptive_schema_probe" not in [s.name for s in result.stages]


def test_where_clause_change_causes_explicit_fallback(physical_db):
    original = BAD + " WHERE orders.amount > 1"
    changed = FIXED + " WHERE orders.amount > 8"
    qwen = SequenceQwenStub(original, changed)
    result = run_guarded(
        database=physical_db, provider=qwen,
        question="How many customers have orders of amount above 1?",
        schema_context=SCHEMA, adaptive_physical=True,
    )
    assert result.status == "ok"
    assert result.candidate.sql == original
    assert result.rows == ((3,),)
    assert "adaptive_fallback" in [s.name for s in result.stages]
    assert len(qwen.calls) == 2


def test_invalid_repair_falls_back_without_execution(physical_db):
    qwen = SequenceQwenStub(BAD, "DELETE FROM orders")
    result = run_guarded(
        database=physical_db, provider=qwen,
        question="How many customers have orders?", schema_context=SCHEMA,
        adaptive_physical=True,
    )
    assert result.status == "ok"
    assert result.candidate.sql == BAD
    assert result.rows == ((3,),)
    with sqlite3.connect(physical_db) as con:
        assert con.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 3


def test_zero_semantic_budget_skips_probes_and_model_repair(physical_db):
    qwen = SequenceQwenStub(BAD)
    result = run_guarded(
        database=physical_db, provider=qwen,
        question="How many customers have orders?", schema_context=SCHEMA,
        adaptive_physical=True, max_repairs=0,
    )
    assert len(qwen.calls) == 1
    assert result.rows == ((3,),)


def test_legacy_default_graph_and_service_inert(physical_db):
    names = set(_build_guarded_graph().get_graph().nodes)
    assert "adaptive_decide" not in names
    qwen = SequenceQwenStub(BAD)
    svc = SemanticSQLService(physical_db, provider=qwen)
    result = svc.ask("How many customers have orders?")
    assert result.rows == ((3,),)
    assert len(qwen.calls) == 1


def test_complex_sql_abstains_and_probes_are_readonly(physical_db):
    assert semantic_risk("How many customers have orders?", "WITH c AS (SELECT 1) SELECT * FROM c") is None
    relation, has_fk = schema_probe(physical_db, BAD)
    sample = data_probe(physical_db, BAD)
    assert has_fk and "customer_id->customers.id" in relation
    assert "one" in sample
    with sqlite3.connect(physical_db) as con:
        assert con.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 3


def test_flags_mutually_exclusive(physical_db):
    qwen = SequenceQwenStub(BAD)
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_guarded(
            database=physical_db, provider=qwen,
            question="How many customers have orders?",
            schema_context=SCHEMA,
            adaptive_physical=True, semantic_revision=True,
        )
    assert qwen.calls == []
