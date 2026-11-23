from __future__ import annotations

import sqlite3
from pathlib import Path

from experiments.dev100_etsa import (
    ResearchContext, blind_choose, graph, resolves_risk, structural_risk,
)
from semantic_sql.catalog import DatabaseCatalog


class CountingFake:
    def __init__(self, sql: str):
        self.sql = sql
        self.calls = 0

    def complete_text(self, *, system: str, user: str) -> str:
        self.calls += 1
        return self.sql


def make_db(tmp_path: Path) -> tuple[Path, DatabaseCatalog]:
    path = tmp_path / "synthetic.sqlite"
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, amount INTEGER)")
        con.executemany("INSERT INTO orders VALUES (?, ?)", [(1, 10), (2, 20)])
    return path, DatabaseCatalog.from_sqlite(path)


def test_missing_count_has_bounded_source_only_witness(tmp_path):
    db, catalog = make_db(tmp_path)
    w = structural_risk("How many orders are there?", "SELECT id FROM orders", catalog, 2)
    assert w["code"] == "COUNT_MISSING"
    assert resolves_risk("SELECT COUNT(*) FROM orders", w)
    assert not resolves_risk("SELECT id FROM orders LIMIT 1", w)
    assert blind_choose("SELECT id FROM orders", "SELECT COUNT(*) FROM orders", w, db)[1] == "selected_resolved_risk"


def test_no_witness_graph_skips_model_calls(tmp_path):
    db, catalog = make_db(tmp_path)
    g1=CountingFake("SELECT 999")
    g2=CountingFake("SELECT 888")
    result=graph().invoke({
        "question": "List the order IDs",
        "schema_context": catalog.schema_text(),
        "original_sql": "SELECT id FROM orders",
        "original_rows": 2,
    },context=ResearchContext(db,catalog,g1,g2))
    assert result["routed"] is False
    assert g1.calls == g2.calls == 0
    assert result["g1_sql"] == result["g2_sql"] == "SELECT id FROM orders"


def test_routed_graph_same_extra_call_budget_and_fallback(tmp_path):
    db,catalog=make_db(tmp_path)
    g1=CountingFake("SELECT id FROM orders")
    g2=CountingFake("SELECT COUNT(*) FROM orders")
    result=graph().invoke({
        "question":"How many orders?",
        "schema_context":catalog.schema_text(),
        "original_sql":"SELECT id FROM orders",
        "original_rows":2,
    }, context=ResearchContext(db,catalog,g1,g2))
    assert result["routed"] is True
    assert g1.calls==g2.calls==1
    assert result["g1_selected"] is False
    assert result["g2_selected"] is True
    assert result["g2_sql"]=="SELECT COUNT(*) FROM orders"


def test_unverified_alternative_must_not_replace_original(tmp_path):
    db,catalog=make_db(tmp_path)
    witness={"code":"COUNT_MISSING","foreign_keys":[]}
    _,why=blind_choose("SELECT id FROM orders","DROP TABLE orders",witness,db)
    assert why=="risk_not_resolved"
    _,why=blind_choose("SELECT id FROM orders","SELECT COUNT(*) FROM nonexistent",witness,db)
    assert why=="failed_preflight"
