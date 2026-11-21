"""Regression for real Qwen9B SCV12 crash: mixed qualified/unqualified projections."""
from __future__ import annotations

import sqlite3

from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.contract_verifier import verify_semantic_contract
from semantic_sql.semantic_contract import SemanticContract
from semantic_sql.sql_semantics import parse_sql_semantics


def test_real_sql_qualified_and_unqualified_projections_are_sortable(tmp_path):
    db = tmp_path / "real_case.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE people(id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("CREATE TABLE orders(id INTEGER PRIMARY KEY, person_id INTEGER, amount REAL)")
    sql = ("SELECT people.name, amount FROM people "
           "JOIN orders ON people.id=orders.person_id")
    parsed = parse_sql_semantics(sql)
    assert parsed.status == "ok" and parsed.semantics is not None
    assert {x.table for x in parsed.semantics.projections} == {None, "people"}

    contract = SemanticContract(confidence="high", projection=("people.id",))
    check = verify_semantic_contract(contract, parsed.semantics, DatabaseCatalog.from_sqlite(db))
    assert check.status == "violation"
    violation = next(x for x in check.violations if x.code == "PROJECTION_MISMATCH")
    assert violation.actual == "amount, people.name"
    assert violation.expected == "people.id"


def test_missing_qualified_projection_does_not_false_pass(tmp_path):
    db = tmp_path / "minimal.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE people(id INTEGER, name TEXT)")
        conn.execute("CREATE TABLE orders(person_id INTEGER, amount REAL)")
    parsed = parse_sql_semantics(
        "SELECT orders.amount, people.name FROM orders JOIN people "
        "ON orders.person_id=people.id"
    )
    assert parsed.semantics is not None
    verdict = verify_semantic_contract(
        SemanticContract(confidence="high", projection=("orders.person_id",)),
        parsed.semantics,
        DatabaseCatalog.from_sqlite(db),
    )
    assert verdict.status == "violation"
    assert any(item.code == "PROJECTION_MISMATCH" for item in verdict.violations)
