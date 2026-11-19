from __future__ import annotations

import sqlite3
from pathlib import Path

from semantic_sql.contracts import SQLCandidate
from semantic_sql.grain_guard import correct_entity_grain
from semantic_sql.inference import run_guarded


class NoModelCalls:
    def complete_text(self, *, system: str, user: str):
        raise AssertionError("A and B must reuse identical original SQL, no model call")


def demo_db(tmp_path: Path) -> Path:
    db = tmp_path / "records.sqlite"
    with sqlite3.connect(db) as conn:
        conn.executescript("""
        CREATE TABLE Patient(ID INTEGER PRIMARY KEY, Description TEXT);
        CREATE TABLE Laboratory(ID INTEGER, Date TEXT, CRP TEXT,
          PRIMARY KEY (ID, Date),
          FOREIGN KEY(ID) REFERENCES Patient(ID));
        INSERT INTO Patient VALUES (1,NULL),(2,NULL),(3,'recorded');
        INSERT INTO Laboratory VALUES (1,'2020-01-01','+'),
          (1,'2021-01-01','+'),(2,'2021-01-01','+'),(3,'2021-01-01','+');

        CREATE TABLE customers(CustomerID INTEGER PRIMARY KEY, Currency TEXT);
        CREATE TABLE transactions_1k(TransactionID INTEGER PRIMARY KEY,
          CustomerID INTEGER, Date TEXT,
          FOREIGN KEY(CustomerID) REFERENCES customers(CustomerID));
        INSERT INTO customers VALUES(1,'EUR'),(2,'CZK');
        INSERT INTO transactions_1k VALUES(1,1,'2012-08-25'),
          (2,1,'2012-08-25'),(3,1,'2012-08-25'),(4,2,'2012-08-25');

        CREATE TABLE employee(EmployeeID INTEGER PRIMARY KEY, DeptID INTEGER);
        CREATE TABLE dept(DeptID INTEGER PRIMARY KEY);
        INSERT INTO dept VALUES(1);
        INSERT INTO employee VALUES(1,1),(2,1);
        """)
    return db


def value(db: Path, sql: str):
    with sqlite3.connect(db) as conn:
        return conn.execute(sql).fetchone()[0]


def test_entity_count_rewrites_only_if_real_join_row_fanout(tmp_path):
    db = demo_db(tmp_path)
    sql = """SELECT COUNT(*) FROM Laboratory L JOIN Patient P ON L.ID=P.ID
    WHERE L.CRP = '+' AND P.Description IS NULL"""
    updated, reason = correct_entity_grain(
        db, "How many patients with abnormal C-reactive protein have no data recorded?", sql)
    assert reason == "target_entity_pk_grain_with_physical_fanout"
    assert 'COUNT(DISTINCT "P"."ID")' in updated
    assert value(db, sql) == 3 and value(db, updated) == 2


def test_entity_percentage_rewrites_both_numerator_and_denominator(tmp_path):
    db = demo_db(tmp_path)
    sql = """SELECT CAST(SUM(CASE WHEN c.Currency = 'EUR' THEN 1 ELSE 0 END) AS FLOAT)
    * 100.0 / COUNT(*) FROM customers c JOIN transactions_1k t
    ON c.CustomerID = t.CustomerID WHERE t.Date = '2012-08-25'"""
    updated, reason = correct_entity_grain(
        db, "What is the percentage of the customers who used EUR in 2012/8/25?", sql)
    assert reason == "target_entity_pk_grain_with_physical_fanout"
    assert 'COUNT(DISTINCT CASE WHEN c.Currency' in updated
    assert value(db, sql) == 75.0 and value(db, updated) == 50.0


def test_one_to_one_join_and_incompatible_count_entity_do_not_change(tmp_path):
    db = demo_db(tmp_path)
    sql = "SELECT COUNT(*) FROM employee e JOIN dept d ON e.DeptID=d.DeptID"
    unchanged, why = correct_entity_grain(db, "How many employees are in departments?", sql)
    assert unchanged == sql and why == "no_observed_join_fanout"
    ignored, why = correct_entity_grain(db, "How many departments have employees?", sql)
    assert ignored == sql and why == "no_observed_join_fanout"


def test_extra_joins_subqueries_and_unsafe_grain_are_not_modified(tmp_path):
    db = demo_db(tmp_path)
    examples = [
        "SELECT COUNT(*) FROM Patient P",
        "SELECT COUNT(*) FROM Laboratory L JOIN Patient P ON L.ID=P.ID GROUP BY P.ID",
        "SELECT COUNT(*) FROM Laboratory L JOIN Patient P ON L.ID=P.ID WHERE L.ID IN (SELECT ID FROM Patient)",
    ]
    for sql in examples:
        updated, why = correct_entity_grain(db, "How many patients?", sql)
        assert updated == sql


def test_real_original_guarded_graph_optin_vs_baseline_no_qwen(tmp_path):
    db = demo_db(tmp_path)
    sql = """SELECT COUNT(*) FROM Laboratory L JOIN Patient P ON L.ID = P.ID
    WHERE L.CRP = '+' AND P.Description IS NULL"""
    base = run_guarded(
        database=db, provider=NoModelCalls(), question="How many patients with abnormal CRP?",
        schema_context="", initial_candidate=SQLCandidate(sql=sql), max_repairs=2)
    treated = run_guarded(
        database=db, provider=NoModelCalls(), question="How many patients with abnormal CRP?",
        schema_context="", initial_candidate=SQLCandidate(sql=sql), max_repairs=2,
        grain_guard=True)
    assert base.status == "ok" and treated.status == "ok"
    assert base.rows == [(3,)] and treated.rows == [(2,)]
    assert base.candidate.attempt == treated.candidate.attempt == 0
    assert not any(stage.name=="grain_guard" for stage in base.stages)
    assert any(stage.name=="grain_guard" for stage in treated.stages)
