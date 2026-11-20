"""CPU-only preregistered source and historical SQL probe regression tests."""
from __future__ import annotations
import sqlite3
from pathlib import Path

import pytest

from scripts.typed_erp_source_first import (
    SEALED, category, digest, match_tables, schema_for, type_source, norm_question,
)
from scripts.typed_erp_historical_score import decide, _parse_count


def fixture_schema():
    return {
        "tables":{
            "customers":{"pk":["id"],"columns":["id","name"]},
            "orders":{"pk":["id"],"columns":["id","customer_id"]},
            "ambiguous":{"pk":[],"columns":["value"]},
        },
        "foreign_keys":[{
            "child":"orders","column":"customer_id",
            "parent":"customers","parent_column":"id","fk_id":0,"seq":0,
        }],
    }


@pytest.mark.parametrize(("q","expected"),[
    ("How many unique customers placed orders?","ENTITY_COUNT"),
    ("What percentage of customers placed orders?","RATIO_PERCENTAGE"),
    ("Which customers have both orders and payments?","RELATION_SET"),
    ("List customers by name.","OTHER"),
    ("?","UNKNOWN"),
])
def test_question_only_classes(q,expected):
    assert category(q)==expected


def test_source_grounding_has_explicit_abstentions():
    s=fixture_schema()
    r=type_source({"question":"How many distinct customers placed orders?"},s)
    assert r["candidate_label"]=="ENTITY_COUNT"
    assert r["target_entity"]=="customers"
    assert r["counting_unit"]=={
        "table":"customers","pk":"id","explicit_unique_requested":True,
    }
    assert r["candidate_only_not_ground_truth"]
    assert r["status"]=="GROUNDED_CANDIDATE"
    ratio=type_source({"question":"What percentage of customers made orders?"},s)
    assert ratio["status"]=="ABSTAIN"
    assert ratio["numerator"] is None and ratio["denominator"] is None


def test_relation_requires_real_schema_fk_not_lexical_guess():
    schema=fixture_schema()
    row={"question":"Which customers have both orders and invoices?"}
    a=type_source(row,schema)
    assert a["candidate_label"]=="RELATION_SET"
    assert a["status"]=="GROUNDED_CANDIDATE"
    assert a["relation"]["child"]=="orders"
    assert a["relation"]["parent"]=="customers"
    schema["foreign_keys"]=[]
    b=type_source(row,schema)
    assert b["status"]=="ABSTAIN"
    assert b["relation"] is None


def test_source_function_never_serializes_gold_or_predictions():
    row={"question":"How many distinct customers placed orders?",
         "SQL":"SELECT secret_gold","original_prediction":"private"}
    decision=type_source(row,fixture_schema())
    assert "SQL" not in decision and "original_prediction" not in decision
    assert "secret_gold" not in str(decision)
    assert SEALED=={"simpson_episodes","works_cycles"}


def test_sql_probe_only_conflicts_on_explicit_unit_pk_and_fk():
    r=type_source({"question":"How many unique customers placed orders?"},fixture_schema())
    wrong="SELECT COUNT(DISTINCT o.id) FROM customers c JOIN orders o ON o.customer_id=c.id"
    right="SELECT COUNT(DISTINCT c.id) FROM customers c JOIN orders o ON o.customer_id=c.id"
    a=decide(r,wrong)
    assert a["decision"]=="CONFLICT_CANDIDATE"
    assert a["semantic_conflict_verified"] is False
    assert decide(r,right)["decision"]=="CONSISTENT_UNIT"
    assert decide(r,"SELECT COUNT(*) FROM customers c JOIN orders o ON o.customer_id=c.id")["decision"]=="ABSTAIN"


def test_abstains_for_unknown_relation_or_ratio():
    s=fixture_schema()
    ratio=type_source({"question":"What percentage of customers have orders?"},s)
    assert decide(ratio,"SELECT COUNT(DISTINCT orders.id) FROM orders")["decision"]=="ABSTAIN"
    raw=type_source({"question":"Which customers have both orders and payments?"},s)
    assert decide(raw,"SELECT * FROM orders")["decision"]=="ABSTAIN"


@pytest.mark.parametrize("sql",[
    "WITH x AS (SELECT 1) SELECT COUNT(*) FROM x",
    "SELECT COUNT(*) FROM customers GROUP BY id",
    "SELECT COUNT(*) FROM customers; DELETE FROM customers",
    "SELECT COUNT(*) FROM customers UNION SELECT COUNT(*) FROM orders",
])
def test_probe_rejects_complex_unsafe_or_multi_query(sql):
    assert _parse_count(sql) is None


def test_read_only_physical_schema_uses_declared_keys(tmp_path:Path):
    db=tmp_path/"t.sqlite"
    with sqlite3.connect(db) as con:
        con.executescript("""
            CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT);
            CREATE TABLE orders(id INTEGER PRIMARY KEY, customer_id INTEGER
                                REFERENCES customers(id));
        """)
    s=schema_for(db)
    assert s["tables"]["customers"]["pk"]==["id"]
    assert s["tables"]["orders"]["pk"]==["id"]
    assert len(s["foreign_keys"])==1
    assert s["foreign_keys"][0]["parent"]=="customers"


def test_question_id_independent_and_no_output_label_dependency():
    s=fixture_schema()
    a=type_source({"question":"How many unique customers placed orders?","source_index":1},s)
    b=type_source({"question":"How many unique customers placed orders?","source_index":90001},s)
    assert a==b
    assert norm_question(" A   B ")=="a b"
    assert len(digest(b"frozen"))==64
