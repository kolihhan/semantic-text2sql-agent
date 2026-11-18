"""Minimal self-contained preflight for the locked real Qwen9B semantic plan."""
import json
import sqlite3
from pathlib import Path

import pytest
from experiments.qwen9b_grounded_plan_train96 import (
    DIGEST,MODEL,N,GATE,checked_rows,validate_plan,sql_result,PLAN_KEYS
)
from semantic_sql.catalog import DatabaseCatalog,TableInfo


@pytest.fixture
def catalog():
    return DatabaseCatalog(
        database=Path("/does/not/exist"),
        tables={
            "orders":TableInfo(name="orders",columns=("Order ID","Customer ID","Total")),
            "customers":TableInfo(name="customers",columns=("Customer ID","Country")),
        },
        foreign_keys=(("orders","Customer ID","customers","Customer ID"),)
    )


def valid_plan():
    return {
        "tables":["orders","customers"],
        "join_keys":[["orders.Customer ID","customers.Customer ID"]],
        "answer_unit":"one distinct order",
        "aggregation":"COUNT DISTINCT orders.Order ID after joining customers",
        "filters":["customers.Country = 'UK'"],
        "ordering":"NONE"
    }


def test_plan_schema_verified_without_gold(catalog):
    p=validate_plan(json.dumps(valid_plan()),catalog)
    assert p is not None
    assert p["aggregation"].startswith("COUNT DISTINCT")
    assert set(p)==PLAN_KEYS


def test_plan_invented_join_column_rejected(catalog):
    p=valid_plan()
    p["join_keys"]=[["orders.customer_email","customers.Country"]]
    assert validate_plan(json.dumps(p),catalog) is None


def test_plan_unknown_table_rejected(catalog):
    p=valid_plan()
    p["tables"]+=["invented_table"]
    assert validate_plan(json.dumps(p),catalog) is None


def test_malformed_and_unstructured_plans_rejected(catalog):
    assert validate_plan("SELECT * FROM orders",catalog) is None
    assert validate_plan(json.dumps({"tables":["orders"]}),catalog) is None
    p=valid_plan()
    p["filters"]=["x"*200]
    assert validate_plan(json.dumps(p),catalog) is None


def test_schema_ambiguous_or_missing_table_join_rejected(catalog):
    p=valid_plan()
    p["tables"]=["orders"]
    assert validate_plan(json.dumps(p),catalog) is None


def test_actual_readonly_sql_set_bag_and_reject_mutation(tmp_path):
    db=tmp_path/"test.sqlite"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE orders(id INTEGER)")
        con.executemany("INSERT INTO orders VALUES (?)",[(1,),(1,),(2,)])
    a=sql_result(db,"SELECT id FROM orders")
    b=sql_result(db,"SELECT DISTINCT id FROM orders")
    assert a["valid"] and b["valid"]
    assert a["set"]==b["set"] and a["bag"]!=b["bag"]
    x=sql_result(db,"DROP TABLE orders")
    assert not x["valid"]
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT COUNT(*) FROM orders").fetchone()[0]==3


def test_unchecked_source_gates_stop_before_inference(tmp_path):
    a=tmp_path/"questions.jsonl";m=tmp_path/"freeze.json"
    a.write_text('{"source_index":1,"db_id":"book","question":"How many books?"}\n')
    m.write_text('{"status":"FROZEN_SOURCE_NO_GOLD_BEFORE_REAL_QWEN_GENERATION"}')
    with pytest.raises(ValueError,match="drift"):
        checked_rows(a,m)


def test_fixed_original_qwen_and_large_enough_source():
    assert MODEL=="qwen3.5:9b-q4_K_M"
    assert DIGEST=="56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
    assert N==96 and GATE["min_net_correct"]>=8
