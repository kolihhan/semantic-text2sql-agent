"""Synthetic physical SQLite controls, no real model EX or source labels used."""
from pathlib import Path
import sqlite3
import pytest

from evaluation.entity_count_relational_witness import propose_entity_count_witness
from scripts.audit_birdtrain64_entity_count_witness import (
    read_source,original_predictions,aggregate,MODEL_DIGEST,PROTOCOL
)


@pytest.fixture
def db(tmp_path):
    path=tmp_path/"orders.sqlite"
    with sqlite3.connect(path) as con:
        con.executescript("""
        CREATE TABLE orders ("Order ID" TEXT, "Product ID" TEXT, "Customer ID" TEXT);
        CREATE TABLE people ("Customer ID" TEXT, state TEXT);
        INSERT INTO orders VALUES
        ('o1','p1','c1'),('o1','p2','c1'),('o2','p3','c2'),('o3','p4','c1');
        INSERT INTO people VALUES ('c1','Texas'),('c2','Oregon');
        """)
    return path


def prompt(q):
    return "Question: "+q+"\nSchema context:\norders(Order ID, Product ID, Customer ID)"


def test_true_entity_rows_are_not_order_entities(db):
    sql="SELECT COUNT(*) FROM orders o WHERE o.\"Customer ID\" = 'c1'"
    w=propose_entity_count_witness(db,prompt("How many orders were placed by c1?"),sql)
    assert w.changed and w.original_row_count==3 and w.distinct_entity_count==2
    assert w.entity_table=="orders" and w.entity_column=="Order ID"
    assert 'COUNT(DISTINCT "o"."Order ID")' in w.sql


def test_sql_join_alias_same_filters_not_rewritten(db):
    sql=("SELECT COUNT(*) AS count FROM orders AS o JOIN people AS p "
         "ON o.\"Customer ID\"=p.\"Customer ID\" WHERE p.state='Texas'")
    w=propose_entity_count_witness(db,prompt("How many orders by people in Texas?"),sql)
    assert w.changed and (w.original_row_count,w.distinct_entity_count)==(3,2)
    assert "WHERE p.state='Texas'" in w.sql


def test_no_duplicate_witness_abstains(db):
    sql='SELECT COUNT(*) FROM orders o WHERE o."Order ID" = \'o2\''
    w=propose_entity_count_witness(db,prompt("How many orders?"),sql)
    assert not w.changed and w.reason=="no_physical_duplicate_entity_witness"


def test_compound_question_abstains(db):
    w=propose_entity_count_witness(
        db,prompt("How many orders were there? List those names."),
        "SELECT COUNT(*) FROM orders")
    assert not w.changed and w.reason=="compound_answer_request"


def test_self_join_key_ambiguous(db):
    w=propose_entity_count_witness(
        db,prompt("How many orders?"),
        'SELECT COUNT(*) FROM orders o JOIN orders x ON o."Order ID"=x."Order ID"')
    assert not w.changed and w.reason=="no_unique_entity_key_mapping"


def test_no_valid_question_entity_identifier(db):
    w=propose_entity_count_witness(
        db,prompt("How many patients?"),"SELECT COUNT(*) FROM orders")
    assert not w.changed and w.reason=="no_unique_entity_key_mapping"


@pytest.mark.parametrize("bad_sql",[
    "SELECT COUNT(*) FROM orders; DELETE FROM orders;",
    'SELECT COUNT(*) FROM orders GROUP BY "Customer ID"',
    "SELECT COUNT(*) * 100.0 / 5 FROM orders",
    'SELECT COUNT(DISTINCT "Order ID") FROM orders',
    "WITH prep AS (SELECT * FROM orders) SELECT COUNT(*) FROM prep",
])
def test_invalid_structure_fail_closed(db,bad_sql):
    result=propose_entity_count_witness(db,prompt("How many orders?"),bad_sql)
    assert not result.changed
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT COUNT(*) FROM orders").fetchone()[0]==4


def test_times_not_entity(db):
    result=propose_entity_count_witness(
        db,prompt("How many times have we shipped?"),
        "SELECT COUNT(*) FROM orders")
    assert not result.changed and result.reason=="not_entity_count_intent"


def test_original_source_hash_and_model_sha_fail_closed(tmp_path):
    source=tmp_path/"source.jsonl"
    source.write_text('{"source_index":1}\n')
    with pytest.raises(ValueError,match="source hash"):
        read_source(source)
    model=tmp_path/"model.json"
    model.write_text('{"status":"complete","model_digest":"switched model","n":64}')
    with pytest.raises(ValueError,match="identity"):
        original_predictions(model)


def test_aggregate_refuses_partial_source(tmp_path):
    with pytest.raises((ValueError,FileNotFoundError)):
        aggregate([],tmp_path/"missing-original-source.jsonl")


def test_original_fixed_model_and_prespecified_gate():
    assert MODEL_DIGEST=="56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
    assert "NOGOLD" in PROTOCOL
