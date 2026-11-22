"""Synthetic safety and negative controls for original-Qwen zero-row research."""
import sqlite3
from scripts.qwen9b_empty_result_targeted_repair import (
    extract_sql, evidence, original75, digest, MODEL_DIGEST, SOURCE,
)

def test_sql_only_no_multistatement_or_commentary():
    assert extract_sql("SELECT * FROM t;") == "SELECT * FROM t;"
    assert extract_sql("Here is SQL: SELECT * FROM t") is None
    assert extract_sql("SELECT * FROM t; DROP TABLE t") is None
    assert extract_sql("SELECT * FROM t -- hi") is None
    assert extract_sql("SELECT * FROM t /* hi */") is None
    fence = chr(96)*3
    assert extract_sql(fence+"sql\nSELECT 1;\n"+fence) == "SELECT 1;"

def test_schema_evidence_reads_db_but_does_not_change_rows(tmp_path):
    db=tmp_path/"demo.sqlite"
    con=sqlite3.connect(db)
    con.execute("CREATE TABLE person (id INTEGER PRIMARY KEY, full_name TEXT)")
    con.execute("INSERT INTO person VALUES (1, 'ALICE')")
    con.commit()
    con.close()
    schema, obs=evidence(db,"SELECT id FROM person WHERE full_name='Al'")
    assert "person(" in schema
    assert obs and obs[0]["column"]=="person.full_name"
    assert "ALICE" in obs[0]["same_prefix_db_values"]
    con=sqlite3.connect(db)
    assert con.execute("SELECT COUNT(*) FROM person").fetchone()[0]==1
    con.close()

def test_gold_label_leak_rejected():
    rows=[{"case_id":str(i), "db_id":"fake", "question":"q", "selected_sql":"SELECT 1", "shard":i//25} for i in range(100)]
    src={"study":SOURCE, "status":"frozen_unlabeled", "model_digest":MODEL_DIGEST, "cases":rows}
    assert len(original75(src))==75
    rows[0]["gold_sql"]="SELECT 1"
    try:
        original75(src)
    except ValueError:
        pass
    else:
        raise AssertionError("gold SQL leaked to inference")
