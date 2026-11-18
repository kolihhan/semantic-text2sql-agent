"""Minimal physical safety + fixed-model paired-budget checks."""
import json
import sqlite3
from pathlib import Path
import pytest
from experiments.qwen9b_relation_probe96 import (
    DIGEST,GATE,MODEL,N,checked_source,execute,FINAL_EVIDENCE
)


def test_query_only_physical_preview_and_strict_set_bag(tmp_path:Path):
    db=tmp_path/"demo.sqlite"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE orders(id INTEGER, customer TEXT)")
        c.executemany("INSERT INTO orders VALUES(?,?)",
                      [(1,"a"),(1,"a"),(2,"b"),(3,"b"),(4,"c"),(5,"d")])
    # A probe with >5 rows is still a valid sample, never an authoritative
    # 5-row complete answer; strict EX refuses the same truncation.
    sample=execute(db,"SELECT * FROM orders",max_rows=5,preview=True)
    assert sample["ok"] and len(sample["rows"])==5
    assert not execute(db,"SELECT * FROM orders",max_rows=5)["ok"]
    a=execute(db,"SELECT id FROM orders")
    b=execute(db,"SELECT DISTINCT id FROM orders")
    assert a["ok"] and b["ok"]
    assert a["set_sha256"]==b["set_sha256"]
    assert a["bag_sha256"]!=b["bag_sha256"]


def test_query_only_denies_writes_and_multi_statement(tmp_path:Path):
    db=tmp_path/"write.sqlite"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE x(a INT)")
        c.execute("INSERT INTO x VALUES(1)")
    for sql in ("DROP TABLE x","SELECT a FROM x; DELETE FROM x;",
                "WITH x AS (DELETE FROM x RETURNING a) SELECT a FROM x"):
        assert execute(db,sql)["ok"] is False
    assert execute(db,"SELECT COUNT(*) FROM x")["ok"]


def test_source_refuses_forged_meta_and_gold_fields(tmp_path:Path):
    source=tmp_path/"source96.jsonl"
    meta=tmp_path/"freeze.json"
    source.write_text(json.dumps({"source_index":0,"db_id":"a",
                                  "question":"Which orders?", "SQL":"gold"})+"\n")
    meta.write_text(json.dumps({"status":"FULL_HISTORY_EXCLUDED_QUESTION_ONLY_SOURCE_FROZEN_PRE_MODEL"}))
    with pytest.raises(ValueError,match="drift"):
        checked_source(source,meta)


def test_original_model_equal_budget_and_durable_gate():
    assert MODEL=="qwen3.5:9b-q4_K_M"
    assert DIGEST=="56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
    assert N==96 and GATE["min_net_correct"]==8
    assert "real, read-only SQLite probe" in FINAL_EVIDENCE
    assert "gold" not in FINAL_EVIDENCE.lower()
