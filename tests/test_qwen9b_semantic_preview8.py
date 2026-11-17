"""Gold-blind SQL reviewer preview contracts. CPU tests are not new SQL EX."""
from collections import Counter
from pathlib import Path
import json
import sqlite3

import pytest
from experiments import qwen9b_semantic_preview8 as m


def test_review_prompt_has_no_oracle_reference():
    x=m._prompt("How many orders?","Question: How many orders?\nSchema context:\norders(id)","SELECT COUNT(*) FROM orders",{
        "column_names":["COUNT(*)"],"rows":[[5]],"full_row_count":1})
    d=json.loads(x)
    assert set(d)=={"question","original_question_and_schema","baseline_sql","physical_sqlite_preview"}
    assert "gold" not in x.lower() and "reference_sql" not in x


def test_preview_reads_exact_real_sqlite_columns_only(tmp_path):
    db=tmp_path/"x.sqlite"
    conn=sqlite3.connect(db)
    conn.execute("create table orders(id integer, amount integer)")
    conn.executemany("insert into orders values (?,?)",[(1,20),(2,30),(3,40),(4,50)])
    conn.commit()
    conn.close()
    p=m._preview(db,"SELECT amount FROM orders ORDER BY id")
    assert p["column_names"]==["amount"]
    assert p["full_row_count"]==4 and p["rows"]==[[20],[30],[40]]
    assert p["not_reference_or_gold"] is True


def test_sql_semantic_correctness_set_and_bag_not_conflated(tmp_path):
    db=tmp_path/"x.sqlite"
    conn=sqlite3.connect(db)
    conn.execute("create table x(v integer)")
    conn.executemany("insert into x values (?)",[(1,),(1,),(2,)])
    conn.commit();conn.close()
    assert m.score(db,"SELECT DISTINCT v FROM x",{(1,),(2,)},Counter({(1,):2,(2,):1}))=={
        "ex":True,"bag":False,"exec":True}
    assert not m.score(db,"DROP TABLE x",{(1,)},Counter())["exec"]


def test_missing_frozen_source_not_pretended_physical(tmp_path):
    cohort=tmp_path/"bad.jsonl"
    cohort.write_text('{"source_index":1}\n')
    meta=tmp_path/"meta.json"
    meta.write_text(json.dumps({"status":"FROZEN_BEFORE_ANY_NEW_MODEL_INFERENCE",
                                 "protocol":m.PROTOCOL,"source_sha256":"0"*64}))
    with pytest.raises(RuntimeError):
        m.load_frozen(cohort,meta)


def test_fail_closed_on_partial_aggregate(tmp_path):
    with pytest.raises(ValueError):
        m.aggregate([])
    with pytest.raises(FileNotFoundError):
        m.aggregate([tmp_path/"missing.json"]*4)


def test_review_rule_and_predeclared_gate_stable():
    assert len(m.RULE_SHA)==64
    assert m.GATE["min_net_set_ex"]==2
    assert m.GATE["max_set_regressions"]==0
    assert m.N==8 and m.SHARDS==4
    assert "no gold" in m.RULE.lower()
