"""Freeze the K8 deduplicated-program selector before any publisher EX labels."""
import json
import sqlite3
from pathlib import Path

import pytest

from evaluation.k8_program_diversity_vote import (
    FIXED_MODEL_DIGEST,PROTOCOL,conservative_program,choose_index,
    _bounded_fingerprint,score_original_candidates,bootstrap,validate_cache
)


def test_preserve_quoted_literal_case_and_whitespace():
    s="SELECT  item FROM books WHERE name = 'First  Class' ;"
    t="select item\nfrom books where name = 'First  Class';"
    assert conservative_program(s)==conservative_program(t)
    assert conservative_program(s)!=conservative_program(
        "SELECT item FROM books WHERE name='first  class';")


def test_identical_sql_program_mode_gets_one_vote():
    sqls=["SELECT 0"]*5+["SELECT 1","select 1","SELECT 2"]
    outcomes=[
        {"valid":True,"signature":"A"} for _ in range(5)
    ] + [{"valid":True,"signature":"B"} for _ in range(3)]
    old=choose_index(sqls,outcomes,dedup_program=False)
    new=choose_index(sqls,outcomes,dedup_program=True)
    assert old["index"]==0 and new["index"]==5
    assert old["votes"]==5 and new["votes"]==2


def test_distinct_equivalent_programs_do_not_get_collapsed():
    sqls=[
        "SELECT 1 AS x","select 1 as x","SELECT CAST(1 AS INTEGER) AS x",
        "SELECT 1+0 AS x","SELECT 1*1 AS x","SELECT 2","SELECT 2","SELECT 2"
    ]
    outcomes=[
        {"valid":True,"signature":"A"} for _ in range(5)
    ] + [{"valid":True,"signature":"B"} for _ in range(3)]
    one=choose_index(sqls,outcomes,dedup_program=True)
    assert one["index"]==0 and one["votes"]==4


def test_no_valid_result_fails_closed_to_original_first():
    sql=["SELECT 1"]*8
    bad=[{"valid":False,"signature":None} for _ in sql]
    assert choose_index(sql,bad,dedup_program=False)["index"]==0
    assert choose_index(sql,bad,dedup_program=True)["index"]==0


def test_cannot_use_candidate_reference_or_outcome_labels_to_select():
    sql=["SELECT 1","SELECT 2"]*4
    result=[{"valid":True,"signature":x,"gold_is_this":False,
             "official_EX":False} for x in "ABABABAB"]
    original=choose_index(sql,result,dedup_program=True)["index"]
    for row in result:
        row["gold_is_this"]=True
        row["official_EX"]=True
    assert choose_index(sql,result,dedup_program=True)["index"]==original


def test_bounded_sqlite_original_candidate_execution(tmp_path:Path):
    p=tmp_path/"x.sqlite"
    with sqlite3.connect(p) as con:
        con.execute("CREATE TABLE t(id INTEGER, val TEXT)")
        con.executemany("INSERT INTO t VALUES (?,?)",[(1,"hello"),(1,"hello"),(2,"world")])
    a=_bounded_fingerprint(p,"SELECT id FROM t")
    b=_bounded_fingerprint(p,"SELECT DISTINCT id FROM t")
    assert a["valid"] and b["valid"]
    assert a["set_hash"]==b["set_hash"]
    assert a["bag_hash"]!=b["bag_hash"]
    arr=score_original_candidates(p,["select id from t","SELECT id FROM t"])
    assert arr[0]["signature"]==arr[1]["signature"]


def test_reject_non_readonly_sql_without_mutation(tmp_path:Path):
    p=tmp_path/"w.sqlite"
    with sqlite3.connect(p) as con:
        con.execute("CREATE TABLE t(id INTEGER)")
        con.execute("INSERT INTO t VALUES (1)")
    x=_bounded_fingerprint(p,"DELETE FROM t")
    assert not x["valid"]
    with sqlite3.connect(p) as con:
        assert con.execute("SELECT COUNT(*) FROM t").fetchone()[0]==1


def test_bootstrap_db_cluster_diff_from_question():
    rows=[
        {"db_id":"A","base":False,"new":True} for _ in range(8)
    ] + [{"db_id":"B","base":True,"new":False}]
    a=bootstrap(rows,"base","new",cluster=True,rounds=400)
    b=bootstrap(rows,"base","new",cluster=False,rounds=400)
    assert a["n"]==b["n"]==9
    assert a["unit"]=="database" and b["unit"]=="question"
    assert a["database_clusters"]==2


def test_digest_and_cache_gate_rejects_fabricated_source():
    with pytest.raises(ValueError,match="protocol drift"):
        validate_cache({
            "status":"CANDIDATE_CACHE_COMPLETE",
            "model_digest":"changed_base",
            "k":8
        },0)


def test_program_lowercases_only_outside_literals():
    assert conservative_program("SELECT 'A'")!=conservative_program("SELECT 'a'")
    assert conservative_program('SELECT "Case Name"')!=conservative_program('SELECT "case name"')


def test_protocol_only_same_original_model():
    assert FIXED_MODEL_DIGEST=="56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
    assert "ORIGINAL-K8" in PROTOCOL
