"""CPU-only preregistration: no fresh generation and no gold-informed selector."""
import sqlite3
from pathlib import Path
import pytest
from evaluation import offline_bag_selector as s
from evaluation.kcache_protocol import candidate_outcome, majority_selector


def test_bag_consensus_uses_row_multiplicity_not_gold(tmp_path):
    db=tmp_path/"sample.sqlite"
    conn=sqlite3.connect(db)
    conn.executescript("CREATE TABLE z(a INT);INSERT INTO z VALUES(1),(1),(2);")
    conn.commit();conn.close()
    cache={"case_id":"synthetic","db_id":"z","k":8,
        "samples":[{"sql":q} for q in (
            ["SELECT DISTINCT a FROM z"]+
            ["SELECT a FROM z"]*6+
            ["SELECT DISTINCT a FROM z"])]}
    outcomes=[candidate_outcome(db,row["sql"]) for row in cache["samples"]]
    assert majority_selector(outcomes)["index"]==0
    assert s.bag_majority_selector(outcomes)["index"]==1
    duplicate=s.evaluate_one(cache,db,"SELECT a FROM z")
    dedup=s.evaluate_one(cache,db,"SELECT DISTINCT a FROM z")
    assert not duplicate["same_index"] and not dedup["same_index"]
    assert duplicate["original_set_vote_ex"] and duplicate["bag_vote_set_ex"]
    assert not duplicate["original_set_vote_bag_ex"]
    assert duplicate["bag_vote_bag_ex"]
    # Changing the private publisher reference does NOT change the selector.
    assert duplicate["same_index"]==dedup["same_index"]
    assert duplicate["selector_no_gold"] and dedup["selector_no_gold"]


def test_selector_handles_nonexecuting_and_ties():
    assert s.bag_majority_selector([
        {"valid":False,"bag_hash":None}]*8)["index"]==0
    result=s.bag_majority_selector([
        {"valid":True,"bag_hash":"a"},
        {"valid":True,"bag_hash":"b"},
        {"valid":True,"bag_hash":"a"}])
    assert result["index"]==0 and result["agreement"]==2/3


def test_complete_500_fail_closed_not_77_case_pilot():
    with pytest.raises(ValueError):
        s.aggregate([],{"dev_case_ids":[]})
    assert s.MIN_EFFECT==.05
    assert s.PROTOCOL=="BIRD-DEV500-K8-OFFLINE-BAG-SET-V1"
