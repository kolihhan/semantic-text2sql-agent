"""Synthetic SQLite anti-false-positive tests for frozen Spider posthoc scoring."""
import json
import sqlite3

import pytest

from experiments.audit_spider_exec_strict import score_readonly_pair,audit


@pytest.fixture
def db(tmp_path):
    path=tmp_path/"simple.sqlite"
    with sqlite3.connect(path) as conn:
        conn.executescript("CREATE TABLE vals(x INTEGER); INSERT INTO vals VALUES (1),(1),(2);")
    return path


def test_set_equality_can_hide_duplicate_row_mismatch(db):
    p=score_readonly_pair(db,"SELECT DISTINCT x FROM vals","SELECT x FROM vals")
    assert p["execution_success"]
    assert p["set_equal"] is True
    assert p["bag_equal"] is False
    assert p["exact_ast_equal"] is False
    assert p["pred_rows"]==2 and p["gold_rows"]==3


def test_order_sensitivity_is_reported_separately_not_overclaimed(db):
    p=score_readonly_pair(db,
        "SELECT x FROM vals ORDER BY x ASC",
        "SELECT x FROM vals ORDER BY x DESC")
    assert p["set_equal"] and p["bag_equal"]
    assert p["ordered_equal"] is False
    assert p["gold_has_order"] is True


def test_empty_result_match_marked_as_uninformative(db):
    p=score_readonly_pair(db,"SELECT x FROM vals WHERE x = 99",
                         "SELECT x FROM vals WHERE x = 100")
    assert p["set_equal"] and p["bag_equal"] and p["both_empty"]
    assert not p["exact_ast_equal"]


def test_failed_or_write_sql_is_not_scored_as_correct(db):
    p=score_readonly_pair(db,"DELETE FROM vals","SELECT x FROM vals")
    assert not p["execution_success"] and not p["bag_equal"]
    p=score_readonly_pair(db,"SELECT missing FROM vals","SELECT x FROM vals")
    assert not p["execution_success"]


def _frozen():
    source=[]
    cases=[]
    for i in range(100):
        source.append({"db_id":"synthetic","question":f"What are the values {i}?",
                       "query":"SELECT x FROM vals"})
        base="SELECT DISTINCT x FROM vals" if i==0 else "SELECT x FROM vals"
        cases.append({
            "case_id":f"synthetic:{i}","row_index":i,"database_id":"synthetic",
            "question":source[-1]["question"],
            "sql":{"a0":base,"a4":"SELECT x FROM vals"},
            "scores":{
                "a0":{"frozen_execution_match":True},
                "a4":{"frozen_execution_match":True},
            },
        })
    frozen={"schema":"spider-source-shift-a0-a4-aggregate-v1",
            "cohort_sha256":"sha256", "cases":cases,
            "a0_correct":100,"a4_correct":100}
    return frozen,source


def test_posthoc_detects_hidden_bag_win_even_when_set_scores_same(tmp_path):
    root=tmp_path/"root"
    folder=root/"synthetic"
    folder.mkdir(parents=True)
    with sqlite3.connect(folder/"synthetic.sqlite") as conn:
        conn.executescript("CREATE TABLE vals(x INTEGER); INSERT INTO vals VALUES (1),(1),(2);")
    frozen,source=_frozen()
    report=audit(frozen,source,root)
    assert report["sample_size"]==100
    assert report["metrics"]["a0"]["set_correct"]==100
    assert report["metrics"]["a0"]["bag_correct"]==99
    assert report["metrics"]["a4"]["bag_correct"]==100
    assert report["metrics"]["a0"]["set_false_positive_vs_bag"]==1
    assert report["paired_transitions"]["bag_equal"]["net"]==1
    assert report["paired_transitions"]["set_equal"]["net"]==0
    assert report["changed_sql_count"]==1
    source[0]["question"]="Changed question"
    with pytest.raises(ValueError,match="identity"):
        audit(frozen,source,root)


def test_immutable_score_drift_rejected(tmp_path):
    root=tmp_path/"root"
    folder=root/"synthetic"
    folder.mkdir(parents=True)
    with sqlite3.connect(folder/"synthetic.sqlite") as conn:
        conn.executescript("CREATE TABLE vals(x INTEGER); INSERT INTO vals VALUES (1),(1),(2);")
    frozen,source=_frozen()
    frozen["cases"][0]["scores"]["a0"]["frozen_execution_match"]=False
    with pytest.raises(ValueError,match="scoring drift"):
        audit(frozen,source,root)
