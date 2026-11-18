"""Cheap CPU-only coverage for fail-closed bounded query probe."""
import sqlite3

from evaluation.bounded_shard7_probe import bounded_outcome, score_one


def test_readonly_bounded_fast_and_invalid_query(tmp_path):
    db = tmp_path / "sample.sqlite"
    con = sqlite3.connect(db)
    con.executescript("CREATE TABLE z(v INTEGER);INSERT INTO z VALUES(1),(2);")
    con.close()
    good = bounded_outcome(db, "SELECT v FROM z", 10)
    assert good["valid"] and good["reason"] == "ok"
    bad = bounded_outcome(db, "DROP TABLE z", 10)
    assert not bad["valid"] and bad["reason"] == "SQLExecutionError"


def test_recursive_expensive_query_times_out_without_hanging(tmp_path):
    db = tmp_path / "sample.sqlite"
    sqlite3.connect(db).close()
    slow = ("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL "
            "SELECT x+1 FROM c) SELECT MAX(x) FROM c")
    result = bounded_outcome(db, slow, 0.4)
    assert not result["valid"] and result["reason"] == "query_timeout"


def test_same_k8_goldblind_selected_sql_as_original(tmp_path):
    db = tmp_path / "sample.sqlite"
    con = sqlite3.connect(db)
    con.executescript("CREATE TABLE z(a INT);INSERT INTO z VALUES(1),(1),(2);")
    con.close()
    cache = {"case_id": "only-test", "db_id": "sample", "k": 8,
             "samples": [{"sql": q} for q in (
                 ["SELECT a FROM z"] * 5 + ["SELECT DISTINCT a FROM z"] * 3)]}
    result = score_one(db, cache, "SELECT a FROM z", 10)
    assert result["candidate_timeout_count"] == 0
    assert result["gold_scorable"]
    assert result["first_set_ex"] and result["selected_set_ex"]
    assert result["zero_gold_in_selector"]
