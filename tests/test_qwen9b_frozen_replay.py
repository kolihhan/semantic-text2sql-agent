"""Regressions for exact artifact replay: never silently promote invalid runs."""
import sqlite3
from pathlib import Path

from evaluation.qwen9b_frozen_replay import replay_case, aggregate


def _sqlite(tmp_path: Path) -> Path:
    db=tmp_path/"db.sqlite"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE t (value INTEGER)")
        c.executemany("INSERT INTO t VALUES (?)",[(1,),(2,),(3,)])
    return db


def _case(db_id="trial", *, gold_wrong=False):
    return {
        "case_id":"42", "question":"Find values", "database_id":db_id,
        "initial_sql_identical":True,
        "direct":{"final_sql":"SELECT value FROM t WHERE value <= 2",
                  "official_ex":False,"status":"ok","execution_success":True},
        "guarded":{"final_sql":"SELECT value FROM t",
                   "official_ex":not gold_wrong,"status":"ok","execution_success":True},
    }


def test_replay_identifies_gold_rows_and_true_truncation(tmp_path):
    db=_sqlite(tmp_path)
    good=replay_case(_case(), db, "SELECT value FROM t", max_rows=10)
    assert good["reference_scorable"] is True
    assert good["original_ex_verified"] is True
    assert good["guarded"]["replay_ex"] is True
    assert good["direct"]["replay_ex"] is False
    assert good["guarded"]["rows"]==3
    capped=replay_case(_case(), db, "SELECT value FROM t", max_rows=2)
    assert capped["reference_truncated"] is True
    assert capped["reference_scorable"] is False
    assert capped["guarded"]["replay_ex"] is False


def test_gold_failure_cannot_be_reclassified_as_ex(tmp_path):
    db=_sqlite(tmp_path)
    bad=replay_case(_case(),db,"SELECT missing FROM t",max_rows=10)
    assert bad["reference_scorable"] is False
    assert bad["reference_error"]
    assert bad["guarded"]["replay_ex"] is False


def test_frozen_aggregate_retains_invalidity_even_when_metric_counts_match():
    base={"protocol":"qwen9b-offline-replay-shard-v1",
          "model_digest":"m","source_sha256":"source",
          "cases":[{"case_id":str(i),"db_id":f"db{i}","reference_scorable":True,
                    "original_ex_verified":True,"direct":{"replay_ex":i<1},
                    "guarded":{"replay_ex":i<2},"original":{"direct":i<1,"guarded":i<2}} for i in range(25)]}
    reports=[]
    for i in range(4):
        r={**base,"shard":i,"original_status":"invalid" if i==3 else "complete",
           "cases":[{**c,"case_id":str(i*25+j)} for j,c in enumerate(base["cases"])]}
        reports.append(r)
    summary=aggregate(reports,expected_digest="m",expected_source="source")
    assert summary["raw_guarded_ex"]==8
    assert summary["raw_direct_ex"]==4
    assert summary["clean_replay"] is False
    assert summary["invalid_original_shards"]==[3]
