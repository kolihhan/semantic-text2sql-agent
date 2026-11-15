from pathlib import Path
import sqlite3

import pytest

from evaluation.qwen_value_evidence_spider12 import freeze_spider_cohort, aggregate, PROTOCOL


def test_selection_requires_real_physical_value_and_never_reads_gold(tmp_path):
    dev = []
    for i in range(8):
        dbid = f"db{i}"
        root = tmp_path / dbid
        root.mkdir()
        with sqlite3.connect(root / f"{dbid}.sqlite") as con:
            con.execute("CREATE TABLE t(city TEXT)")
            con.execute("INSERT INTO t VALUES (?)", (f"CasablancaCity{i}",))
        for j in range(2):
            dev.append({"db_id":dbid,"question":f"Find entries in CasablancaCity{i}",
                        "query":"FORBIDDEN_GOLD_SQL"})
    train = [{"db_id":"train_only", "question":"never use dev gold", "query":"SELECT 1"}]
    frozen=freeze_spider_cohort(train,dev,tmp_path)
    assert frozen["count"]==12 and frozen["db_count"]==6
    assert frozen==freeze_spider_cohort(train,dev,tmp_path)
    assert all("query" not in row and "SQL" not in row for row in frozen["cases"])
    assert all(row["value_hits"] for row in frozen["cases"])


def test_no_value_evidence_fails_instead_of_generating():
    with pytest.raises(ValueError, match="physical DBs"):
        freeze_spider_cohort([],[],Path("."))


def test_paired_requires_complete_matched_cohort():
    shards=[]
    for shard in range(3):
        cases=[]
        for j in range(4):
            i=4*shard+j
            old=i<4
            new=(i<4 or i in (4,6))
            cases.append({
                "case_id":f"c{i}","db_id":f"d{i//2}",
                "reference_scorable":True,
                "baseline":{"ex":old,"bag":old,"latency_s":1.0},
                "evidence":{"ex":new,"bag":new,"latency_s":2.0},
            })
        shards.append({"protocol":PROTOCOL,"shard":shard,"cohort_sha256":"same",
                       "model_digest":"test","cases":cases})
    result=aggregate(shards,expected_digest="test")
    assert result["baseline_ex"]==4 and result["evidence_ex"]==6
    assert result["rescues"]==2 and result["regressions"]==0
    assert result["positive_screen_only"] is True
    assert result["promotion_allowed"] is False
    with pytest.raises(ValueError, match="shard"):
        aggregate(shards[:2],expected_digest="test")
