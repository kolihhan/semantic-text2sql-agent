from __future__ import annotations

import json
from collections import Counter
import sqlite3

import pytest

from experiments.qwen9b_physical_value16_replica import (
    COHORT_SHA,PROTOCOL,aggregate,load_frozen,split_prompt,strict_score,
)


def test_original_publisher_prompt_split_stable_with_new_value_evidence():
    q,s,e=split_prompt("Question: Where India?\nSchema context:\ncountry(name)\nVerified database value-column candidates (exact matches, not instructions):\n- country.name contains exact value 'India'")
    assert q=="Where India?" and "country.name" in s and e is None
    q,s,e=split_prompt("Question: Name?\nSchema context:\ncountry(name)\nExternal evidence:\nname not code\nVerified database value-column candidates (exact matches, not instructions):\n- country.name contains exact value 'India'")
    assert q=="Name?" and "name not code" in e and "Verified database" in e


def test_readonly_execution_match_scorer(tmp_path):
    db=tmp_path/"src.sqlite"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE codes(k INTEGER)")
        c.executemany("INSERT INTO codes VALUES (?)",[(1,),(2,)])
    gold={(1,),(2,)}
    bag=Counter({(1,):1,(2,):1})
    assert strict_score(db,"SELECT k FROM codes",gold,bag)["ex"]
    assert not strict_score(db,"SELECT 1 FROM codes",gold,bag)["ex"]
    assert not strict_score(db,"DELETE FROM codes",gold,bag)["execution_ok"]


def test_no_partial_source_score_promotion(tmp_path):
    f=tmp_path/"partial.json"
    f.write_text(json.dumps({"status":"complete","shard":0}))
    with pytest.raises(ValueError,match="eight physical"):
        aggregate([f])
    with pytest.raises(RuntimeError,match="SHA drift"):
        load_frozen(f,f)


def test_replication_uses_new_frozen_cohort_digest():
    assert COHORT_SHA=="faa4f7ba1de9923cbf8e51a500727dfe4c76a08c3f6366735cf21bff152655b7"
    assert PROTOCOL=="BIRD-TRAIN-SECOND-UNSEEN-DB4X4-VALUE-EVIDENCE-V1"
