from __future__ import annotations
import json
import sqlite3
from pathlib import Path

import pytest

from experiments.qwen9b_birdtrain_valid64_baseline import (
    SOURCE_SHA,aggregate,gold_score,split_prompt,
)

def test_split_exact_original_system_user_contract():
    q, schema, evidence = split_prompt(
        "Question: How many rows?\nSchema context:\nPhysical SQLite schema:\na(id)"
    )
    assert (q,schema,evidence)==("How many rows?","Physical SQLite schema:\na(id)",None)
    q, schema, evidence = split_prompt(
        "Question: Find some\nSchema context:\nSchema\nExternal evidence:\ncode is X"
    )
    assert (q,schema,evidence)==("Find some","Schema","code is X")
    with pytest.raises(ValueError):
        split_prompt("BAD INPUT")

def test_strict_physical_sql_execution_not_string_match(tmp_path):
    db=tmp_path/"audit.sqlite"
    with sqlite3.connect(db) as cx:
        cx.execute("CREATE TABLE t(k INTEGER)")
        cx.executemany("INSERT INTO t VALUES (?)", [(1,),(2,)])
    assert gold_score(db,"SELECT k FROM t",{(1,),(2,)},
                      __import__("collections").Counter({(1,):1,(2,):1}))["ex"]
    assert not gold_score(db,"SELECT 999 FROM t",{(1,),(2,)},
                          __import__("collections").Counter({(1,):1,(2,):1}))["ex"]
    assert not gold_score(db,"DELETE FROM t",{(1,),(2,)},
                          __import__("collections").Counter({(1,):1,(2,):1}))["execution_ok"]

def test_aggregation_refuses_missing_or_duplicate_shards(tmp_path):
    d={"status":"complete","split":"BIRD TRAIN-only, DB-disjoint original publisher SFT-validation64",
       "validation_sha256":SOURCE_SHA,
       "model_digest":"56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90",
       "model":"qwen3.5:9b-q4_K_M","shard":0,"n":8,"database_id":"db",
       "no_training":True,"no_dev300":True,"cases":[]}
    p=tmp_path/"shard.json";p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match="8 shards"):
        aggregate([p])
