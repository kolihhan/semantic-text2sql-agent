"""Historical expanded-Qwen9B evaluation: no hidden reuse or false promotion."""
from __future__ import annotations
import json
import pytest

from evaluation import qwen9b_typed_plan_exposed75 as e
from evaluation import qwen9b_typed_plan_exposed12 as old


def _source75():
    return {
        "protocol":"HISTORICALLY_EXPOSED_DEV100_SCV12_SOURCE_LOCKED_V1",
        "status":"FROZEN_GOLD_FREE", "gold_sql_present":False,
        "model_digest":old.QWEN_DIGEST,
        "source_sha256":old.DEV100_SOURCE_SHA, "sample_size":75,
        "selected":[
            {"case_id":str(i),"db_id":"db"+str(i%5),"question":str(i),
             "historical_guarded_sql":"SELECT 1"}
            for i in range(75)
        ],
    }


def test_splits_are_exact_75_no_duplicates_or_gold(tmp_path):
    full=tmp_path/"frozen.json"
    full.write_text(json.dumps(_source75()))
    seen=[]
    for shard in range(3):
        part=tmp_path/f"shard-{shard}.json"
        e.split75(full,shard,part)
        result=json.loads(part.read_text())
        assert result["status"]=="FROZEN_GOLD_FREE"
        assert result["gold_sql_present"] is False
        assert len(result["selected"])==25
        assert result["exposed75_cohort"]==e.COHORT
        seen.extend(x["case_id"] for x in result["selected"])
    assert seen==[str(i) for i in range(75)]
    assert len(set(seen))==75


def test_no_silent_use_of_invalid_fourth_shard(tmp_path):
    full=tmp_path/"frozen.json"
    p=_source75()
    p["selected"][4]["official_ex"]=True
    full.write_text(json.dumps(p))
    with pytest.raises(AssertionError):
        e.split75(full,0,tmp_path/"should-not-exist.json")
    assert not (tmp_path/"should-not-exist.json").exists()


def test_exact_mcnemar_for_small_paired_counts():
    assert e._exact_p(0,0)==1.0
    assert e._exact_p(0,1)==1.0
    assert e._exact_p(0,5)==0.0625
    assert e._exact_p(0,6)==0.03125


def test_aggregate_refuses_incomplete_75(tmp_path):
    f=tmp_path/"frozen.json"
    f.write_text(json.dumps(_source75()))
    scored=[]
    blinds=[]
    for i in range(3):
        a=tmp_path/f"score{i}.json"
        b=tmp_path/f"blind{i}.json"
        a.write_text(json.dumps({"status":"incomplete"}))
        b.write_text(json.dumps({"status":"INCOMPLETE"}))
        scored.append(a)
        blinds.append(b)
    with pytest.raises(ValueError,match="incomplete"):
        e.aggregate75(frozen=f,scored=scored,blinds=blinds,output=tmp_path/"out.json")
    assert not (tmp_path/"out.json").exists()
