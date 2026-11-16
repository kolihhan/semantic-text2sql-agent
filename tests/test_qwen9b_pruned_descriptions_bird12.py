import json
from pathlib import Path

import pytest

from evaluation.qwen9b_pruned_descriptions_bird12 import (
    PROTOCOL, MODEL_DIGEST, aggregate, exclusion_ids,
)


def fake_case(index,original,pruned):
    return {
        "case_id":str(index),"database_id":f"db{index}",
        "reference_scorable":True,
        "arms":{
            "original":{"strict_ex":original,"strict_bag":original},
            "pruned_descriptions":{"strict_ex":pruned,"strict_bag":pruned},
        },
    }


def make_shards(records):
    return [
        {"protocol":PROTOCOL,"model_digest":MODEL_DIGEST,
         "cohort_sha256":"cohort-source-frozen","shard":s,
         "cases":records[s*3:(s+1)*3]}
        for s in range(4)
    ]


def test_success_requires_three_rescues_in_three_dbs_with_no_regressions():
    rows=[fake_case(i,original=i<4,pruned=i<7) for i in range(12)]
    x=aggregate(make_shards(rows))
    assert x["screen_positive_only"] is True
    assert x["net_rescues"]==3
    assert x["promoted"] is False
    rows[0]=fake_case(0,True,False)
    y=aggregate(make_shards(rows))
    assert y["screen_positive_only"] is False
    assert y["regressions"]==["0"]


def test_no_source_overlap_and_exact_124_previous_public_identities(tmp_path):
    ledger=tmp_path/"base.json"
    full=tmp_path/"full.json"
    spider=tmp_path/"spider.json"
    ledger.write_text(json.dumps({
        "status":"complete","sample_size":100,
        "cases":[{"case_id":str(i)} for i in range(100)],
    }))
    full.write_text(json.dumps({
        "protocol":"original-qwen9b-full-physical-schema-vs-lexical-bird12-v1",
        "cases":[{"case_id":str(i)} for i in range(100,112)],
    }))
    spider.write_text(json.dumps({
        "protocol":"same-qwen9b-spider-train-structure-two-examples-bird12-v1",
        "cases":[{"case_id":str(i)} for i in range(112,124)],
    }))
    assert len(exclusion_ids(ledger,full,spider))==124
    a=json.loads(spider.read_text())
    a["cases"][0]["case_id"]="110"
    spider.write_text(json.dumps(a))
    with pytest.raises(ValueError,match="124"):
        exclusion_ids(ledger,full,spider)


def test_corrupt_pair_or_wrong_model_fails_closed():
    rows=[fake_case(i,True,True) for i in range(12)]
    shards=make_shards(rows)
    shards[0]["model_digest"]="other model"
    with pytest.raises(ValueError,match="integrity"):
        aggregate(shards)
