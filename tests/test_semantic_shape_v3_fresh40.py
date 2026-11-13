"""Frozen independent count-intent evaluation protocol tests (no Qwen needed)."""
from __future__ import annotations

import json
from pathlib import Path
import pytest
from evaluation.semantic_shape_v3_fresh40 import COHORT_COUNTS, source_cohort_ids, manifest, aggregate


def prior_and_source(tmp_path:Path):
    prior=[]
    for i,n in enumerate(COHORT_COUNTS):
        ids=[{"case_id":str(sum(COHORT_COUNTS[:i])+j)} for j in range(n)]
        p=tmp_path/f"prior{i}.json"
        p.write_text(json.dumps({"selected":ids} if i==6 else {"cases":ids}))
        prior.append(p)
    source=tmp_path/"bird_dev.json"
    source.write_text(json.dumps([{
        "question_id":str(500+i),
        "db_id":f"database_{(i//2)%12}",
        "question":(
            f"How many events happened in district {i}?"
            if i%2==0 else f"Find event records for district {i}."
        ),
        "SQL":"SELECT 1","evidence":"",
    } for i in range(120)]))
    return prior,source


def test_new_40_frozen_selection_excludes_all_previous_500(tmp_path):
    paths,source=prior_and_source(tmp_path)
    existing=source_cohort_ids(paths)
    assert len(existing)==500
    assert "0" in existing and "499" in existing
    m=manifest(source=source,excluded=paths)
    assert m["excluded_identity_count"]==500
    assert m["group_counts"]=={"count_intent":20,"control":20}
    assert m["gold_used_in_selection"] is False
    assert len({x["case_id"] for x in m["selected"]})==40
    assert not {x["case_id"] for x in m["selected"]}&existing
    assert min(m["source_database_count"].values())>=7
    assert m==manifest(source=source,excluded=paths)


def test_one_previous_identity_overlap_is_rejected(tmp_path):
    paths,source=prior_and_source(tmp_path)
    with pytest.raises(ValueError,match="seven"):
        source_cohort_ids(paths[:-1])
    final=json.loads(paths[-1].read_text())
    final["selected"][0]["case_id"]="50"
    paths[-1].write_text(json.dumps(final))
    with pytest.raises(ValueError,match="duplicate or overlapping"):
        manifest(source=source,excluded=paths)


def test_aggregate_refuses_missing_unmatched_or_duplicate_shards():
    with pytest.raises(ValueError,match="8 unique"):
        aggregate([])


def test_aggregate_requires_exact_paired_transitions():
    allrows=[]
    for i in range(40):
        before=i<15
        after=(i<14) or i in (15,16)
        allrows.append({
            "case_id":str(500+i),"group":"count_intent" if i<20 else "control",
            "database_id":f"db{i//5}", "baseline_ex":before,
            "v3_ex":after,"v3_changed":i in (14,15,16),
            "v3_reason":"synthetic", "baseline_sql":"SELECT 1","v3_sql":"SELECT 2",
        })
    shards=[{
        "schema":"sql-semantic-v3-new40-shard/v1",
        "shard":i,"model":"qwen3.5:9b-q4_K_M","seed":42,
        "manifest_sha256":"same","gold_used_during_generation_or_selection":False,
        "cases":allrows[i::8],
    } for i in range(8)]
    result=aggregate(shards)
    overall=result["overall"]
    assert overall["count"]==40
    assert overall["baseline_correct"]==15
    assert overall["v3_correct"]==16
    assert overall["rescued"]==2 and overall["regressed"]==1
    assert overall["net_gain"]==1
    assert result["promote"] is False
    shards[0]["cases"][0]["case_id"]=shards[1]["cases"][0]["case_id"]
    with pytest.raises(ValueError,match="duplicate paired"):
        aggregate(shards)
