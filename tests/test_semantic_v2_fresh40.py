"""No GPU/model required: fresh40 sample independence and frozen paired scoring."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from evaluation.semantic_v2_fresh40 import COHORT_COUNTS, aggregate, manifest, source_cohort_ids


def _fixtures(tmp_path:Path):
    files=[]
    for i,n in enumerate(COHORT_COUNTS):
        cases=[{"case_id":str(sum(COHORT_COUNTS[:i])+j)} for j in range(n)]
        p=tmp_path/f"prior-{i}.json"
        p.write_text(json.dumps({"cases":cases}),encoding="utf8")
        files.append(p)
    src=tmp_path/"dev.json"
    src.write_text(json.dumps([
        {
            "question_id":str(460+i),
            "db_id":f"database_{(i//2)%12}",
            "question":(
                f"Find matches for 'Entity {i}' in test."
                if i%2==0 else f"How many numbered values for record {i}?"
            ),
            "evidence":"",
            "SQL":"SELECT 1",
        } for i in range(120)
    ]),encoding="utf8")
    return files,src


def test_prior_identity_exclusion_is_exact_and_disjoint(tmp_path):
    files,src=_fixtures(tmp_path)
    ids=source_cohort_ids(files)
    assert len(ids)==460
    assert "0" in ids and "459" in ids
    p=manifest(source=src,excluded=files)
    assert p["excluded_identity_count"]==460
    assert p["group_counts"]=={"quoted":20,"no_quote":20}
    assert p["gold_used_in_selection"] is False
    assert len({x["case_id"] for x in p["selected"]})==40
    assert not {x["case_id"] for x in p["selected"]}&ids
    assert min(p["source_database_count"].values())>=7
    assert p==manifest(source=src,excluded=files)


def test_missing_prior_evidence_fails_closed(tmp_path):
    files,src=_fixtures(tmp_path)
    with pytest.raises(ValueError,match="six"):
        source_cohort_ids(files[:5])
    payload=json.loads(files[5].read_text())
    payload["cases"][0]["case_id"]="1"
    files[5].write_text(json.dumps(payload))
    with pytest.raises(ValueError,match="duplicate or overlapping"):
        manifest(source=src,excluded=files)


def test_mismatched_or_missing_8_shards_never_claim_a_result():
    with pytest.raises(ValueError,match="8 unique"):
        aggregate([])


def test_paired_transitions_are_accounted_for_and_not_promoted():
    rows=[]
    for i in range(40):
        a=i<15
        v=(i<14) or i==15 or i==16
        rows.append({
            "case_id":str(i),"database_id":f"db{i//5}",
            "group":"quoted" if i<20 else "no_quote",
            "baseline_sql":"SELECT 1","v2_sql":"SELECT 2",
            "baseline_ex":a,"v2_ex":v,
            "v2_changed":i in (14,15,16),
            "v2_reason":"synthetic_test",
        })
    input_data=[]
    for shard in range(8):
        input_data.append({
            "schema":"sql-semantic-v2-new40-shard/v1",
            "shard":shard,"model":"qwen3.5:9b-q4_K_M",
            "seed":42,"manifest_sha256":"frozen",
            "gold_used_during_generation_or_selection":False,
            "cases":rows[shard::8],
        })
    result=aggregate(input_data)
    assert result["overall"]["count"]==40
    assert result["overall"]["baseline_correct"]==15
    assert result["overall"]["v2_correct"]==16
    assert result["overall"]["changed"]==3
    assert result["overall"]["rescued"]==2
    assert result["overall"]["regressed"]==1
    assert result["overall"]["net_gain"]==1
    assert result["promote"] is False
    assert result["paired_shared_baseline_generation"] is True
    input_data[0]["cases"][0]["case_id"]=input_data[1]["cases"][0]["case_id"]
    with pytest.raises(ValueError,match="duplicate paired"):
        aggregate(input_data)
