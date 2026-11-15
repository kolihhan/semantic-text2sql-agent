"""Freeze-only source contract for a no-training Qwen-only bigger-model screen."""
import json
from hashlib import sha256

import pytest
from evaluation import colab_qwen3_14b_bird12_screen as mod


def pair(tmp_path,monkeypatch):
    cases=[
        {"case_id":str(i),"database_id":f"db_{i//2}",
         "question":f"How many objects belong to group {i}?",
         "evidence":""} for i in range(12)
    ]
    src_hash=sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest()
    source={
        "cases":cases,"cases_sha":src_hash,"n":12,"db_count":6,
        "gold_used_to_select":False,
        "independent_question_ids_from_exposed_160":True,
        "digest":mod.BASE_QWEN9B_DIGEST
    }
    prior={
        "protocol":"qwen9b-native-batched-tool-observation-versus-guarded-bird-dev12-v2",
        "n":12,"arms":{"original":{"strict_set_ex":3,"strict_bag_ex":3}},
        "cases":[{**r,"arms":{"original":{
            "sql":"SELECT 1","status":"ok","set_ex":i<3,
            "bag_ex":i<3,"model_calls":1
        }}} for i,r in enumerate(cases)]
    }
    m=tmp_path/"source.json"
    b=tmp_path/"original.json"
    m.write_text(json.dumps(source))
    b.write_text(json.dumps(prior))
    monkeypatch.setattr(mod,"COHORT",m)
    monkeypatch.setattr(mod,"ORIGINAL",b)
    monkeypatch.setattr(mod,"COHORT_SHA",src_hash)
    monkeypatch.setattr(mod,"SOURCE_CODE_SHA","0"*40)
    return m,b


def test_existing_original_qwen9b_source_remains_pinned_without_gold(tmp_path,monkeypatch):
    pair(tmp_path,monkeypatch)
    rows,baselines=mod.source_inputs()
    assert len(rows)==12 and len(baselines)==12
    assert all("SQL" not in r and "answer" not in r for r in rows)
    assert len({r["database_id"] for r in rows})==6
    assert sum(c["arms"]["original"]["set_ex"] for c in (json.loads(mod.ORIGINAL.read_text())["cases"]))==3


def test_original_model_question_evidence_inconsistency_rejected(tmp_path,monkeypatch):
    _,b=pair(tmp_path,monkeypatch)
    d=json.loads(b.read_text())
    d["cases"][0]["evidence"]="injected reference answer"
    b.write_text(json.dumps(d))
    with pytest.raises(RuntimeError,match="question/schema/evidence drift"):
        mod.source_inputs()


def test_refuse_previous_predicted_results_from_wrong_cohort(tmp_path,monkeypatch):
    m,_=pair(tmp_path,monkeypatch)
    d=json.loads(m.read_text())
    d["cases"][3]["case_id"]="modified after freeze"
    m.write_text(json.dumps(d))
    with pytest.raises(RuntimeError,match="cohort changed"):
        mod.source_inputs()


def test_pinned_same_qwen_family_and_strict_semantic_evaluation():
    s=__import__("pathlib").Path(mod.__file__).read_text()
    assert mod.MODEL=="Qwen/Qwen3-14B"
    assert "enable_thinking=False" in s
    assert 'max_repairs=2' in s
    assert "max_rows=MAX_ROWS" in s
    assert "gold_used_to_select" in s
    assert "set(map(tuple,pred))==set(map(tuple,ref.rows))" in s
    assert "Counter(map(tuple,pred))==Counter(map(tuple,ref.rows))" in s
    assert "positive_screen_only" in s and "promoted" in s
