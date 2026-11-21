"""CPU-only source-lock and no-label leak controls for actual Qwen9B 12-case screen."""
from __future__ import annotations
import json
import pytest

from evaluation import qwen9b_typed_plan_exposed12 as e


def test_exact_model_and_read_only_source_identity():
    assert e.QWEN_MODEL == "qwen3.5:9b-q4_K_M"
    assert e.QWEN_DIGEST == "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
    assert e.DEV100_SOURCE_SHA == "0310b7ca2dfb6247234e2073f93e7a75106ee41ba55231f75d06d2ef68efe2ec"
    assert e.N == 12


def test_frozen_historical_cohort_strips_correctness_and_gold(tmp_path, monkeypatch):
    records = [
        {"question_id": i + 1, "db_id": "db" + str(i%3),
         "question": f"How many records for {i}?", "evidence": "context",
         "SQL": "SELECT 42 AS private_gold"}
        for i in range(100)
    ]
    monkeypatch.setattr(e, "_load_source", lambda _p: records)
    raw = tmp_path / "shards"
    raw.mkdir()
    for k in range(4):
        cases = []
        for i in range(k*25,(k+1)*25):
            cases.append({
                "case_id": str(i+1), "database_id": "db"+str(i%3),
                "question": records[i]["question"],
                "guarded": {
                    "final_sql": "SELECT 1 AS original_not_gold",
                    "status": "ok", "official_ex": i%2 == 0,
                },
                "direct": {"official_ex": i%3==0},
                "shared_initial_sql": "SELECT 2",
            })
        (raw / f"qwen9b-shard-{k}.json").write_text(json.dumps({
            "status": "invalid" if k==3 else "complete",
            "source_sha256":e.DEV100_SOURCE_SHA,
            "model_digest":e.QWEN_DIGEST,
            "sample_size":25, "cases":cases,
        }))
    result = e.freeze(source=tmp_path/"unused-source",shards=raw,output=tmp_path/"sealed.json")
    assert result["status"] == "FROZEN_GOLD_FREE"
    assert len(result["selected"]) == 12
    assert len({x["case_id"] for x in result["selected"]})==12
    assert all(x["case_id"] not in {str(i+1) for i in range(75,100)} for x in result["selected"])
    serialized=(tmp_path/"sealed.json").read_text()
    assert "private_gold" not in serialized
    assert "official_ex" not in serialized
    assert "shared_initial_sql" not in serialized
    assert "SELECT 2" not in serialized
    assert all(x["historical_guarded_sql"]=="SELECT 1 AS original_not_gold" for x in result["selected"])


def test_frozen_sample_refuses_historial_model_drift(tmp_path,monkeypatch):
    monkeypatch.setattr(e,"_load_source",lambda _:[])
    root=tmp_path/"shards";root.mkdir()
    for i in range(4):
        (root / f"qwen9b-shard-{i}.json").write_text(json.dumps({
            "status":"complete","source_sha256":e.DEV100_SOURCE_SHA,
            "model_digest":"not_original_model","sample_size":25,"cases":[],
        }))
    with pytest.raises(ValueError,match="model/source"):
        e.freeze(source=tmp_path/"no-source",shards=root,output=tmp_path/"no-result")


def test_score_refuses_early_gold_access_and_mismatched_source(tmp_path):
    f=tmp_path/"frozen.json"
    f.write_text(json.dumps({"selected":[{"case_id":"1"}]*12}))
    b=tmp_path/"blind.json"
    b.write_text(json.dumps({"status":"INCOMPLETE","reference_sql_opened":False,"cases":[]}))
    with pytest.raises(ValueError,match="incomplete"):
        e.score(frozen=f,blind=b,source=tmp_path/"no-source",db_root=tmp_path,out=tmp_path/"score")
