"""BLIND baseline construction and independent-model paired-ex contracts."""
import pytest
from experiments.bird_diverse_backbone import extract_blinded_baseline, aggregate


def test_historical_outcomes_and_old_alternative_can_never_enter_prompt():
    full={"schema":"bird-fk-bridge-aggregate-v1",
          "cohort_sha256":"test-cohort",
          "cases":[{"case_id":str(i),"database_id":"db",
                    "sql":{"a0":"SELECT 1","a1":"SELECT gold-leaking-answer"},
                    "ex":{"a0":i<64,"a1":i<59},
                    "extra_hints":"do_not_leak"}
                   for i in range(100)]}
    masked=extract_blinded_baseline(full)
    assert len(masked["cases"])==100
    assert "ex" not in str(masked)
    assert "SELECT gold-leaking-answer" not in str(masked)
    assert all(set(r)=={"case_id","database_id","xiyan_a0_sql"} for r in masked["cases"])
    full["cases"][0]["sql"].pop("a0")
    with pytest.raises(ValueError):
        extract_blinded_baseline(full)


def test_two_backbone_oracle_and_nonpromoted_score():
    all_shards=[]
    for shard in range(4):
        rows=[]
        for local in range(25):
            i=shard*25+local
            a0=i<64
            candidate=(i<64 and i>=4) or (i in {64,65,66,67,68,69,70,71})
            rows.append({"case_id":str(i),"ex_a0":a0,"ex_coder":candidate,
                         "changed":True})
        all_shards.append({
            "shard":shard,"schema":"bird-diverse-backbone-shard-v1",
            "model":"qwen2.5-coder:7b-instruct-q4_K_M",
            "cohort_sha256":"test-cohort",
            "gold_visible_in_generation":False,
            "cases":rows})
    result=aggregate(all_shards)
    assert result["xiyan_correct"]==64
    assert result["coder_correct"]==68
    assert result["wrong_to_correct"]==8
    assert result["correct_to_wrong"]==4
    assert result["candidate_oracle_correct"]==72
    assert result["automatic_selector_ran"] is False
    all_shards[0]["gold_visible_in_generation"]=True
    with pytest.raises(ValueError):
        aggregate(all_shards)
