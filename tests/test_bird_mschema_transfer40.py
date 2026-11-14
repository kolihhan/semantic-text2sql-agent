import pytest

from evaluation.bird_mschema_transfer40 import ARMS, aggregate


def make_shard(shard):
    rows = []
    for i in range(4):
        caseid = shard * 4 + i
        arms = {}
        for arm in ARMS:
            correct = arm == "mschema_examples" or (arm == "lexical" and i == 0)
            arms[arm] = {
                "sql": "SELECT 1", "ex": correct, "bag_equal": correct,
                "execution_success": True, "preflight": True, "latency_s": 2.0,
            }
        rows.append({
            "case_id": caseid, "database_id": "db-" + str(caseid // 2),
            "question": "how many?", "evidence": "",
            "present_chars": {key: 100 for key in ARMS},
            "selection_hash": "fixed", "arms": arms,
        })
    return {"shard": shard, "model": "same", "cohort_sha256": "cohort",
            "cases": rows}


def test_complete_40_case_paired_aggregate():
    summary = aggregate([make_shard(i) for i in range(10)])
    assert summary["count"] == 40
    assert summary["arms"]["lexical"]["ex"] == 10
    assert summary["arms"]["mschema_examples"]["ex"] == 40
    assert summary["paired"]["lexical->mschema_examples"] == {
        "rescues": 30, "regressions": 0,
    }
    assert summary["gold_available_only_after_generation"] is True


def test_reject_missing_shard_case_model_or_arm():
    with pytest.raises(ValueError, match="ten unique"):
        aggregate([make_shard(0)])
    shards = [make_shard(i) for i in range(10)]
    shards[1]["cases"][0]["case_id"] = 0
    with pytest.raises(ValueError, match="incomplete"):
        aggregate(shards)
    shards = [make_shard(i) for i in range(10)]
    shards[0]["cases"][0]["arms"].pop("mschema_examples")
    with pytest.raises(ValueError, match="arms missing"):
        aggregate(shards)
