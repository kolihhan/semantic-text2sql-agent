import pytest

from evaluation.bird_qwen_semantic_plan40 import aggregate, ARMS


def shard(index):
    cases = []
    for j in range(4):
        baseline, planned = (j % 2 == 0), (j % 2 == 1)
        cases.append({
            "case_id": 100 * index + j, "plan_valid": True,
            "latency_s": 3.0, "arms": {
                ARMS[0]: {"ex": baseline, "bag": baseline,
                          "execution_success": True, "attempt": 0},
                ARMS[1]: {"ex": planned, "bag": planned,
                          "execution_success": True, "attempt": 1},
            },
        })
    return {"shard": index, "digest": "qwen-pinned",
            "cohort_hash": "train40", "cases": cases}


def test_full_paired_aggregate():
    result = aggregate([shard(i) for i in range(10)])
    assert result["n"] == 40 and result["no_gold_routing"]
    assert result["arms"][ARMS[0]]["ex"] == 20
    assert result["arms"][ARMS[1]]["ex"] == 20
    assert (result["rescues"], result["regressions"], result["net"]) == (20, 20, 0)


def test_reject_partial_or_duplicate():
    with pytest.raises(ValueError):
        aggregate([shard(0)])
    x = [shard(i) for i in range(10)]
    x[1]["cases"][0]["case_id"] = x[0]["cases"][0]["case_id"]
    with pytest.raises(ValueError):
        aggregate(x)
