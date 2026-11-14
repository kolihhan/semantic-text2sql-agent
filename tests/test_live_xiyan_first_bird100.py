import pytest

from evaluation.live_xiyan_first_bird100 import aggregate, MODELS


def fake_shard(index):
    cases = []
    for i in range(10):
        cases.append({
            "case_id": str(index * 10 + i), "ex": i == 0, "bag": i == 0,
            "execution_success": True,
            "route": "qwen_fallback:verification_failed" if i == 1 else "xiyan_primary_valid",
            "xiyan_calls": 1, "qwen_calls": 1 if i == 1 else 0,
            "latency_s": 1.0,
        })
    return {"shard": index, "cohort_sha": "frozen",
            "models": {k: v[1] for k, v in MODELS.items()}, "cases": cases}


def test_aggregate_paired_100_and_fallback_latency_counts():
    result = aggregate([fake_shard(i) for i in range(10)])
    assert result["count"] == 100
    assert result["ex"] == 10 and result["bag"] == 10
    assert result["routes"] == {"xiyan_primary_valid": 90, "qwen_fallback": 10}
    assert result["model_calls"] == {"xiyan": 100, "qwen": 10}
    assert result["mean_latency_s"] == 1.0


def test_aggregate_fails_closed_for_partial_or_duplicate():
    with pytest.raises(ValueError):
        aggregate([fake_shard(0)])
    cases = [fake_shard(i) for i in range(10)]
    cases[1]["cases"][0]["case_id"] = cases[0]["cases"][0]["case_id"]
    with pytest.raises(ValueError, match="100 unique"):
        aggregate(cases)
