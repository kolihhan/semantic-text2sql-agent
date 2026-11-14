import pytest

from evaluation.bird_frozen100_generator_compare import aggregate, digest_cohort


def _fake_shards():
    cohort = [
        {"case_id": str(i), "database_id": "db", "question": f"q{i}"}
        for i in range(100)
    ]
    digest = digest_cohort(cohort)
    shards = []
    for shard in range(10):
        cases = []
        for i in range(shard, 100, 10):
            def arm(correct):
                return {"ex": correct, "bag_equal": correct,
                        "execution_success": True, "preflight": True,
                        "model_calls": 1, "latency_s": 2.0}
            cases.append({
                **cohort[i],
                "arms": {"qwen": arm(i < 40), "xiyan": arm(i < 65)}
            })
        shards.append({"shard": shard, "count": 10,
                       "cohort_sha256": digest,
                       "models": {"qwen": "q", "xiyan": "x"},
                       "cases": cases})
    return shards


def test_frozen_aggregate_pairs_100_unique_cases_and_transitions():
    result = aggregate(_fake_shards())
    assert result["count"] == 100
    assert result["arms"]["qwen"]["ex"] == 40
    assert result["arms"]["xiyan"]["ex"] == 65
    assert result["ex_rescues_regressions"] == (25, 0)


def test_frozen_aggregate_rejects_missing_shard():
    with pytest.raises(ValueError, match="missing or duplicate shard"):
        aggregate(_fake_shards()[:-1])


def test_frozen_aggregate_rejects_cohort_drift():
    shards = _fake_shards()
    shards[2]["cohort_sha256"] = "bad"
    with pytest.raises(ValueError, match="cohort/model/protocol drift"):
        aggregate(shards)


def test_frozen_aggregate_rejects_duplicate_case():
    shards = _fake_shards()
    shards[0]["cases"][0]["case_id"] = shards[1]["cases"][0]["case_id"]
    with pytest.raises(ValueError, match="missing/duplicate frozen cases"):
        aggregate(shards)
