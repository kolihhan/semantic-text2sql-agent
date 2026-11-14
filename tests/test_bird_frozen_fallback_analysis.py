import pytest

from evaluation.bird_frozen_fallback_analysis import analyze, select_without_gold


def arm(valid, ex=False):
    return {"preflight": valid, "execution_success": valid,
            "ex": ex, "bag_equal": ex}


def shard(index):
    cases = []
    for i in range(10):
        xi, qwen = arm(True, True), arm(True, False)
        if i == 1:
            xi, qwen = arm(False, False), arm(True, True)
        cases.append({"case_id": str(index * 10 + i), "database_id": "db",
                      "arms": {"xiyan": xi, "qwen": qwen}})
    return {"shard": index, "count": 10, "models": {"xiyan": "xi", "qwen": "q"},
            "cohort_sha256": "fixed", "cases": cases}


def test_fallback_does_not_see_gold_or_change_valid_primary():
    good = {"xiyan": arm(True, False), "qwen": arm(True, True)}
    assert select_without_gold(good) == ("xiyan", "primary_ok")
    good["xiyan"]["ex"] = True
    assert select_without_gold(good) == ("xiyan", "primary_ok")
    bad = {"xiyan": arm(False), "qwen": arm(True, True)}
    assert select_without_gold(bad) == ("qwen", "primary_failed")


def test_partial_requires_explicit_override_and_reports_fallback():
    with pytest.raises(ValueError, match="requires ten"):
        analyze([shard(0)])
    result = analyze([shard(0)], partial=True)
    assert result["status"] == "partial"
    assert result["metrics"]["xiyan_ex"] == 9
    assert result["metrics"]["selected_ex"] == 10
    assert result["metrics"]["fallback_ex_rescues"] == 1
    assert result["metrics"]["fallback_ex_regressions"] == 0


def test_rejects_duplicate_shard_and_duplicate_case():
    with pytest.raises(ValueError, match="duplicate"):
        analyze([shard(0), shard(0)], partial=True)
    shards = [shard(0), shard(1)]
    shards[1]["cases"][0]["case_id"] = shards[0]["cases"][0]["case_id"]
    with pytest.raises(ValueError, match="duplicate or incomplete"):
        analyze(shards, partial=True)


def test_complete_ten_shards():
    result = analyze([shard(i) for i in range(10)])
    assert result["status"] == "complete"
    assert result["cases"] == 100
    assert result["metrics"]["selected_ex"] == 100
