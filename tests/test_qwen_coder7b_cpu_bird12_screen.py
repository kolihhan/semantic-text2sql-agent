"""Synthetic scorer checks; real BIRD strict EX comes from physical SQLite only."""
import copy

import pytest

from evaluation.qwen_coder7b_cpu_bird12_screen import (
    aggregate, PROTOCOL, COHORT_SHA, MODEL, PUBLISHED_DIGEST_PREFIX,
)


def fake_parts():
    parts = []
    for shard in range(3):
        cases = []
        for index in range(4):
            n = shard * 4 + index
            correct = n in (0, 1, 2)
            cases.append({
                "case_id": str(n),
                "database_id": "db" + str(n // 2),
                "original9b": {"set_ex": correct, "bag_ex": correct, "sql": "SELECT 1"},
                "coder7b": {"set_ex": correct, "bag_ex": correct, "sql": "SELECT 1"},
            })
        parts.append({
            "protocol": PROTOCOL, "model": MODEL,
            "digest": PUBLISHED_DIGEST_PREFIX + "frozen-test-digest",
            "source_cohort_sha": COHORT_SHA, "no_training": True,
            "shard": shard, "cases": cases,
        })
    return parts


def test_stable_control_and_zero_new_gains_never_promote():
    report = aggregate(fake_parts())
    assert report["original_set_ex"] == report["coder7b_set_ex"] == 3
    assert report["positive_screen_only"] is False
    assert report["promoted"] is False
    assert report["no_sql_training"] is True


def test_positive_screen_is_not_production_promotion():
    parts = fake_parts()
    # Three rescues on three different new DBs; no loss of original successes.
    for index in (6, 8, 10):
        item = parts[index // 4]["cases"][index % 4]["coder7b"]
        item["set_ex"] = item["bag_ex"] = True
    report = aggregate(parts)
    assert report["positive_screen_only"] is True
    assert report["promoted"] is False
    assert report["distinct_rescue_databases"] == 3


def test_regression_blocks_positive_gate_even_with_more_rescues():
    parts = fake_parts()
    for index in (4, 6, 8, 10):
        item = parts[index // 4]["cases"][index % 4]["coder7b"]
        item["set_ex"] = item["bag_ex"] = True
    parts[0]["cases"][0]["coder7b"]["set_ex"] = False
    parts[0]["cases"][0]["coder7b"]["bag_ex"] = False
    report = aggregate(parts)
    assert report["coder7b_set_ex"] > 3
    assert report["positive_screen_only"] is False


def test_rejects_incomplete_or_cross_model_or_missing_reference():
    parts = fake_parts()
    with pytest.raises(ValueError):
        aggregate(parts[:2])
    bad = copy.deepcopy(parts)
    bad[1]["digest"] = "wrong"
    with pytest.raises(ValueError):
        aggregate(bad)
    bad = copy.deepcopy(parts)
    bad[1]["cases"][0]["coder7b"]["set_ex"] = None
    with pytest.raises(ValueError):
        aggregate(bad)
