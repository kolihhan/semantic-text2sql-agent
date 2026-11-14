"""Bounded A5 plan and paired transfer integrity checks."""
import json
import pytest
from evaluation.bird_a5_plan_transfer40 import aggregate, parse_plan, plan_prompt


def test_frozen_plan_json_contract():
    row = {"answer_entity": "customers", "relation_role": "buyer",
           "aggregation_grain": "one customer", "filters": "date after 2020"}
    assert parse_plan(json.dumps(row)) == row
    assert parse_plan("SELECT 1") is None
    assert parse_plan('{"answer_entity": "a"}') is None
    assert parse_plan(json.dumps({**row, "SQL": "SELECT 1"})) is None
    assert parse_plan("x"*4001) is None
    assert "INITIATED" in plan_prompt("Which customer?", "customers(id)")


def fake(shard):
    return {
        "shard": shard, "model": "frozen", "cohort_sha256": "frozen cohort",
        "cases": [{
            "case_id": f"{shard}-{i}", "plan_valid": True,
            "generation_latency_s": 2.0,
            "arms": {
                "a0": {"ex": i < 2, "bag": i < 2, "execution_success": True, "sql": "SELECT 1"},
                "a5": {"ex": i < 3, "bag": i < 3, "execution_success": True, "sql": "SELECT 2"},
            },
        } for i in range(4)],
    }


def test_all_40_paired_cases_and_invariant():
    out = aggregate([fake(i) for i in range(10)])
    assert out["count"] == 40
    assert out["arms"]["a0"]["ex"] == 20 and out["arms"]["a5"]["ex"] == 30
    assert out["rescues"] == 10 and out["regressions"] == 0
    assert out["changed_sql"] == 40
    with pytest.raises(ValueError, match="ten unique"):
        aggregate([fake(0)])
    shards = [fake(i) for i in range(10)]
    shards[1]["cases"][0]["case_id"] = "0-0"
    with pytest.raises(ValueError, match="incomplete"):
        aggregate(shards)
