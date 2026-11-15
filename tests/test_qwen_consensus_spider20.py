"""Same-Qwen ensemble is a falsifiable source-shift hypothesis, not a gold oracle."""
from collections import Counter

import pytest

from evaluation.qwen_consensus_spider20 import (
    select_cases, choose_consensus, aggregate, PROTOCOL, SEED_OLD, SEED_IR,
    _hash,
)


def fake_source(n=130):
    train = [{"db_id": f"domain_{i}", "question": f"Question {i} row {j}",
              "query": "Gold SQL forbidden in selector"}
             for i in range(n) for j in range(3)]
    return train, [{"db_id": "outside_dev"}]


def test_selector_excludes_all_prior_spider_100_plus_typed_ir_10_and_gold():
    train, dev = fake_source()
    chosen = select_cases(train, dev)
    assert len(chosen) == 20
    assert len({x["database_id"] for x in chosen}) == 10
    assert chosen == select_cases(train, dev)
    assert all("query" not in row and "sql" not in row for row in chosen)
    old = set(sorted({x["db_id"] for x in train},
                     key=lambda d: _hash(f"{SEED_OLD}:database:{d}"))[:100])
    fresh_ir = sorted(({x["db_id"] for x in train} - old),
                      key=lambda d: _hash(f"{SEED_IR}:database:{d}"))
    selected = {x["database_id"] for x in chosen}
    assert not (selected & (old | set(fresh_ir[:10])))


def test_selector_fails_closed_without_10_fresh_databases():
    train, dev = fake_source(115)
    with pytest.raises(ValueError, match="fresh"):
        select_cases(train, dev)


def test_consensus_requires_two_equal_successful_executions():
    samples = [
        {"execution_success": True, "rows": [(2, "a"), (2, "a")]},
        {"execution_success": True, "rows": [(1, "b")]},
        {"execution_success": True, "rows": [(2, "a"), (2, "a")]},
    ]
    assert choose_consensus(samples) == (0, 2)
    assert choose_consensus(samples[:2]) == (None, 0)
    samples[2]["rows"] = [(2, "a")]
    assert choose_consensus(samples) == (None, 0)
    samples[0]["execution_success"] = False
    assert choose_consensus(samples) == (None, 0)


def test_aggregate_reports_oracle_headroom_but_never_deploys_by_oracle():
    shards = []
    for shard in range(5):
        cases = []
        for j in range(4):
            i = shard * 4 + j
            baseline = i < 10
            consensus = baseline
            if i in (10, 11, 12):
                consensus = True
            if i == 0:
                consensus = False
            cases.append({
                "case_id": str(i), "db_id": f"db_{i // 2}",
                "reference_scorable": True, "consensus_index": 1 if i % 2 else None,
                "candidate_diversity": 2,
                "total_latency_s": 5.0,
                "baseline": {"strict_ex": baseline, "bag": baseline, "execution_success": True, "latency_s": 1},
                "consensus": {"strict_ex": consensus, "bag": consensus, "execution_success": True, "latency_s": 4},
                "oracle_correct": baseline or consensus,
            })
        shards.append({"protocol": PROTOCOL, "shard": shard,
                       "cohort_sha256": "same", "model_digest": "expected",
                       "cases": cases})
    report = aggregate(shards, expected_digest="expected")
    assert report["baseline_ex"] == 10 and report["consensus_ex"] == 12
    assert report["rescues"] == 3 and report["regressions"] == 1
    assert report["oracle_ex"] == 13
    assert report["promotion_allowed"] is False
    assert report["requested"] == 20 and report["databases"] == 10
    shards[2]["model_digest"] = "unexpected"
    with pytest.raises(ValueError, match="drift"):
        aggregate(shards, expected_digest="expected")
