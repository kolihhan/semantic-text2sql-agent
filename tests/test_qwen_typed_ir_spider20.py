"""Frozen source-only Spider TRAIN selector and paired-score contract for Qwen IR."""
from evaluation.qwen_typed_ir_spider20 import select_cases, aggregate


def fake_train(n=118):
    return [{"db_id": f"domain_{i}", "question": f"Find category {i} item {j}",
             "query": "THIS REFERENCE SQL MUST NEVER ENTER A MANIFEST"}
            for i in range(n) for j in (0, 1, 2)]


def test_selector_excludes_prior_100_databases_and_does_not_expose_gold():
    train = fake_train()
    chosen = select_cases(train, [])
    assert len(chosen) == 20
    assert len({x["database_id"] for x in chosen}) == 10
    assert all("query" not in x and "sql" not in x for x in chosen)
    assert all(sum(x["database_id"] == db for x in chosen) == 2
               for db in {x["database_id"] for x in chosen})
    assert chosen == select_cases(train, [])
    assert len(set(x["case_id"] for x in chosen)) == 20


def test_selector_rejects_too_few_fresh_databases_and_dev_overlap():
    import pytest
    with pytest.raises(ValueError, match="fresh"):
        select_cases(fake_train(105), [])
    with pytest.raises(ValueError, match="overlap"):
        select_cases(fake_train(), [{"db_id": "domain_0"}])


def test_pair_aggregate_counts_all_cases_and_prevents_optimistic_claims():
    shards = []
    for shard in range(5):
        rows = []
        for j in range(4):
            i = shard * 4 + j
            a = i < 13
            b = a
            if i == 13: b = True
            if i == 0: b = False
            rows.append({
                "case_id": str(i), "db_id": f"d{i//2}", "reference_scorable": True,
                "typed_ir_compiled": i % 2 == 0,
                "arms": {
                    "original_guarded": {"strict_ex": a, "bag": a, "execution_success": True, "latency_s": 1},
                    "typed_ir_guarded": {"strict_ex": b, "bag": b, "execution_success": True, "latency_s": 2},
                }
            })
        shards.append({"shard": shard, "cohort_sha256": "x", "model_digest": "m",
                       "protocol": "qwen9b-typed-ir-spider-shift20-v1", "cases": rows})
    report = aggregate(shards)
    assert report["rescues"] == 1 and report["regressions"] == 1
    assert report["arms"]["original_guarded"]["strict_ex"] == 13
    assert report["arms"]["typed_ir_guarded"]["strict_ex"] == 13
    assert report["promotion_allowed"] is False
