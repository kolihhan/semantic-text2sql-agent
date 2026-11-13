"""Independent source split and paired SQL A5 100-DB aggregation contracts."""
from copy import deepcopy

import pytest

from experiments.confirm_plan_spider_train import select_cases,aggregate


def _source():
    train=[{
        "db_id":f"train_domain_{db:03d}",
        "question":f"Which entities belong to domain {db}, version {j}?",
        "query":f"GOLD_SQL_{db}_{j}",
    } for db in range(130) for j in range(3)]
    dev=[{"db_id":f"dev_domain_{db:03d}","question":"What?",
          "query":"SELECT 1"} for db in range(20)]
    return train,dev


def test_schema_disjoint_balanced_selection_does_not_consult_any_reference_sql():
    train,dev=_source()
    cases=select_cases(train,dev)
    assert len(cases)==100
    assert len({case["database_id"] for case in cases})==100
    assert len({case["case_id"] for case in cases})==100
    assert set(case["database_id"] for case in cases).isdisjoint(
        {row["db_id"] for row in dev})
    assert all("query" not in r and "SQL" not in r and "sql" not in r for r in cases)
    assert all(train[r["row_index"]]["db_id"]==r["database_id"] and
               train[r["row_index"]]["question"]==r["question"] for r in cases)
    changed=deepcopy(train)
    for r in changed:r["query"]="EVEN IF GOLD CHANGES, SELECTION MUST NOT"
    assert select_cases(changed,dev)==cases
    with pytest.raises(ValueError,match="overlap"):
        select_cases(train,dev+[{"db_id":"train_domain_000"}])
    with pytest.raises(ValueError,match="insufficient"):
        select_cases(train[:50],dev)


def _aggregate():
    cohorts=[]
    for shard in range(4):
        cases=[]
        for i in range(25):
            n=shard*25+i
            baseline=n<50
            treatment=(n<50 or n in (50,51,52,53)) and n != 1
            cases.append({
                "case_id":f"newdomain-{n}:{n}",
                "database_id":f"newdomain-{n}",
                "sql":{"a0":"SELECT 1","a5":"SELECT 1"},
                "planner_valid":True,
                "strict":{
                    "a0":{"bag_equal":baseline,
                          "exact_ast_equal":baseline,"both_empty":False},
                    "a5":{"bag_equal":treatment,
                          "exact_ast_equal":treatment,"both_empty":False},
                },
            })
        cohorts.append({
            "schema":"spider-train-database-shift-a0-a5-shard-v1",
            "model":"XiYan-7B-Q4_K_M","cohort_sha256":"frozen-cohort",
            "shard":shard,"requested":25,"unscorable":[],"cases":cases,
        })
    return cohorts


def test_100_database_paired_bag_wins_regressions_and_unchanged_cohort():
    a=aggregate(_aggregate())
    assert a["sample_size"]==100 and a["database_count"]==100
    assert a["a0_bag_correct"]==50 and a["a5_bag_correct"]==53
    assert a["wrong_to_correct"]==4 and a["correct_to_wrong"]==1
    assert a["net_gain"]==3
    assert a["planner_valid"]==100
    assert a["candidate_changed"]==0


def test_mixed_cohorts_and_incomplete_runs_are_refused():
    a=_aggregate()
    with pytest.raises(ValueError,match="four"):
        aggregate(a[:3])
    a=_aggregate();a[1]["cohort_sha256"]="changed"
    with pytest.raises(ValueError,match="cohort"):
        aggregate(a)
    a=_aggregate();a[1]["cases"][0]["database_id"]=a[0]["cases"][0]["database_id"]
    with pytest.raises(ValueError,match="duplicate database"):
        aggregate(a)
    a=_aggregate();a[0]["cases"][0]["strict"]["a5"]["bag_equal"]=False
    # Paired arithmetic is internally recomputed, so this remains a valid
    # regression; the audit reports it rather than silently discarding it.
    assert aggregate(a)["net_gain"]==2
    a=_aggregate();a[0]["cases"]=[]
    with pytest.raises(ValueError,match="insufficient"):
        aggregate(a)
