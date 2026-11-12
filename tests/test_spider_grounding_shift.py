"""Reproducible Spider source-shift sampling/paired-scoring contract tests."""
from copy import deepcopy

import pytest

from experiments.spider_grounding_shift import frozen_selection,aggregate


def _source():
    return [
        {"db_id":f"domain_{d}","question":f"Question {i} about domain {d}?",
         "query":"PRIVATE-GOLD-THAT-MUST-NOT-ALTER-COHORT"}
        for d in range(20) for i in range(6)
    ]


def test_balanced_100_spider_queries_from_20_distinct_databases():
    source=_source()
    cohort=frozen_selection(source)
    assert len(cohort)==100
    assert len({row["case_id"] for row in cohort})==100
    assert len({row["database_id"] for row in cohort})==20
    assert all(sum(row["database_id"]==f"domain_{d}" for row in cohort)==5
               for d in range(20))
    assert all("query" not in row and "sql" not in row for row in cohort)
    rerun=deepcopy(source)
    for row in rerun:row["query"]="CHANGED GOLD LABEL MUST NOT AFFECT SELECTION"
    assert frozen_selection(rerun)==cohort
    nine_dbs=[
        {"db_id":f"db{d}","question":f"Question {i}","query":"private gold"}
        for d in range(9) for i in range(12)
    ]
    with pytest.raises(ValueError,match=">=10"):
        frozen_selection(nine_dbs)


def _shards():
    source=_source()
    cohort=frozen_selection(source)
    parts=[]
    for shard in range(4):
        cases=[]
        for index,row in enumerate(cohort[shard::4]):
            i=shard+index*4
            base=i<50
            enriched=i<52
            cases.append({
                **row,"sql":{"a0":"SELECT 1","a4":"SELECT 1"},
                "scores":{
                    "a0":{"frozen_execution_match":base},
                    "a4":{"frozen_execution_match":enriched},
                },
                "a4_grounding":{"literal_hits":1,"verified_fk_declarations":2},
                "latency_s":5.0,
            })
        parts.append({
            "schema":"spider-source-shift-a0-a4-shard-v1",
            "shard":shard,"requested":25,"cohort_sha256":"frozen",
            "model":"xiyan","unscorable":[],"cases":cases,
        })
    return parts


def test_paired_spider_aggregator_reports_wins_and_loss_counts():
    shards=_shards()
    report=aggregate(shards)
    assert report["requested_cases"]==100
    assert report["sample_size"]==100
    assert report["database_count"]==20
    assert report["a0_correct"]==50 and report["a4_correct"]==52
    assert report["wrong_to_correct"]==2
    assert report["correct_to_wrong"]==0
    assert report["net_gain"]==2
    assert report["total_literal_hits"]==100
    assert report["total_fk_declarations"]==200
    assert report["average_pair_latency_s"]==5.0


def test_aggregate_rejects_lost_shard_or_duplicates_or_wrong_cohort():
    a=_shards()
    with pytest.raises(ValueError):
        aggregate(a[:3])
    a=_shards()
    a[2]["cohort_sha256"]="different"
    with pytest.raises(ValueError,match="cohort"):
        aggregate(a)
    a=_shards()
    a[0]["cases"][0]["case_id"]=a[1]["cases"][0]["case_id"]
    with pytest.raises(ValueError,match="duplicate"):
        aggregate(a)
    a=_shards()
    a[3]["cases"].pop()
    with pytest.raises(ValueError,match="incomplete"):
        aggregate(a)


def test_unscorable_public_gold_is_reported_not_silently_counted_wrong():
    a=_shards()
    removed=a[0]["cases"].pop()
    a[0]["unscorable"]=[{"case_id":removed["case_id"],"reason":"RuntimeError"}]
    report=aggregate(a)
    assert report["sample_size"]==99
    assert report["requested_cases"]==100
    assert report["unscorable"]==a[0]["unscorable"]
