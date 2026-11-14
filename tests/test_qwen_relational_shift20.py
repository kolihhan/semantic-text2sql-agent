"""Gold-blind, DB-disjoint selection and honest paired metrics."""
from copy import deepcopy

import pytest

from evaluation.qwen_relational_shift20 import select_manifest, aggregate, ARMS, N_SHARDS


def _sources():
    train=[]
    mirror={}
    for i in range(51):
        database=f"domain_{i:02d}"
        mirror[database]={"size":2048}
        for j in range(3):
            train.append({
                "db_id":database, "question":f"Question {j} in {database}",
                "evidence":"", "SQL":f"secret_gold_{i}_{j}",
            })
    return train,[{"db_id":"dev_only","question":"question from dev"}],mirror


def test_10_unseen_databases_not_older_40_or_dev_and_two_per_db():
    train,dev,mirror=_sources()
    manifest=select_manifest(train,dev,mirror)
    chosen=manifest["selected"]
    assert len(chosen)==20
    assert len({r["case_id"] for r in chosen})==20
    assert len({r["database_id"] for r in chosen})==10
    assert all(sum(r["database_id"]==db for r in chosen)==2
               for db in {r["database_id"] for r in chosen})
    assert manifest["previous_cohort_db_count"]==40
    assert all(set(row)=={"case_id","database_id","question","evidence"} for row in chosen)


def test_select_never_looks_at_gold_or_relational_constraints():
    train,dev,mirror=_sources()
    selected=select_manifest(train,dev,mirror)
    for row in train:
        row.pop("SQL")
    assert selected==select_manifest(train,dev,mirror)
    # Source schema/foreign keys cannot influence selection, only DB identity.
    mirror={k:{**v,"ignored_join_key_count":999} for k,v in mirror.items()}
    assert selected==select_manifest(train,dev,mirror)


def test_missing_new_db_count_fails_closed():
    train,dev,mirror=_sources()
    del mirror["domain_50"]
    del mirror["domain_49"]
    with pytest.raises(ValueError,match="50"):
        select_manifest(train,dev,mirror)


def _single_case(i, db, baseline=False, challenger=False, changed=True):
    return {
        "case_id":i,"db_id":db,
        "reference_scorable":True,
        "schema_changed":changed,
        "declared_key_count":2 if changed else 0,
        "fk_count":1 if changed else 0,
        "arms":{
            ARMS[0]:{"strict_ex":baseline,"bag":baseline,
                     "execution_success":True,"model_repairs":0,
                     "sql":"SELECT 1","latency_s":5},
            ARMS[1]:{"strict_ex":challenger,"bag":challenger,
                     "execution_success":True,"model_repairs":0,
                     "sql":"SELECT 2" if changed else "SELECT 1",
                     "latency_s":6},
        },
    }


def test_aggregate_reports_all_cases_and_regressions_even_when_no_metadata():
    items=[_single_case(i,f"domain_{i//2}",i%3==0,i%3==1,
                        changed=i//2<8) for i in range(20)]
    shards=[{
        "protocol":"fixed","shard":j,"model_digest":"pinned",
        "cohort_sha256":"frozen","no_training":True,
        "constraints_only_from_ddl":True,"cases":items[j*4:(j+1)*4],
    } for j in range(N_SHARDS)]
    summary=aggregate(shards)
    assert summary["total_cases"]==20 and summary["different_physical_databases"]==10
    assert summary["metadata_changed"]==16
    assert summary["rescues"]==sum(
        not r["arms"][ARMS[0]]["strict_ex"] and r["arms"][ARMS[1]]["strict_ex"] for r in items)
    assert summary["regressions"]==sum(
        r["arms"][ARMS[0]]["strict_ex"] and not r["arms"][ARMS[1]]["strict_ex"] for r in items)
    assert summary["unchanged_schema_sql_divergences"]==0
    assert summary["promotion_allowed"] is False
    shards[0]["model_digest"]="different"
    with pytest.raises(ValueError,match="drift"):
        aggregate(shards)
