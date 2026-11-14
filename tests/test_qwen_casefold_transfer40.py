"""Known Qwen9B cohorts are immutable, question-disjoint transfer subjects."""
from evaluation.qwen_casefold_transfer40 import (
    assemble, summarize,
    PHYSICAL_PROTOCOL, SCHEMA_PROTOCOL,
    MODEL_DIGEST,
)


def _sample(schema, index):
    if schema:
        return {
            "case_id":str(index),"database_id":"db1",
            "reference_scorable":True,
            "arms":{"legacy":{"sql":"SELECT 1", "ex":True,"bag":True}},
        }
    return {
        "case_id":str(index),"db_id":"db2",
        "reference_scorable":True,
        "arms":{"baseline":{"sql":"SELECT 2", "ex":False,"bag":False}},
    }


def test_source_and_model_identity_check_fail_closed():
    import pytest
    schema=[{"protocol":SCHEMA_PROTOCOL,"model_digest":MODEL_DIGEST,
             "cohort_sha256":"schema","shard":i,
             "cases":[_sample(True,4*i+j) for j in range(4)]} for i in range(4)]
    physical=[{"protocol":PHYSICAL_PROTOCOL,"model_digest":MODEL_DIGEST,
               "cohort_sha256":"physical","shard":i,
               "cases":[_sample(False,16+4*i+j) for j in range(4)]} for i in range(6)]
    rows=assemble(schema,physical,set())
    assert len(rows)==40
    assert len({r["case_id"] for r in rows})==40
    assert rows[0]["group"]=="schema16"
    assert rows[-1]["group"]=="physical24"
    physical[0]["model_digest"]="wrong"
    with pytest.raises(ValueError,match="model"):
        assemble(schema,physical,set())


def test_exclusion_known_old100_must_hold():
    import pytest
    schema=[{"protocol":SCHEMA_PROTOCOL,"model_digest":MODEL_DIGEST,
             "cohort_sha256":"schema","shard":i,
             "cases":[_sample(True,4*i+j) for j in range(4)]} for i in range(4)]
    physical=[{"protocol":PHYSICAL_PROTOCOL,"model_digest":MODEL_DIGEST,
               "cohort_sha256":"physical","shard":i,
               "cases":[_sample(False,16+4*i+j) for j in range(4)]} for i in range(6)]
    with pytest.raises(ValueError,match="frozen100"):
        assemble(schema,physical,{"0"})


def test_aggregate_includes_abstentions_not_only_revisions():
    rows=[
        {"group":"schema16","eligible":True,"revised":True,
         "original_ex":False,"new_ex":True,"original_bag":False,"new_bag":True},
        {"group":"schema16","eligible":False,"revised":False,
         "original_ex":True,"new_ex":True,"original_bag":True,"new_bag":True},
        {"group":"physical24","eligible":True,"revised":True,
         "original_ex":True,"new_ex":False,"original_bag":True,"new_bag":False}
    ]
    x=summarize(rows)
    assert x["n"]==3 and x["rescues"]==1 and x["regressions"]==1
    assert x["original_ex"]==2 and x["new_ex"]==2
    assert x["changed"]==2 and x["groups"]["schema16"]["n"]==2
