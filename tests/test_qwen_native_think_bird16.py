from evaluation.qwen_native_think_bird16 import aggregate, _group, DIGEST


def test_question_only_semantic_strata():
    assert _group("How many orders were placed?") == "analytic"
    assert _group("Which drivers ranked first?") == "analytic"
    assert _group("List customers in Dublin.") == "other"


def _part(i):
    rows=[]
    for j in range(2):
        k=2*i+j
        no=k not in (0,1)
        yes=k not in (1,2)
        rows.append({
            "case_id":str(k),
            "group":"analytic" if j==0 else "other",
            "reference_scorable":k!=3,
            "arms":{
                "no_think":{"ex":no if k!=3 else None,"bag":no if k!=3 else None,
                            "sql":"SELECT 1","latency_s":1.5},
                "native_think":{"ex":yes if k!=3 else None,"bag":yes if k!=3 else None,
                                "sql":"SELECT 2","latency_s":5.5,"thinking_chars":50},
            },
        })
    return {
        "protocol":"qwen-native-reasoning-bird16-v1",
        "shard":i,"model_digest":DIGEST,"cohort_sha256":"frozen",
        "cases":rows,
    }


def test_aggregate_reports_oracle_without_gold_on_invalid_reference():
    a=aggregate([_part(i) for i in range(8)])
    assert a["overall"]["n"]==16
    assert a["overall"]["valid_gold"]==15
    assert a["overall"]["rescues"]==1
    assert a["overall"]["regressions"]==1
    assert a["overall"]["oracle_ex"]==14
    assert a["groups"]["analytic"]["n"]==8
    assert a["groups"]["other"]["n"]==8
    assert a["no_chain_of_thought_saved"] is True


def test_fail_closed_on_missing_shards_and_model_drift():
    import pytest
    with pytest.raises(ValueError,match="eight unique"):
        aggregate([_part(i) for i in range(7)])
    shards=[_part(i) for i in range(8)]
    shards[2]["model_digest"]="other"
    with pytest.raises(ValueError,match="model"):
        aggregate(shards)
