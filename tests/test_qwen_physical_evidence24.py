"""Paired Qwen-only evidence candidate oracle is NOT a deployment selector."""
from evaluation.qwen_physical_evidence24 import aggregate, COHORT_COUNTS, QWEN_DIGEST


def _part(i):
    group=["value"]*4+["relation"]*12+["control"]*8
    rows=[]
    for j in range(4):
        k=4*i+j
        old=k not in (0,1,2)
        new=k not in (1,2,3)
        rows.append({
            "case_id":str(k),"group":group[k],"reference_scorable":True,
            "value_matches":int(group[k]=="value"),
            "arms":{
                "baseline":{"ex":old,"bag":old,"sql":"SELECT 1"},
                "physical_facts":{"ex":new,"bag":new,
                                  "sql":"SELECT 1" if group[k]=="control" else "SELECT 2"},
            },
        })
    return {"protocol":"qwen-physical-db-facts24-v1","shard":i,
            "model_digest":QWEN_DIGEST,"cohort_sha256":"pinned",
            "cases":rows}


def test_aggregate_counts_hindsight_oracle_and_controls():
    result=aggregate([_part(i) for i in range(6)])
    assert result["overall"]["n"]==24
    assert result["overall"]["rescues"]==1
    assert result["overall"]["regressions"]==1
    assert result["overall"]["oracle_ex"]==22
    assert result["groups"]["control"]["changed_sql"]==0
    assert sum(COHORT_COUNTS.values())==24


def test_aggregate_rejects_missing_or_model_drift():
    import pytest
    with pytest.raises(ValueError,match="six unique"):
        aggregate([_part(i) for i in range(5)])
    bad=[_part(i) for i in range(6)]
    bad[0]["model_digest"]="different"
    with pytest.raises(ValueError,match="model"):
        aggregate(bad)


def test_aggregate_accounts_unscorable_gold_without_counting_model_error():
    parts=[_part(i) for i in range(6)]
    parts[0]["cases"][0]["reference_scorable"]=False
    parts[0]["cases"][0]["arms"]["baseline"]["ex"]=None
    parts[0]["cases"][0]["arms"]["physical_facts"]["ex"]=None
    parts[0]["cases"][0]["arms"]["baseline"]["bag"]=None
    parts[0]["cases"][0]["arms"]["physical_facts"]["bag"]=None
    result=aggregate(parts)
    assert result["overall"]["invalid_gold"]==1
    assert result["overall"]["scorable"]==23
