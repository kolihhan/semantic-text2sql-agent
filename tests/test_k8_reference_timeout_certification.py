"""Conservative reference-timeout contract: ALL candidates invalid -> known wrong."""
from evaluation import k8_prefix_frontier as p


def parts_with_one_gold_timeout():
    parts=[]
    for w in range(8):
        rows=[]
        for i in range(w,500,8):
            rows.append({
                "case_id":str(i),"db_id":"db-"+str(i%11),
                "goldblind_selection":True,
                "gold_scorable":True,"gold_timeout":False,
                "candidate_executed":8,"candidate_timeouts":0,
                "k1_set_ex":False,"k2_set_ex":False,
                "k4_set_ex":False,"k8_set_ex":False,
                "pass8_oracle_set":False,
            })
        parts.append({
            "status":"SOURCE_LOCKED_K_PREFIX_WORKER_COMPLETE",
            "protocol":p.PROTOCOL,"worker":w,"workers":8,
            "source_sha256":p.SOURCE_SHA256,"model_digest":p.MODEL_DIGEST,
            "query_timeout_s":p.SQL_TIMEOUT,"model_calls":0,
            "holdout300_used":False,"cases":rows
        })
    target=next(x for x in parts[0]["cases"] if x["case_id"]=="0")
    target.update(gold_scorable=False,gold_timeout=True,
                  candidate_executed=0,candidate_timeouts=4)
    return parts,{"dev_case_ids":[str(i) for i in range(500)]}


def test_reference_timeout_all_invalid_is_provably_incorrect(monkeypatch):
    monkeypatch.setattr(p,"paired_bootstrap",lambda *args,**kwargs: {"ci95_lower":0,"ci95_upper":0,"estimate":0})
    parts,m=parts_with_one_gold_timeout()
    result=p.aggregate(parts,m)
    assert result["status"]=="COMPLETE_K_PREFIX_BOUNDED_DEV500_WITH_CERTIFIED_CANDIDATE_FAILURE"
    assert result["candidate_failure_certified_reference_timeouts"]==1
    assert result["candidate_failure_certified_case_ids"]==["0"]
    assert result["unscorable_count"]==0 and result["reference_timeout_count"]==1
    assert result["scores"]["8"]["correct"]==0
    assert result["new_model_inference_calls"]==0
    assert result["holdout300_unused"]


def test_reference_timeout_with_any_executable_candidate_must_block(monkeypatch):
    monkeypatch.setattr(p,"paired_bootstrap",lambda *args,**kwargs: {"estimate":0})
    parts,m=parts_with_one_gold_timeout()
    parts[0]["cases"][0]["candidate_executed"]=1
    result=p.aggregate(parts,m)
    assert result["status"]=="BLOCKED_REFERENCE_SQL_TIMEOUT"
    assert result["unscorable_count"]==1 and result["scores"]=={}


def test_reference_timeout_with_suspected_correct_selection_must_block(monkeypatch):
    parts,m=parts_with_one_gold_timeout()
    parts[0]["cases"][0]["k8_set_ex"]=True
    result=p.aggregate(parts,m)
    assert result["status"]=="BLOCKED_REFERENCE_SQL_TIMEOUT"
    assert not result["scores"]
