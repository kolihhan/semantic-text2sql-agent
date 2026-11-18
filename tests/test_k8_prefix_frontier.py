"""CPU-only source and gold-blind selection contract for K8 prefix frontier."""
import pytest

from evaluation import k8_prefix_frontier as p


def _outcome(signature: str):
    return {"valid": True, "signature": signature, "set_hash": signature,
            "bag_hash": signature, "reason": "ok"}


def test_prefix_vote_uses_only_original_candidate_outputs_no_reference():
    cache={"case_id":"test","db_id":"test_db","k":8}
    outcomes=[_outcome(x) for x in "ABBABBBB"]
    first=p.case_from_outcomes(cache,outcomes,_outcome("B"))
    other=p.case_from_outcomes(cache,outcomes,_outcome("A"))
    assert first["k_indices"]==other["k_indices"]
    assert first["k_indices"]=={"1":0,"2":0,"4":0,"8":1}
    assert first["k1_set_ex"] is False and first["k8_set_ex"] is True
    assert other["k1_set_ex"] is True and other["k8_set_ex"] is False
    assert first["goldblind_selection"] and other["goldblind_selection"]
    assert first["candidate_timeouts"]==0


def test_prefix_refuses_missing_candidates():
    with pytest.raises(ValueError):
        p.case_from_outcomes({"k":8},[_outcome("a")],_outcome("a"))


def test_fail_closed_without_all_eight_worker_evidence():
    with pytest.raises(ValueError):
        p.aggregate([],{"dev_case_ids":[]})


def test_complete_500_frozen_sample_with_no_model_calls(monkeypatch):
    monkeypatch.setattr(p,"paired_bootstrap",
                        lambda rows,baseline,treatment,cluster=False: {
                            "estimate":0,"ci95_lower":0,"ci95_upper":0,
                            "unit":"physical_db_cluster" if cluster else "paired_question"})
    m={"dev_case_ids":[str(i) for i in range(500)]}
    parts=[]
    for w in range(8):
        rows=[]
        for i in range(w,500,8):
            rows.append({
                "case_id":str(i),"db_id":"physical-"+str(i%11),
                "gold_scorable":True,"gold_timeout":False,
                "candidate_timeouts":0,"goldblind_selection":True,
                "pass8_oracle_set":True,
                "k1_set_ex":False,"k2_set_ex":False,
                "k4_set_ex":True,"k8_set_ex":True,
            })
        parts.append({"status":"SOURCE_LOCKED_K_PREFIX_WORKER_COMPLETE",
                      "protocol":p.PROTOCOL,"worker":w,"workers":8,
                      "source_sha256":p.SOURCE_SHA256,
                      "model_digest":p.MODEL_DIGEST,
                      "query_timeout_s":p.SQL_TIMEOUT,
                      "model_calls":0,"holdout300_used":False,"cases":rows})
    result=p.aggregate(parts,m)
    assert result["status"]=="COMPLETE_K_PREFIX_BOUNDED_DEV500"
    assert result["n"]==500 and result["n_physical_databases"]==11
    assert result["scores"]["1"]["correct"]==0
    assert result["scores"]["4"]["correct"]==500
    assert result["scores"]["8"]["original_Qwen_generations_per_task"]==8
    assert result["new_model_inference_calls"]==0
    assert result["holdout300_unused"]


def test_reference_timeout_blocks_official_prefix_report(monkeypatch):
    monkeypatch.setattr(p,"paired_bootstrap",lambda *args,**kwargs: {"estimate":0})
    manifest={"dev_case_ids":[str(i) for i in range(500)]}
    parts=[]
    for w in range(8):
        rows=[{"case_id":str(i),"db_id":"db","gold_scorable":True,
               "gold_timeout":False,"candidate_timeouts":0,
               "goldblind_selection":True,"pass8_oracle_set":False}
              for i in range(w,500,8)]
        if w==0:rows[0]["gold_scorable"]=False;rows[0]["gold_timeout"]=True
        parts.append({"status":"SOURCE_LOCKED_K_PREFIX_WORKER_COMPLETE",
                      "protocol":p.PROTOCOL,"worker":w,"workers":8,
                      "source_sha256":p.SOURCE_SHA256,
                      "model_digest":p.MODEL_DIGEST,
                      "query_timeout_s":p.SQL_TIMEOUT,"model_calls":0,
                      "holdout300_used":False,"cases":rows})
    out=p.aggregate(parts,manifest)
    assert out["status"]=="BLOCKED_REFERENCE_SQL_TIMEOUT"
    assert not out["scores"] and out["unscorable_count"]==1
