"""Zero-model unit tests: oracle-labelled diagnostic cannot become a selector."""
import pytest
from evaluation import k8_oracle_gap_census as c


def o(v):
    return {"valid":True,"reason":"ok","signature":v,"set_hash":v,"bag_hash":v}


def test_correct_minority_is_observed_only_after_original_plurality_choice():
    outcomes=[o(x) for x in ["bad","bad","bad","bad","good","good","other","other"]]
    report=c.describe_support(outcomes,o("good"),selected_index=0)
    assert report["selected_index_matches_physical_record"]
    assert not report["selected_is_correct"]
    assert report["correct_candidate_count"]==2
    assert report["correct_result_support"]==2
    assert report["selected_result_support"]==4
    assert report["gold_scorable"] and report["candidate_timeouts"]==0


def test_tied_correct_result_loses_stable_original_first_index():
    outcomes=[o(x) for x in ["bad","good","bad","good","other","other","third","third"]]
    a=c.describe_support(outcomes,o("good"),selected_index=0)
    assert a["correct_result_support"]==a["selected_result_support"]==2
    assert not a["selected_is_correct"]
    assert a["selected_index_matches_physical_record"]
    # Gold changes diagnostic counts but cannot modify frozen candidate index.
    b=c.describe_support(outcomes,o("bad"),selected_index=0)
    assert b["selected_index_matches_physical_record"]


def test_replay_drift_detects_nonreproducible_voting():
    outcomes=[o(x) for x in ["bad"]*4+["good"]*2+["other"]*2]
    report=c.describe_support(outcomes,o("good"),selected_index=5)
    assert report["selected_index_matches_physical_record"] is False


def test_wrong_source_candidate_count_rejected():
    with pytest.raises(ValueError):
        c.describe_support([o("bad")],o("good"),0)


def parts(*,drift=False):
    out=[]
    for i in range(4):
        cases=[]
        for j in range(13):
            case={"case_id":str(i*13+j),"db_id":f"physical_{j%11}",
                  "correct_candidate_count":2,
                  "correct_result_support":2,"selected_result_support":4,
                  "valid_candidate_count":8,
                  "selected_index_matches_physical_record":True,
                  "selected_is_correct":False,"gold_scorable":True,
                  "candidate_timeouts":0,"gold_timeout":False}
            cases.append(case)
        out.append({
            "status":"SOURCE_ONLY_ORACLE_GAP_CENSUS_PART_COMPLETE",
            "protocol":c.CENSUS_PROTOCOL,
            "physical_protocol":c.PHYSICAL_PROTOCOL,
            "source_sha256":c.SOURCE_SHA256,"model_digest":c.MODEL_DIGEST,
            "parts":4,"part":i,
            "query_timeout_s":c.SQL_TIMEOUT,"model_calls":0,"holdout300_used":False,
            "cases":cases,
        })
    if drift:
        out[0]["cases"][0]["selected_index_matches_physical_record"]=False
    return out


def test_fail_closed_incomplete_52_cohort():
    with pytest.raises(ValueError):
        c.aggregate(parts()[:-1])


def test_stable_52_support_histogram_only_no_new_accuracy():
    result=c.aggregate(parts())
    assert result["status"]=="COMPLETE_REPRODUCED_52_CASE_ORACLE_GAP_CENSUS"
    assert result["replayed_cases"]==52 and result["stable_oracle_gap_cases"]==52
    assert result["correct_candidate_count_histogram"]=={"2":52}
    assert result["correct_result_vote_support_histogram"]=={"2":52}
    assert result["new_original_Qwen_inference_calls"]==0 and result["holdout300_unused"]


def test_any_replay_drift_blocks_claim():
    result=c.aggregate(parts(drift=True))
    assert result["status"]=="DIAGNOSTIC_REPLAY_INSTABILITY_DO_NOT_INFER"
    assert result["selected_candidate_index_drift"]==1
