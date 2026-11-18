"""No gold/data leakage; exact 49/51 source DB family certification tests."""
from hashlib import sha256
from pathlib import Path
import json
import pytest

from scripts.freeze_birdtrain_residual_db_holdout import (
    EXPECTED_TOTAL_DBS, EXPECTED_USED_DBS, MODEL_DIGEST,
    PROTOCOL, original_question_heldout_indices, make_report,
    historical_source_ledger
)


def artificial():
    names=[f"db-{j:02d}" for j in range(EXPECTED_TOTAL_DBS)]
    used=set(names[:EXPECTED_USED_DBS])
    manifest={name:{"size":1024,"oid":"immutable"} for name in names}
    train=[]
    for db in names:
        for n in range(35):
            train.append({
                "db_id":db,"question":f"Question {n} for {db}",
                "evidence":"opaque evidence",
                "SQL":"SELECT * FROM GOLD_SECRET_SHOULD_NEVER_LEAK",
            })
    return manifest,used,train


def test_49_of_51_certifies_two_unexposed_physical_databases():
    manifest,used,train=artificial()
    result=make_report(train,manifest,used,{"prior_sources":{}})
    assert result["original_train_physical_database_count"]==51
    assert result["verified_historically_exposed_database_count"]==49
    assert result["verified_unexposed_database_count"]==2
    assert result["frozen_case_n"]==64
    assert {x["db_id"] for x in result["unexposed_database_summaries"]}=={"db-49","db-50"}
    assert {x["source_db_id"] for x in result["frozen_question_identities"]}=={"db-49","db-50"}
    assert result["fixed_base_model_digest"]==MODEL_DIGEST
    assert result["protocol"]==PROTOCOL
    assert result["new_Qwen_inference_calls"]==0
    assert not result["bird_HOLDOUT300_used"]
    assert not result["bird_DEV500_used"]


def test_no_gold_or_question_text_leaks_into_frozen_question_manifest():
    manifest,used,train=artificial()
    data=make_report(train,manifest,used,{})
    serialized=json.dumps(data)
    assert "GOLD_SECRET" not in serialized
    assert "Question 1 for" not in serialized
    assert "opaque evidence" not in serialized
    for x in data["frozen_question_identities"]:
        assert set(x)=={"source_index","source_db_id","question_sha256","evidence_sha256"}


def test_deterministic_source_question_order_without_gold():
    _,_,train=artificial()
    a,n=original_question_heldout_indices(train,"db-50")
    assert n==35 and len(a)==32
    mutated=[{**r,"SQL":"CHANGED GOLD ANSWER"} for r in train]
    b,_=original_question_heldout_indices(mutated,"db-50")
    assert a==b


def test_cannot_relax_49_used_or_51_original_source_databases():
    manifest,used,train=artificial()
    with pytest.raises(ValueError,match="exposure count"):
        make_report(train,manifest,set(list(used)[:48]),{})
    with pytest.raises(ValueError,match="EXACTLY 51"):
        make_report(train,{k:v for k,v in manifest.items() if k!="db-50"},used,{})


def test_unverified_previous_database_fails_closed():
    manifest,used,train=artificial()
    with pytest.raises(ValueError,match="absent from pinned original"):
        make_report(train,manifest,used|{"FORGED_DB"},{})


def test_unusable_too_large_db_is_flagged_not_replaced():
    manifest,used,train=artificial()
    manifest["db-50"]["size"]=130*1024*1024
    result=make_report(train,manifest,used,{})
    assert result["verified_unexposed_database_count"]==2
    assert [x for x in result["unexposed_database_summaries"] if x["db_id"]=="db-50"][0][
        "within_existing_96MiB_source_inference_limit"] is False


def test_missing_prior_exposure_provenance_is_not_treated_as_no_overlap():
    with pytest.raises(ValueError,match="missing exact source history"):
        historical_source_ledger({})
