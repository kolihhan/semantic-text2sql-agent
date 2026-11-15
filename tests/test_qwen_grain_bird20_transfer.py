"""Strict gold-blind cross-source Qwen9B prompt-transfer test."""
import pytest
from hashlib import sha256

from evaluation.qwen_grain_direct_spider import _GRAIN_HINT,GrainDirectProvider
from evaluation.qwen_grain_bird20_transfer import aggregate,freeze,N,DBS,PROTOCOL,BIRD_SOURCE_SHA,DIGEST,SHARDS


class MockQwen:
    def __init__(self):
        self.call=None
    def complete_text(self,*,system,user):
        self.call=(system,user)
        return "SELECT 1"


def test_source_prompt_byte_identical_not_model_cheat():
    q=MockQwen()
    t=GrainDirectProvider(q)
    assert t.complete_text(system="Unmodified original system",user="Question")=="SELECT 1"
    assert q.call==("Unmodified original system\n"+_GRAIN_HINT,"Question")
    assert sha256(_GRAIN_HINT.encode()).hexdigest()==(
        "b8eab1e89db42ebdaf4c0b5cb32c15dd586e49d7c3f8c78dd07c64c1d829d4f"
    )


def dummy_shards(*, gained=3, lost=0, malformed=False):
    rows=[]
    for i in range(N):
        x={"case_id":str(i),
           "database_id":f"db_{i//2}",
           "reference_scorable":True,
           "arms":{
               "original":{"set_ex":i<8,"bag_ex":i<8,"status":"ok",
                           "truncated":False,"model_calls":1,"elapsed_s":1.0},
               "grain_direct":{"set_ex":i<8,"bag_ex":i<8,"status":"ok",
                               "truncated":False,"model_calls":1,"elapsed_s":1.1},
           }}
        if i in range(8,8+gained):
            x["arms"]["grain_direct"]["set_ex"]=True
            x["arms"]["grain_direct"]["bag_ex"]=True
        if i in range(lost):
            x["arms"]["grain_direct"]["set_ex"]=False
            x["arms"]["grain_direct"]["bag_ex"]=False
        rows.append(x)
    shards=[{"protocol":PROTOCOL,"source_sha":BIRD_SOURCE_SHA,
             "cohort_sha":"frozen","hint_sha":"hint",
             "digest":DIGEST,"shard":j,"source_gold_blind_generation":True,
             "cases":rows[5*j:5*(j+1)]} for j in range(SHARDS)]
    return shards


def test_agreement_requires_independent_physical_databases_and_no_regression():
    x=aggregate(dummy_shards(gained=3))
    assert x["n"]==N and x["database_count"]==DBS
    assert x["arms"]["original"]["set_ex"]==8
    assert x["arms"]["grain_direct"]["set_ex"]==11
    assert x["positive_screen_only"] is False  # only 2 DB rescued (8,9 same DB)
    assert x["promotion_allowed"] is False
    y=aggregate(dummy_shards(gained=5,lost=1))
    assert not y["positive_screen_only"] # preserve no-regression gate


def test_no_fake_progress_with_unscorable_or_missing_shards():
    with pytest.raises(ValueError,match="invalid incomplete"):
        aggregate(dummy_shards()[:3])
    y=dummy_shards()
    y[0]["source_gold_blind_generation"]=False
    with pytest.raises(ValueError,match="invalid incomplete"):
        aggregate(y)


def test_question_only_filter_excludes_non_aggregate():
    from evaluation.qwen_grain_direct_spider import is_group_question
    assert is_group_question("What is the average sales per seller?")
    assert not is_group_question("List the cities and country names.")
