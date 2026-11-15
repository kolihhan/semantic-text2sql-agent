import pytest
from evaluation.qwen_tools_bird12 import (
    aggregate,PROTOCOL,DIGEST,BIRD_SOURCE_SHA,N,SHARDS
)


def samples(*,rescue=2,regress=0):
    rows=[]
    for i in range(N):
        old=i<5
        new=old
        if 5<=i<5+rescue:
            new=True
        if i<regress:
            new=False
        def arm(ex):
            return {"set_ex":ex,"bag_ex":ex,"execution_success":True,
                    "model_calls":1,"elapsed_s":3.5}
        treated=arm(new)
        treated.update({
            "actual_tool_calls":2,"actual_schema_inspects":1,
            "actual_row_samples":0,"actual_sql_trials":1,"actual_tool_errors":0,
            "tool_agent_stop_reason":"model_submitted"
        })
        rows.append({"case_id":str(i),"database_id":"db_"+str(i//2),
                     "scorable":True,"arms":{"original":arm(old),
                                            "observed_tools":treated}})
    return [{
        "protocol":PROTOCOL,"digest":DIGEST,"source_sha":BIRD_SOURCE_SHA,
        "cohort_sha":"frozen","goldblind_generation":True,"shard":shard,
        "cases":rows[4*shard:4*(shard+1)]
    } for shard in range(SHARDS)]


def test_real_tools_positive_gate_needs_two_dbs_and_no_regression():
    out=aggregate(samples(rescue=3))
    assert out["arms"]["original"]["strict_set_ex"]==5
    assert out["arms"]["observed_tools"]["strict_set_ex"]==8
    assert out["tool_calls"]["actual_tool_calls"]==24
    assert out["positive_source_screen_only"] is True
    assert out["promotion_allowed"] is False
    x=aggregate(samples(rescue=3,regress=1))
    assert x["positive_source_screen_only"] is False
    assert x["regressions"]==["0"]


def test_missing_physical_shards_fail_closed():
    with pytest.raises(ValueError,match="incomplete"):
        aggregate(samples()[:-1])
    x=samples()
    x[0]["goldblind_generation"]=False
    with pytest.raises(ValueError,match="incomplete"):
        aggregate(x)
