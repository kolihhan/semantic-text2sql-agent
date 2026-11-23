from __future__ import annotations

import pytest

from evaluation.paired_scorecard import paired_bird_scorecard, _exact_p


def record(i, db, old, new, **extra):
    return {
        "source_index":i,"db_id":db,"scorable":True,
        "control_set":old,"treatment_set":new,
        "control_bag":old,"treatment_bag":new,
        "model_calls_shared":1,"model_wall_s_shared":10.0+i,
        "additional_graph_wall_s":0.1,
        "graph_changed":old!=new,
        **extra
    }


def test_real_paired_rescues_regressions_and_db_slices():
    rows=[
        record(1,"a",False,True),
        record(2,"b",True,False),
        record(3,"b",True,True),
        record(4,"c",False,False),
    ]
    s=paired_bird_scorecard(rows)
    assert s["questions_total"]==4 and s["questions_scorable"]==4
    assert s["distinct_database_families"]==3
    assert s["strict_set_ex"]=={
        "control_correct":2,"treatment_correct":2,
        "net_correct":0,"rescue_n":1,"regression_n":1,
        "mcnemar_exact_p":1.0,
    }
    assert s["strict_bag_ex"]["net_correct"]==0
    assert s["per_database_net_set"]=={"a":1,"b":-1,"c":0}
    assert s["changed_sql_n"]==2
    assert s["cost"]["shared_Qwen_calls"]==4
    assert s["cost"]["shared_Qwen_wall_s"]==50.0
    assert s["cost"]["incremental_graph_wall_s"]==.4
    assert s["cost"]["shared_Qwen_latency_p50_s"]==12.5
    assert s["cost"]["shared_Qwen_latency_p95_s"]==14.0
    # Real token/dollar accounting cannot be inferred from number of calls.
    assert s["cost"]["prompt_tokens"] is None
    assert s["cost"]["generated_tokens"] is None
    assert s["cost"]["actual_USD_cost"] is None
    assert s["cost"]["end_to_end_latency_p95_s"] is None


def test_known_tokens_and_actual_money_only_when_explicit():
    r=record(8,"db",False,True,prompt_eval_count=165,eval_count=25,
             actual_usd_cost=0.012,end_to_end_wall_s=15)
    s=paired_bird_scorecard([r])
    assert s["cost"]["prompt_tokens"]==165
    assert s["cost"]["generated_tokens"]==25
    assert s["cost"]["actual_USD_cost"]==0.012
    assert s["cost"]["end_to_end_latency_p95_s"]==15


def test_frozen_unscorable_cases_not_counted_as_wrong_answers():
    bad=record(7,"dbx",False,False)
    bad.update(scorable=False,control_set=None,treatment_set=None,
               control_bag=None,treatment_bag=None,graph_changed=False)
    s=paired_bird_scorecard([record(1,"db",True,True),bad])
    assert s["questions_total"]==2 and s["questions_scorable"]==1
    assert s["questions_unscorable"]==1
    assert s["strict_set_ex"]["control_correct"]==1
    assert s["cost"]["shared_Qwen_calls"]==2


def test_exact_discordant_test_no_change_is_unknown_not_falsely_significant():
    assert _exact_p(0,0) is None
    assert _exact_p(3,0)==.25
    assert _exact_p(5,0)==.0625
    assert _exact_p(6,0)==.03125
    assert _exact_p(1,3)==.625


@pytest.mark.parametrize("rows",[
    [],
    [record(1,"a",False,False),record(1,"a",True,True)],
    [{**record(1,"a",False,False),"control_set":None}],
    [{**record(1,"a",False,False),"source_index":"1"}],
    [{**record(1,"a",False,False),"scorable":None}],
])
def test_source_identity_and_publisher_truth_shape_gate(rows):
    with pytest.raises(ValueError):
        paired_bird_scorecard(rows)


def test_zero_changes_is_no_evidence_of_precision():
    s=paired_bird_scorecard([record(1,"a",False,False,graph_changed=False)])
    assert s["changed_sql_n"]==0
    assert s["strict_set_ex"]["mcnemar_exact_p"] is None
