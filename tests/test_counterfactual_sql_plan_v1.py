"""CPU-only SQL A5 counterfactual domain and gold-leakage contract tests."""
from copy import deepcopy
import sqlite3

import pytest

from experiments.sql_roles_synthetic import create_domain,mschema
from experiments.counterfactual_sql_plan_v1 import parse_plan,prompt,aggregate
from experiments.audit_spider_exec_strict import score_readonly_pair


def test_48_new_synthetic_cases_are_executable_and_role_sensitive(tmp_path):
    for domain in range(4):
        db=tmp_path/f"synthetic_{domain}.sqlite"
        cases=create_domain(db,domain)
        assert len(cases)==12
        assert len({x["case_id"] for x in cases})==12
        assert len({x["question"] for x in cases})==12
        assert all(score_readonly_pair(db,x["reference_sql"],x["reference_sql"])["bag_equal"]
                   for x in cases)
        # The first pair must be genuinely asymmetric: initiator vs recipient.
        with sqlite3.connect(db) as con:
            initiated=con.execute(cases[0]["reference_sql"]).fetchall()
            received=con.execute(cases[1]["reference_sql"]).fetchall()
        assert initiated==[("Eli",)]
        assert received==[("Bea",)]
        assert not any("reference_sql" in str(mschema(db,domain))
                       for _ in (0,))
        with pytest.raises(FileExistsError):
            create_domain(db,domain)


def test_count_distinct_and_not_exists_have_real_nonempty_counterexamples(tmp_path):
    db=tmp_path/"novel.sqlite"
    rows=create_domain(db,0)
    with sqlite3.connect(db) as con:
        assert con.execute(rows[10]["reference_sql"]).fetchone()[0]==4
        assert con.execute(rows[5]["reference_sql"]).fetchall()==[("Fay",)]
        assert con.execute(rows[6]["reference_sql"]).fetchall()==[("Ada",),("Fay",)]
        # Distinct prevents duplicated urgent label for item 3.
        sql=rows[10]["reference_sql"].replace("COUNT(DISTINCT a.id)","COUNT(*)")
        assert con.execute(sql).fetchone()[0]==5
    assert not score_readonly_pair(db,sql,rows[10]["reference_sql"])["bag_equal"]


def test_planner_only_accepts_small_complete_json_spec():
    x={"answer_entity":"members","relation_role":"target",
       "aggregation_grain":"distinct senders","filters":"UNKNOWN"}
    assert parse_plan(__import__("json").dumps(x))==x
    assert parse_plan("SELECT COUNT(*) FROM members") is None
    assert parse_plan("{}") is None
    assert parse_plan("[]") is None
    assert parse_plan(__import__("json").dumps({**x,"SQL":"SELECT *"})) is None
    assert parse_plan("a"*4001) is None
    assert "reference SQL" not in prompt("Who received?", "members(id,name)")
    assert "JSON" in prompt("Who received?", "members(id,name)")


def _shards():
    results=[]
    for shard in range(4):
        cases=[]
        for idx in range(12):
            n=shard*12+idx
            baseline=n<23
            treat=n<25 and n!=1
            cases.append({
                "case_id":f"{shard}:{idx}",
                "sql":{"a0":"SELECT 1","a5":"SELECT 1"},
                "strict":{"a0":{"bag_equal":baseline},
                          "a5":{"bag_equal":treat}},
                "plan_valid":True,"elapsed_s":5.,
            })
        results.append({
            "schema":"counterfactual-sql-plan-shard-v1",
            "model":"XiYanSQL-7B",
            "domain":shard,"count":12,"gold_used_for_generation":False,
            "cases":cases,
        })
    return results


def test_aggregation_uses_paired_wins_and_losses_without_hiding_regressions():
    shards=_shards()
    a=aggregate(shards)
    assert a["a0_bag_correct"]==23
    assert a["a5_bag_correct"]==24
    assert a["wrong_to_correct"]==2 and a["correct_to_wrong"]==1
    assert a["net_gain"]==1
    assert a["sample_size"]==48
    assert a["plan_valid_count"]==48
    with pytest.raises(ValueError,match="four"):
        aggregate(shards[:3])
    broken=deepcopy(shards)
    broken[0]["gold_used_for_generation"]=True
    with pytest.raises(ValueError,match="provenance"):
        aggregate(broken)
    broken=deepcopy(shards)
    broken[0]["cases"][0]["case_id"]=broken[1]["cases"][0]["case_id"]
    with pytest.raises(ValueError,match="unique"):
        aggregate(broken)
