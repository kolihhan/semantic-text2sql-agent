"""Untouched-prompt A5 holdout fixtures, unique graph topology and pairing."""
import sqlite3
from copy import deepcopy
import pytest
from experiments.sql_roles_holdout_v2 import make_cases,build_holdout_schema
from experiments.counterfactual_sql_holdout_v2 import aggregate,prompt,parse_plan
from experiments.counterfactual_sql_plan_v1 import prompt as original_prompt
from experiments.audit_spider_exec_strict import score_readonly_pair


def test_second_topology_cases_are_executable_and_gold_is_not_a_prompt():
    assert prompt is original_prompt
    for domain in range(4):
        # Run fresh DBs only, no precomputed result-set leaks.
        from tempfile import TemporaryDirectory
        from pathlib import Path
        with TemporaryDirectory() as folder:
            db=Path(folder)/"holdout.sqlite"
            cases=make_cases(db,domain)
            assert len(cases)==12
            assert all(score_readonly_pair(db,x["reference_sql"],x["reference_sql"])["bag_equal"]
                       for x in cases)
            schema=build_holdout_schema(db,domain)
            assert "reference_sql" not in schema
            assert "CREATE TABLE" not in prompt(cases[0]["question"],schema)
            with sqlite3.connect(db) as c:
                assert c.execute(cases[0]["reference_sql"]).fetchone()==("Eli",)
                assert c.execute(cases[1]["reference_sql"]).fetchone()==("Bea",)
                assert c.execute(cases[2]["reference_sql"]).fetchone()==(3,)
                assert c.execute(cases[3]["reference_sql"]).fetchone()==(6,)
                assert c.execute(cases[4]["reference_sql"]).fetchall()==[("Fay",)]
                assert c.execute(cases[5]["reference_sql"]).fetchall()==[("Ada",),("Eli",),("Fay",)]
                assert c.execute(cases[6]["reference_sql"]).fetchone()==(6,)
                assert c.execute(cases[7]["reference_sql"]).fetchall()==[("Bea",)]
                assert c.execute(cases[8]["reference_sql"]).fetchone()==("Bea",)
                assert c.execute(cases[9]["reference_sql"]).fetchone()==("Bea",)
                assert c.execute(cases[11]["reference_sql"]).fetchall()==[("Bea",),("Cal",)]
            with pytest.raises(FileExistsError):
                make_cases(db,domain)


def test_duplicate_labels_and_joined_comments_fool_naive_counts():
    from tempfile import TemporaryDirectory
    from pathlib import Path
    with TemporaryDirectory() as folder:
        db=Path(folder)/"cases.sqlite"
        rows=make_cases(db,0)
        with sqlite3.connect(db) as c:
            distinct=c.execute(rows[6]["reference_sql"]).fetchone()[0]
            naive=c.execute(rows[6]["reference_sql"].replace("COUNT(DISTINCT b.id)","COUNT(*)")).fetchone()[0]
        assert distinct==6 and naive==7
        assert not score_readonly_pair(db,
            rows[6]["reference_sql"].replace("COUNT(DISTINCT b.id)","COUNT(*)"),
            rows[6]["reference_sql"])["bag_equal"]


def _fake_shards():
    result=[]
    for domain in range(4):
        cases=[]
        for row in range(12):
            i=12*domain+row
            base=i<20
            new=i<23 and i!=1
            cases.append({
                "case_id":f"holdout-{domain}:{row}",
                "sql":{"a0":"SELECT 1","a5":"SELECT 1"},
                "plan_valid":True,
                "strict":{"a0":{"bag_equal":base},"a5":{"bag_equal":new}},
            })
        result.append({"schema":"counterfactual-sql-plan-holdout-v2-shard",
                       "domain":domain,"model":"frozen-7b","count":12,
                       "uses_prior_48_gold":False,"gold_used_for_generation":False,
                       "cases":cases})
    return result


def test_holdout_requires_all_four_complete_shards_and_paired_math():
    shards=_fake_shards()
    result=aggregate(shards)
    assert result["sample_size"]==48
    assert result["a0_bag_correct"]==20
    assert result["a5_bag_correct"]==22
    assert result["wrong_to_correct"]==3
    assert result["correct_to_wrong"]==1
    assert result["net_gain"]==2
    with pytest.raises(ValueError,match="four"):
        aggregate(shards[:3])
    q=deepcopy(shards)
    q[2]["uses_prior_48_gold"]=True
    with pytest.raises(ValueError,match="integrity"):
        aggregate(q)
    q=deepcopy(shards)
    q[2]["model"]="different"
    with pytest.raises(ValueError,match="model"):
        aggregate(q)
    q=deepcopy(shards)
    q[2]["cases"][0]["case_id"]=q[0]["cases"][0]["case_id"]
    with pytest.raises(ValueError,match="unique"):
        aggregate(q)
