"""Frozen, gold-blind structural selector tests. No inference or database writes."""
import copy

import pytest

from experiments.bird_structural_contract_gate import choose, evaluate, requirements, structural_support


def test_distinct_count_requires_evidence_in_sql():
    assert "explicit_count_distinct" in requirements("How many different cities have students?")
    c=choose("How many different cities have students?",
        "SELECT COUNT(city) FROM students",
        "SELECT COUNT(DISTINCT city) FROM students")
    assert c["chosen"]=="a1"
    assert c["a0_support"]["explicit_count"] is True
    assert c["a0_support"]["explicit_count_distinct"] is False
    assert c["a1_support"]["explicit_count_distinct"] is True


def test_no_explicit_contract_keeps_original_even_if_alt_differs():
    c=choose("Show the city with greater revenue.",
        "SELECT city FROM sales", "SELECT city FROM sales ORDER BY revenue DESC LIMIT 1")
    assert c["requirements"]==[]
    assert c["chosen"]=="a0"


def test_average_and_top_k_switch_only_on_missing_requested_syntax():
    c=choose("Find the average salary", "SELECT SUM(salary) FROM jobs",
             "SELECT AVG(salary) FROM jobs")
    assert c["chosen"]=="a1"
    c=choose("Show top 3 employees by income", "SELECT name FROM people",
             "SELECT name FROM people ORDER BY income DESC LIMIT 3")
    assert c["chosen"]=="a1"


def test_invalid_challenger_or_already_compliant_baseline_do_not_switch():
    c=choose("How many people live here?",
             "SELECT COUNT(*) FROM people",
             "SELECT COUNT(DISTINCT address) FROM people")
    assert c["chosen"]=="a0"
    c=choose("How many people live here?",
             "SELECT name FROM people", "garbage ???")
    assert c["chosen"]=="a0"
    c=choose("How many people live here?", "SELECT name FROM people", "")
    assert c["chosen"]=="a0"


def _fake_paired():
    rows=[]
    for i in range(100):
        ex0=i < 64
        ex1=(9 <= i < 64) or (64 <= i < 68)
        rows.append({
            "case_id":str(i), "database_id":f"db{i//20}",
            "sql":{"a0":"SELECT COUNT(city) FROM places",
                   "a1":"SELECT COUNT(DISTINCT city) FROM places"},
            "ex":{"a0":ex0,"a1":ex1},
        })
    return {"schema":"bird-fk-bridge-aggregate-v1","count":100,
            "baseline_correct":64,"question_grounded_correct":59,
            "wrong_to_correct":4,"correct_to_wrong":9,"promote":False,
            "cohort_sha256":"test", "cases":rows}


def test_no_contract_fails_closed():
    d=evaluate(_fake_paired(),{str(i):"List all cities" for i in range(100)})
    assert (d["baseline_correct"],d["challenger_correct"],d["selected_correct"])==(64,59,64)
    assert d["switched"]==0 and d["net_gain"]==0 and d["promote"] is False
    assert d["no_learning_or_gold_in_selection"] is True


def test_uses_frozen_gold_only_after_all_deterministic_decisions():
    paired=_fake_paired()
    questions={str(i):"How many distinct cities are present?" for i in range(100)}
    d=evaluate(paired,questions)
    assert d["switched"]==100
    assert d["selected_correct"]==59
    assert d["rescued"]==4 and d["regressed"]==9
    assert d["net_gain"]==-5
    assert sum(x["chosen"]=="a1" for x in (y["decision"] for y in d["decisions"]))==100


def test_corrupt_or_incomplete_evidence_rejected():
    with pytest.raises(ValueError,match="missing original"):
        requirements("")
    paired=_fake_paired()
    paired["cases"][0]["case_id"]=paired["cases"][1]["case_id"]
    with pytest.raises(ValueError,match="duplicated"):
        evaluate(paired,{str(i):"List all cities" for i in range(100)})
    paired=_fake_paired()
    paired["cases"][0]["ex"]["a0"]=False
    with pytest.raises(ValueError,match="labels diverged"):
        evaluate(paired,{str(i):"List all cities" for i in range(100)})
