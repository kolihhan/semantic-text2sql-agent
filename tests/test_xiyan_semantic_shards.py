"""Independent, gold-blind semantic SQL reranking contract for CPU Ollama shards."""
from __future__ import annotations

import sqlite3

import pytest

from experiments.xiyan_semantic_shards import (
    build_holdout_ids,
    check_question_constraints,
    choose_semantic_sql,
    aggregate_semantic_shards,
)


def small_db(tmp_path):
    file=tmp_path/"facts.sqlite"
    with sqlite3.connect(file) as db:
        db.executescript(
            "CREATE TABLE facts(id INTEGER PRIMARY KEY, amount INTEGER, kind TEXT);"
            "INSERT INTO facts VALUES (1, 10, 'x'), (2, 20, 'x'), (3, 30, 'y');"
        )
    return file


def test_holdout_excludes_all_100_explored_cases_and_is_order_independent():
    all_ids=[str(i) for i in range(255)]
    explored={str(i) for i in range(100)}
    first=build_holdout_ids(all_ids,explored,count=100)
    second=build_holdout_ids(list(reversed(all_ids)),explored,count=100)
    assert first==second and len(first)==len(set(first))==100
    assert not explored.intersection(first)
    assert len(first[0::4])==25
    assert len(first[1::4])==25
    assert len(first[2::4])==25
    assert len(first[3::4])==25
    with pytest.raises(ValueError,match="holdout"):
        build_holdout_ids(all_ids,explored,count=156)


def test_semantic_rules_count_avg_and_ranking_are_detected():
    assert check_question_constraints("How many facts are there?","SELECT COUNT(*) FROM facts")["missing"]==[]
    assert check_question_constraints("How many facts are there?","SELECT id FROM facts")["missing"]==["count"]
    assert check_question_constraints("What is the average amount?","SELECT AVG(amount) FROM facts")["missing"]==[]
    assert check_question_constraints("What is the average amount?","SELECT SUM(amount) FROM facts")["missing"]==["average"]
    assert check_question_constraints("Which fact has the highest amount?","SELECT id FROM facts ORDER BY amount DESC LIMIT 1")["missing"]==[]
    assert check_question_constraints("Which fact has the highest amount?","SELECT id FROM facts ORDER BY amount ASC LIMIT 1")["missing"]==["maximum"]
    assert check_question_constraints("Which fact has the lowest amount?","SELECT id FROM facts ORDER BY amount ASC LIMIT 1")["missing"]==[]
    assert check_question_constraints("Which fact has the lowest amount?","SELECT id FROM facts ORDER BY amount DESC LIMIT 1")["missing"]==["minimum"]


def test_semantic_selector_replaces_only_provably_missing_explicit_clause(tmp_path):
    db=small_db(tmp_path)
    base="SELECT amount FROM facts"
    improved="SELECT COUNT(*) FROM facts"
    out=choose_semantic_sql(
        db,question="How many facts?",baseline_sql=base,revised_sql=improved
    )
    assert out["chosen_arm"]=="semantic_revision"
    assert out["reason"]=="explicit_question_constraint_recovered"
    assert out["baseline_violations"]==["count"]
    assert out["chosen_violations"]==[]
    assert out["gold_used_for_selection"] is False


def test_unchanged_or_ambiguous_question_preserves_mschema(tmp_path):
    db=small_db(tmp_path)
    base="SELECT id FROM facts WHERE kind='x'"
    changed="SELECT id FROM facts WHERE kind='y'"
    out=choose_semantic_sql(db,question="List IDs",baseline_sql=base,revised_sql=changed)
    assert out["chosen_arm"]=="mschema"
    assert out["reason"]=="no_gold_blind_semantic_evidence"


def test_no_empty_result_bias(tmp_path):
    db=small_db(tmp_path)
    base="SELECT id FROM facts WHERE kind='missing'"
    altered="SELECT id FROM facts WHERE kind='x'"
    selected=choose_semantic_sql(db,question="List ID of missing kind",baseline_sql=base,revised_sql=altered)
    assert selected["chosen_arm"]=="mschema"


def test_sql_mutations_cannot_be_selected_even_if_revision_looks_better(tmp_path):
    db=small_db(tmp_path)
    out=choose_semantic_sql(db,question="How many facts?",baseline_sql="SELECT id FROM facts",
                            revised_sql="DELETE FROM facts")
    assert out["chosen_arm"]=="mschema"
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]==3


def test_broken_base_falls_back_to_valid_repair(tmp_path):
    db=small_db(tmp_path)
    out=choose_semantic_sql(db,question="List IDs",baseline_sql="SELECT invalid FROM facts",
                            revised_sql="SELECT id FROM facts")
    assert out["chosen_arm"]=="semantic_revision"
    assert out["reason"]=="baseline_non_executable"


def test_no_execution_agreement_with_gold_in_selector_signature(tmp_path):
    db=small_db(tmp_path)
    # Selector has no gold/expected SQL or scoring parameters, only question,
    # two proposed SQLs, and the query's physical SQLite database.
    import inspect
    signature=inspect.signature(choose_semantic_sql)
    assert set(signature.parameters)=={"database","question","baseline_sql","revised_sql","max_rows"}
    assert "gold" not in repr(signature).lower()


def test_four_shard_aggregate_requires_exact_disjoint_cohort_and_matching_config():
    shards=[]
    for shard in range(4):
        cases=[]
        for i in range(shard,100,4):
            cases.append({
                "case_id":str(i),"baseline_ex":i<58,
                "revision_ex":i<65,"selected_ex":i<62,
                "chosen_arm":"semantic_revision" if i%2 else "mschema",
                "reason":"test_fixture",
                "changed":i%2==1,
            })
        shards.append({
            "schema":"semantic-shard-independent-v1",
            "model":"test-7b","shard_index":shard,"num_shards":4,
            "cohort_sha256":"sha-test", "cases":cases,
            "gold_visible_to_generation":False,
        })
    result=aggregate_semantic_shards(shards,min_gain=3)
    assert result["sample_size"]==100
    assert result["baseline_correct"]==58
    assert result["revision_correct"]==65
    assert result["selected_correct"]==62
    assert result["net_gain"]==4
    assert result["gate_met"] is True
    assert result["oracle_headroom"]==7
    assert "independent" in result["limitations"].lower()
    with pytest.raises(ValueError):
        aggregate_semantic_shards(shards[:3])
    shards[1]["cohort_sha256"]="different"
    with pytest.raises(ValueError,match="cohort"):
        aggregate_semantic_shards(shards)
