"""A1/A2/A3 mechanism tests on synthetic SQLite databases, never BIRD gold."""
from __future__ import annotations

import sqlite3

import pytest

from experiments.semantic_layer_abc import (
    aggregate, choose_disjoint_ids, contract_from_question, observe_semantics,
    run_one, _literal_lock,
)


class Responses:
    def __init__(self, *texts):
        self.responses = list(texts)
        self.calls = 0

    def complete_text(self, *, system, user):
        self.calls += 1
        assert "SQL" in user
        return self.responses.pop(0)


@pytest.fixture
def facts(tmp_path):
    db = tmp_path / "facts.sqlite"
    with sqlite3.connect(db) as con:
        con.executescript(
            "CREATE TABLE facts(id INTEGER PRIMARY KEY, amount REAL, kind TEXT);"
            "INSERT INTO facts VALUES (1, 6.0, 'x'), (2, 10.0, 'x'), (3, 12.0, 'y');"
        )
    return db


def test_retrieval_contract_provenance_is_partial_and_does_not_invent_grain():
    contract = contract_from_question("How many facts exceed 6.5?", "")
    assert contract["operation"] == "count"
    assert contract["operation_source_span"] == "How many"
    assert contract["grain"] == "unknown"
    assert contract["schema_bindings"] == "unknown"
    assert contract["literal_evidence"] == ["6.5"]


def test_a1_marks_only_direct_plain_output_as_proven_mismatch():
    assert observe_semantics("How many facts?", "", "SELECT id FROM facts")["reason"] == "count_output_missing"
    assert observe_semantics("What is the mean amount?", "", "SELECT amount FROM facts")["reason"] == "average_output_missing"
    assert observe_semantics("How many facts?", "", "SELECT COUNT(*) FROM facts")["verdict"] == "UNKNOWN"
    assert observe_semantics("How many facts?", "", "SELECT id FROM facts GROUP BY id HAVING COUNT(*)>0")["verdict"] == "PROVEN_MISMATCH"
    assert observe_semantics("List the facts", "", "SELECT id FROM facts")["verdict"] == "UNKNOWN"
    assert observe_semantics("How many facts?", "", "DELETE FROM facts")["reason"] == "sql_unparseable"


def test_a1_never_changes_output_or_calls_provider(facts):
    provider = Responses()
    base = "SELECT COUNT(*) FROM facts"
    result = run_one(facts, question="How many facts?", evidence="", schema="facts(id,amount,kind)", baseline_sql=base, provider=provider)
    assert result["sql"]["a0"] == result["sql"]["a1"] == result["sql"]["a2"] == result["sql"]["a3"]
    assert provider.calls == 0


def test_a2_fixes_only_a_noncompiling_baseline(facts):
    provider = Responses("SELECT id FROM facts")
    base = "SELECT imaginary FROM facts"
    result = run_one(facts, question="List IDs", evidence="", schema="facts(id,amount,kind)", baseline_sql=base, provider=provider)
    assert result["sql"]["a0"] == result["sql"]["a1"] == base
    assert result["sql"]["a2"] == result["sql"]["a3"] == "SELECT id FROM facts"
    assert result["a2_reason"] == "compiler_repair_accepted"
    assert provider.calls == 1


def test_a3_corrects_missing_output_aggregation_without_population_change(facts):
    provider = Responses("SELECT COUNT(*) FROM facts WHERE kind = 'x'")
    base = "SELECT id FROM facts WHERE kind = 'x'"
    result = run_one(facts, question="How many facts have kind x?", evidence="", schema="facts(id,amount,kind)", baseline_sql=base, provider=provider)
    assert result["sql"]["a2"] == base
    assert result["sql"]["a3"] == "SELECT COUNT(*) FROM facts WHERE kind = 'x'"
    assert result["a3_reason"] == "semantic_output_repair_accepted"
    assert provider.calls == 1


def test_a3_rejects_changed_predicate_even_when_count_fixed(facts):
    provider = Responses("SELECT COUNT(*) FROM facts WHERE kind = 'y'")
    base = "SELECT id FROM facts WHERE kind = 'x'"
    result = run_one(facts, question="How many facts have kind x?", evidence="", schema="facts(id,amount,kind)", baseline_sql=base, provider=provider)
    assert result["sql"]["a3"] == base
    assert result["a3_reason"] == "semantic_repair_rejected"
    assert provider.calls == 1


def test_literal_source_number_is_locked(facts):
    original = "SELECT id FROM facts WHERE amount > 6.5"
    revised = "SELECT COUNT(*) FROM facts WHERE amount > 8.0"
    assert not _literal_lock("How many facts have amount > 6.5?", original, revised)
    assert _literal_lock("How many facts have amount > 6.5?", original, "SELECT COUNT(*) FROM facts WHERE amount > 6.5")
    provider = Responses(revised)
    result = run_one(facts, question="How many facts have amount > 6.5?", evidence="", schema="facts(id,amount,kind)", baseline_sql=original, provider=provider)
    assert result["sql"]["a3"] == original
    assert result["a3_reason"] == "semantic_repair_rejected"


def test_unsafe_candidate_rejected_and_no_gpu_or_gold_needed(facts):
    provider = Responses("DELETE FROM facts")
    result = run_one(facts, question="How many facts?", evidence="", schema="facts(id,amount,kind)", baseline_sql="SELECT id FROM facts", provider=provider)
    assert result["a3_reason"] == "semantic_repair_rejected"
    with sqlite3.connect(facts) as con:
        assert con.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 3


def test_heldout_question_selection_excludes_all_previous_200():
    all_ids = [str(i) for i in range(405)]
    old = set(all_ids[:200])
    selected = choose_disjoint_ids(all_ids, old)
    assert len(selected) == 100
    assert set(selected).isdisjoint(old)
    assert selected == choose_disjoint_ids(list(reversed(all_ids)), old)
    with pytest.raises(ValueError, match="insufficient"):
        choose_disjoint_ids(all_ids[:250], old, count=100)


def test_aggregate_checks_all_four_arms_and_strict_pairing():
    shards = []
    for shard in range(4):
        cases = []
        for i in range(shard, 100, 4):
            a0 = i < 50
            a2 = i < 52
            a3 = i < 53
            cases.append({
                "case_id": str(i), "extra_model_calls": int(i >= 50),
                "latency_s": 2.0, "a2_reason": "unchanged",
                "a3_reason": "unchanged",
                "ex": {"a0": a0, "a1": a0, "a2": a2, "a3": a3},
            })
        shards.append({"schema": "semantic-abc-shard-v1", "shard": shard,
                       "model": "7b", "cohort_sha256": "hash",
                       "gold_used_for_routing": False, "cases": cases})
    report = aggregate(shards)
    assert report["scores"] == {"a0": 50, "a1": 50, "a2": 52, "a3": 53}
    assert report["paired_vs_a0"]["a3"]["net_gain"] == 3
    with pytest.raises(ValueError):
        aggregate(shards[:3])
    shards[1]["cases"][0]["case_id"] = "0"
    with pytest.raises(ValueError, match="unique"):
        aggregate(shards)


def test_protected_literal_does_not_block_innocent_sql_alias_change(facts):
    provider = Responses("SELECT COUNT(*) AS total FROM facts WHERE amount > 6.5")
    base = "SELECT id FROM facts WHERE amount > 6.5"
    result = run_one(facts, question="How many facts have amount > 6.5?", evidence="", schema="facts(id,amount,kind)", baseline_sql=base, provider=provider)
    assert result["a3_reason"] == "semantic_output_repair_accepted"
