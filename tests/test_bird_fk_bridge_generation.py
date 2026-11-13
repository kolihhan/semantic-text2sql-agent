"""CPU-only contracts for independent BIRD bridge candidate-generation experiment."""
import pytest
from experiments.bird_fk_bridge_generation import (
    expand_fk_bridges, aggregate, generate_question_grounded, select_new100,
)


def test_only_declared_intermediate_bridge_is_added():
    fk = [("members", "id", "enrollments", "member_id"),
          ("enrollments", "course_id", "courses", "id"),
          ("courses", "tag", "unrelated", "course")]
    selected = ["members", "courses"]
    result = expand_fk_bridges(selected, fk)
    assert result == ["members", "courses", "enrollments"]
    assert selected == ["members", "courses"]


def test_no_hallucinated_bridge_or_combinatorial_expansion():
    assert expand_fk_bridges(["a", "z"], [("a", "id", "x", "id")]) == ["a", "z"]
    assert expand_fk_bridges(["a", "c"], [
        ("a", "id", "b", "id"),
        ("b", "id", "c", "id"),
    ], max_added=0) == ["a", "c"]
    with pytest.raises(ValueError):
        expand_fk_bridges(["a", "a"], [])


def test_question_prompt_contains_grounding_but_no_gold_or_baseline():
    class Provider:
        def complete_text(self, *, system, user):
            assert "directional relations" in user or "Respect directional relations" in user
            assert "Which employees" in user
            assert 'Table: "staff"' in user
            assert "gold SQL" not in user
            return "SELECT name FROM staff"
    value = generate_question_grounded(
        Provider(), question="Which employees?", evidence="",
        physical_schema='Table: "staff"')
    assert value == "SELECT name FROM staff"


def test_golden_cohort_not_guessed_on_changed_population():
    with pytest.raises(ValueError, match="second cohort diverged"):
        select_new100([str(i) for i in range(500)], set(map(str, range(100))))


def test_paired_oracle_accounting_and_reject_shard_drift():
    shards = []
    for i in range(4):
        cases = []
        for j in range(25):
            k = i*25+j
            a0 = k < 65
            a1 = k < 65 or k in {65, 66, 67}
            if k == 2:
                a1 = False
            cases.append({
                "case_id":str(k), "ex":{"a0":a0,"a1":a1},
                "changed":True,"new_bridge_tables":["join_table"] if j==0 else [],
            })
        shards.append({"schema":"bird-fk-bridge-shard-v1",
            "shard":i,"model":"test","cohort_sha256":"frozen",
            "gold_used_for_generation":False,"cases":cases})
    report=aggregate(shards)
    assert report["baseline_correct"]==65
    assert report["question_grounded_correct"]==67
    assert report["wrong_to_correct"]==3
    assert report["correct_to_wrong"]==1
    assert report["candidate_oracle_correct"]==68
    assert report["bridge_used"]==4
    assert report["promote"] is False
    shards[0]["gold_used_for_generation"]=True
    with pytest.raises(ValueError):
        aggregate(shards)
