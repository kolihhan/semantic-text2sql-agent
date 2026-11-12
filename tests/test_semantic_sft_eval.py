"""Predeclared independent BIRD DEV evaluation excludes the explored frozen100."""
from experiments.semantic_sft_eval import choose_eval_cases, compare_paired_scores


def make_cases():
    return [{"question_id": i, "db_id": "db" + str(i % 7), "question": "Question " + str(i)}
            for i in range(240)]


def test_frozen_and_independent_evaluation_are_disjoint_and_deterministic():
    rows = make_cases()
    explored = {str(i) for i in range(100)}
    a = choose_eval_cases(rows, frozen_ids=explored, mode="independent", count=100, seed="v1")
    b = choose_eval_cases(list(reversed(rows)), frozen_ids=explored, mode="independent", count=100, seed="v1")
    assert [str(r["question_id"]) for r in a] == [str(r["question_id"]) for r in b]
    assert not set(str(r["question_id"]) for r in a).intersection(explored)
    f = choose_eval_cases(rows, frozen_ids=explored, mode="frozen", count=100, seed="v1")
    assert {str(r["question_id"]) for r in f} == explored


def test_paired_eval_gate_requires_actual_semantic_gain():
    left = [{"case_id": str(i), "correct": i < 50} for i in range(100)]
    right = [{"case_id": str(i), "correct": i < 54} for i in range(100)]
    out = compare_paired_scores(left, right, min_gain=3)
    assert out["base_correct"] == 50
    assert out["tuned_correct"] == 54
    assert out["wrong_to_correct"] == 4
    assert out["correct_to_wrong"] == 0
    assert out["gate_met"] is True
    right[10]["correct"] = False
    out = compare_paired_scores(left, right, min_gain=3)
    assert out["gate_met"] is True
    assert out["net_gain"] == 3


def test_paired_comparison_rejects_missing_or_duplicate_identity():
    left = [{"case_id": "a", "correct": True}, {"case_id": "b", "correct": False}]
    bad = [{"case_id": "a", "correct": False}, {"case_id": "a", "correct": False}]
    try:
        compare_paired_scores(left, bad)
        assert False
    except ValueError as exc:
        assert "case" in str(exc).lower()
