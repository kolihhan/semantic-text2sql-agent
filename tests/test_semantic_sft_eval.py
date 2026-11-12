"""Predeclared independent BIRD DEV evaluation excludes the explored frozen100."""
from experiments.semantic_sft_eval import choose_eval_cases, compare_paired_scores, reserve_semantic_audit_holdout


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


def test_sql_sft_never_scores_cases_reserved_by_independent_audit():
    rows = make_cases()
    prior = {str(i) for i in range(100)}
    reserved = reserve_semantic_audit_holdout(rows, historical_frozen_ids=prior)
    reversed_reserved = reserve_semantic_audit_holdout(
        list(reversed(rows)), historical_frozen_ids=prior,
    )
    assert reserved == reversed_reserved
    assert len(reserved) == 100
    assert not reserved.intersection(prior)
    sft = choose_eval_cases(
        rows, frozen_ids=prior | reserved, mode="independent", count=40,
    )
    assert not {str(x["question_id"]) for x in sft}.intersection(prior | reserved)


def test_semantic_audit_reservation_matches_other_shards_seed_contract():
    from hashlib import sha256
    rows = make_cases()
    prior = {str(i) for i in range(100)}
    # Mirror PR #14's published selector contract as an independent test oracle.
    ranked = sorted(
        (str(x["question_id"]) for x in rows if str(x["question_id"]) not in prior),
        key=lambda cid: (
            sha256(("semantic-sql-shards-independent-dev-v1:" + cid).encode()).hexdigest(),
            cid,
        ),
    )
    assert reserve_semantic_audit_holdout(
        rows, historical_frozen_ids=prior,
    ) == set(ranked[:100])


def test_audit_reservation_rejects_invalid_population():
    import pytest
    with pytest.raises(ValueError, match="duplicated"):
        reserve_semantic_audit_holdout(
            [{"question_id": 1}, {"question_id": 1}], historical_frozen_ids=set(),
        )
    with pytest.raises(ValueError, match="insufficient"):
        reserve_semantic_audit_holdout(
            make_cases()[:99], historical_frozen_ids=set(), count=100,
        )
