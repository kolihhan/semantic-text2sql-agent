from evaluation.semantic_sketch_ablation import aggregate_pairs, select_pilot_case_ids


def test_select_pilot_case_ids_uses_first_frozen_direct_failures():
    cases = [
        {"case_id": "a", "direct": {"official_ex": True}},
        {"case_id": "b", "direct": {"official_ex": False}},
        {"case_id": "c", "direct": {"official_ex": False}},
        {"case_id": "d", "direct": {"official_ex": True}},
        {"case_id": "e", "direct": {"official_ex": False}},
    ]

    assert select_pilot_case_ids(cases, limit=2) == ["b", "c"]


def test_aggregate_pairs_passes_only_on_positive_ex_without_execution_regression():
    rows = [
        {
            "baseline": {"official_ex": False, "execution_success": True},
            "semantic": {"official_ex": True, "execution_success": True},
        },
        {
            "baseline": {"official_ex": False, "execution_success": False},
            "semantic": {"official_ex": False, "execution_success": True},
        },
    ]

    result = aggregate_pairs(rows)

    assert result["transitions"] == {
        "wrong_to_correct": 1,
        "correct_to_wrong": 0,
        "execution_fail_to_success": 1,
        "execution_success_to_fail": 0,
    }
    assert result["baseline"]["official_bird_ex"] == 0.0
    assert result["semantic"]["official_bird_ex"] == 0.5
    assert result["pilot_gate_pass"] is True


def test_aggregate_pairs_rejects_execution_regression_even_with_ex_gain():
    rows = [
        {
            "baseline": {"official_ex": False, "execution_success": True},
            "semantic": {"official_ex": True, "execution_success": False},
        }
    ]

    assert aggregate_pairs(rows)["pilot_gate_pass"] is False
