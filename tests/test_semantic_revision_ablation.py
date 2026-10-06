from evaluation.semantic_revision_ablation import aggregate_pairs, select_smoke_case_ids


def test_select_smoke_case_ids_uses_first_executable_guarded_cases():
    cases = [
        {"case_id": "a", "guarded": {"execution_success": False, "final_sql": "SELECT 1"}},
        {"case_id": "b", "guarded": {"execution_success": True, "final_sql": "SELECT 2"}},
        {"case_id": "c", "guarded": {"execution_success": True, "final_sql": "SELECT 3"}},
        {"case_id": "d", "guarded": {"execution_success": True, "final_sql": "SELECT 4"}},
    ]

    assert select_smoke_case_ids(cases, limit=2) == ["b", "c"]


def test_aggregate_pairs_counts_semantic_revision_transitions():
    rows = [
        {
            "baseline": {"official_ex": False, "execution_success": True},
            "revised": {"official_ex": True, "execution_success": True},
        },
        {
            "baseline": {"official_ex": True, "execution_success": True},
            "revised": {"official_ex": True, "execution_success": True},
        },
        {
            "baseline": {"official_ex": True, "execution_success": True},
            "revised": {"official_ex": False, "execution_success": False},
        },
    ]

    result = aggregate_pairs(rows)

    assert result["baseline"]["official_bird_ex"] == 2 / 3
    assert result["revised"]["official_bird_ex"] == 2 / 3
    assert result["transitions"] == {
        "wrong_to_correct": 1,
        "correct_to_wrong": 1,
        "execution_fail_to_success": 0,
        "execution_success_to_fail": 1,
    }


def test_aggregate_pairs_reports_zero_rows_as_error():
    try:
        aggregate_pairs([])
    except ValueError as exc:
        assert "zero" in str(exc)
    else:
        raise AssertionError("expected ValueError")
