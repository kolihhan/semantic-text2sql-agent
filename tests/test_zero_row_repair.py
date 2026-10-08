from experiments.qwen9b_zero_row_repair import aggregate_rows, should_attempt_repair


def test_zero_row_repair_only_triggers_on_successful_empty_execution():
    assert should_attempt_repair(execution_success=True, truncated=False, row_count=0)
    assert not should_attempt_repair(execution_success=True, truncated=False, row_count=1)
    assert not should_attempt_repair(execution_success=False, truncated=False, row_count=0)
    assert not should_attempt_repair(execution_success=True, truncated=True, row_count=0)


def test_aggregate_requires_gain_without_regression_for_promotion():
    rows = [
        {"baseline_ex": True, "treatment_ex": True, "repair_attempted": False, "repair_changed": False},
        {"baseline_ex": False, "treatment_ex": True, "repair_attempted": True, "repair_changed": True},
        {"baseline_ex": False, "treatment_ex": False, "repair_attempted": True, "repair_changed": True},
    ]
    summary = aggregate_rows(rows)
    assert summary["baseline_correct"] == 1
    assert summary["treatment_correct"] == 2
    assert summary["wrong_to_correct"] == 1
    assert summary["correct_to_wrong"] == 0
    assert summary["promote"] is True


def test_aggregate_rejects_regression_even_if_net_score_improves():
    rows = [
        {"baseline_ex": True, "treatment_ex": False, "repair_attempted": True, "repair_changed": True},
        {"baseline_ex": False, "treatment_ex": True, "repair_attempted": True, "repair_changed": True},
        {"baseline_ex": False, "treatment_ex": True, "repair_attempted": True, "repair_changed": True},
    ]
    summary = aggregate_rows(rows)
    assert summary["treatment_correct"] > summary["baseline_correct"]
    assert summary["correct_to_wrong"] == 1
    assert summary["promote"] is False
