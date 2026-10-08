from experiments.qwen9b_zero_row_repair import should_attempt_zero_row_repair


def test_only_successful_zero_row_execution_is_eligible():
    assert should_attempt_zero_row_repair(execution_success=True, row_count=0, truncated=False)
    assert not should_attempt_zero_row_repair(execution_success=True, row_count=1, truncated=False)
    assert not should_attempt_zero_row_repair(execution_success=False, row_count=0, truncated=False)
    assert not should_attempt_zero_row_repair(execution_success=True, row_count=0, truncated=True)
