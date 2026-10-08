def _case(*, rows: int, official_ex: bool = False):
    return {
        "case_id": "q1",
        "guarded": {
            "status": "ok",
            "official_ex": official_ex,
            "final_sql": "SELECT 1",
            "stages": [{"name": "execute", "summary": f"rows={rows}"}],
        },
    }


def test_select_zero_row_candidates_is_gold_blind_and_targets_successful_empty_queries():
    from experiments.zero_row_repair_frozen100 import select_zero_row_candidates

    empty_wrong = _case(rows=0, official_ex=False)
    nonempty_wrong = _case(rows=2, official_ex=False)
    empty_correct = _case(rows=0, official_ex=True)
    failed = _case(rows=0, official_ex=False)
    failed["guarded"]["status"] = "execution_failed"

    selected = select_zero_row_candidates([empty_wrong, nonempty_wrong, empty_correct, failed])
    assert selected == [empty_wrong, empty_correct]


def test_aggregate_preserves_untouched_baseline_and_counts_real_transitions():
    from experiments.zero_row_repair_frozen100 import aggregate_result

    baseline_cases = [
        {"case_id": "a", "guarded": {"official_ex": True}},
        {"case_id": "b", "guarded": {"official_ex": False}},
        {"case_id": "c", "guarded": {"official_ex": False}},
    ]
    repaired = {
        "a": {"official_ex": False},
        "b": {"official_ex": True},
    }

    result = aggregate_result(baseline_cases, repaired)
    assert result == {
        "sample_size": 3,
        "baseline_correct": 1,
        "treatment_correct": 1,
        "wrong_to_correct": 1,
        "correct_to_wrong": 1,
    }
