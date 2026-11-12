"""Check diagnostic counts without exposing or tuning real BIRD holdout cases."""
from copy import deepcopy

import pytest

from experiments.analyze_semantic_gap import analyze_semantic_gap


def _case(n, base, revision, arm, *, base_valid=True, revision_valid=True, base_missing=None, revised_missing=None, reason="question_clause"):
    return {
        "case_id": str(n),
        "baseline_ex": base,
        "revision_ex": revision,
        "selected_ex": revision if arm == "semantic_revision" else base,
        "baseline_valid": base_valid,
        "revision_valid": revision_valid,
        "changed": True,
        "chosen_arm": arm,
        "reason": reason,
        "baseline_violations": base_missing if base_missing is not None else ["count"],
        "revised_violations": revised_missing if revised_missing is not None else [],
    }


def _aggregate():
    cases = [
        _case(1, False, False, "mschema"),
        _case(2, False, True, "semantic_revision"),
        _case(3, False, True, "mschema"),
        _case(4, True, False, "mschema"),
        _case(5, True, False, "semantic_revision"),
        _case(6, True, True, "semantic_revision"),
        _case(7, False, False, "semantic_revision", base_valid=False),
    ]
    return {
        "schema": "semantic-shard-independent-aggregate-v1",
        "sample_size": 7,
        "cohort_sha256": "frozen-synthetic-test",
        "cases": cases,
        "baseline_correct": 3,
        "revision_correct": 3,
        "selected_correct": 3,
        "net_gain": 0,
        "wrong_to_correct": 1,
        "correct_to_wrong": 1,
    }


def test_diagnostics_separate_generation_selector_and_execution_only_errors():
    result = analyze_semantic_gap(_aggregate())
    c = result["counts"]
    assert result["count"] == 7 and result["net_gain"] == 0
    assert c["both_candidates_wrong"] == 2
    assert c["revision_only_correct"] == 2
    assert c["revision_opportunity_missed"] == 1
    assert c["revision_rescued"] == 1
    assert c["selection_regression"] == 1
    assert c["revision_regression_prevented"] == 1
    assert c["baseline_executable_but_wrong"] == 3
    assert c["executable_only_repair"] == 1
    assert c["surface_contract_reduced_but_still_wrong"] == 4
    assert result["selection_reasons"]["question_clause"]["cases"] == 7


def test_diagnostics_reject_population_and_accounting_corruption():
    good = _aggregate()
    for corrupted in [
        {"schema": "fake", **{k: v for k, v in good.items() if k != "schema"}},
        {**good, "net_gain": 2},
        {**good, "selected_correct": 4},
        {**good, "sample_size": 8},
    ]:
        with pytest.raises(ValueError):
            analyze_semantic_gap(corrupted)
    duplicate = deepcopy(good)
    duplicate["cases"][1]["case_id"] = "1"
    with pytest.raises(ValueError, match="duplicated"):
        analyze_semantic_gap(duplicate)
    conflicting = deepcopy(good)
    conflicting["cases"][1]["selected_ex"] = False
    with pytest.raises(ValueError, match="selected outcome"):
        analyze_semantic_gap(conflicting)
