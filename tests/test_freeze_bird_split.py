from scripts.freeze_bird_split import allocate, freeze
import pytest


def fixtures():
    rows = [
        {
            "question_id": i, "db_id": f"db_{i % 3}",
            "question": f"Question {i} with different user task description",
            "SQL": f"SELECT {i}",  # Must never influence split
        }
        for i in range(30)
    ]
    labels = {str(i): ["simple", "moderate", "challenging"][i % 3] for i in range(30)}
    return rows, labels


def test_allocation_no_loss():
    out = allocate({("x", "a"): 2, ("x", "b"): 5, ("y", "b"): 1}, 5)
    assert sum(out.values()) == 5
    assert all(v >= 0 for v in out.values())


def test_deterministic_disjoint_stratified():
    rows, labels = fixtures()
    x = freeze(rows, ["0", "1"], labels, "src", "old", "level", dev_size=15,
               holdout_size=8, historical_expected=2)
    y = freeze(rows, ["0", "1"], labels, "src", "old", "level", dev_size=15,
               holdout_size=8, historical_expected=2)
    assert x == y
    assert len(x["dev_case_ids"]) == 15 and len(x["holdout_case_ids"]) == 8
    assert not (set(x["dev_case_ids"]) & set(x["holdout_case_ids"]))
    assert not ({"0", "1"} & (set(x["dev_case_ids"]) | set(x["holdout_case_ids"])))


def test_gold_sql_changes_cannot_change_split():
    rows, labels = fixtures()
    first = freeze(rows, ["0", "1"], labels, "src", "old", "level", dev_size=15,
                   holdout_size=8, historical_expected=2)
    for row in rows:
        row["SQL"] = "SELECT UNKNOWN_GOLD"
    second = freeze(rows, ["0", "1"], labels, "src", "old", "level", dev_size=15,
                    holdout_size=8, historical_expected=2)
    assert first == second


def test_missing_official_difficulty_fails_closed():
    rows, labels = fixtures()
    labels.pop("7")
    with pytest.raises(ValueError, match="official difficulty"):
        freeze(rows, ["0", "1"], labels, "src", "old", "level",
               dev_size=15, holdout_size=8, historical_expected=2)


def test_historical_ids_required_in_source():
    rows, labels = fixtures()
    with pytest.raises(ValueError, match="historical IDs absent"):
        freeze(rows, ["999", "1"], labels, "src", "old", "level",
               dev_size=15, holdout_size=8, historical_expected=2)


def test_excludes_entire_duplicate_question_group_before_freeze():
    rows, labels = fixtures()
    # IDs 22 and 25 share db_1. The duplicate was never selected using SQL/EX.
    rows[25]["question"] = rows[22]["question"]
    result = freeze(rows, ["0", "1"], labels, "src", "old", "level",
                    dev_size=15, holdout_size=8, historical_expected=2)
    chosen = set(result["dev_case_ids"]) | set(result["holdout_case_ids"])
    assert "22" not in chosen and "25" not in chosen
    assert result["excluded_exact_duplicate_question_groups"] == 1
    assert result["excluded_exact_duplicate_question_ids"] == 2


def test_known_previous_exposures_excluded_from_both_new_splits():
    rows, labels = fixtures()
    result = freeze(rows, ["0", "1"], labels, "src", "old", "level",
                    dev_size=15, holdout_size=8, historical_expected=2,
                    additional_exposed=["4", "5"], additional_exposed_sha="exposure")
    chosen = set(result["dev_case_ids"]) | set(result["holdout_case_ids"])
    assert not ({"0", "1", "4", "5"} & chosen)
    assert result["excluded_additional_exposed_count"] == 2
    assert result["additional_exposure_sha256"] == "exposure"
