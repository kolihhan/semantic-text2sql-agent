import pytest

from evaluation.bird_qwen_rank_intent40 import RANK_CUES, aggregate


def shard(i):
    rows = []
    for j in range(4):
        n = i * 4 + j
        group = "rank" if n < 20 else "control"
        rows.append({
            "case_id": n, "group": group, "wall_s": 10.0,
            "arms": {
                "original": {"ex": n % 2 == 0, "sql": "SELECT 1"},
                "rank_contract": {"ex": n % 3 == 0, "sql": "SELECT 2" if group == "rank" else "SELECT 1"},
            },
        })
    return {"shard": i, "model_digest": "qwen", "cohort_sha": "frozen", "rows": rows}


def test_rank_cues_are_question_only():
    assert RANK_CUES.search("Which school has the highest average score?")
    assert RANK_CUES.search("Find the top 10 newest events.")
    assert not RANK_CUES.search("Find all students in California")


def test_rank_aggregate_validates_40_matched_cases():
    result = aggregate([shard(i) for i in range(10)])
    assert result["n"] == 40
    assert result["groups"]["rank"]["n"] == 20
    assert result["groups"]["control"]["n"] == 20
    assert result["groups"]["rank"]["changed_sql"] == 20
    assert result["groups"]["control"]["changed_sql"] == 0


def test_rank_aggregate_rejects_partial_or_changed():
    with pytest.raises(ValueError, match="ten shards"):
        aggregate([shard(0)])
    cases = [shard(i) for i in range(10)]
    cases[0]["rows"][0]["case_id"] = 10
    with pytest.raises(ValueError, match="unique"):
        aggregate(cases)
