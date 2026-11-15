"""Gold-blind and paired-decision checks for answer-grain experiment."""
import sqlite3
from pathlib import Path

import pytest

from evaluation.qwen_answer_grain_spider import freeze, aggregate, PROTOCOL, EXCLUDED_DB


def test_freeze_never_uses_reference_or_previously_exposed_databases(tmp_path: Path):
    dev = []
    for i in range(8):
        db = f"fresh_db_{i}"
        folder = tmp_path / db
        folder.mkdir()
        with sqlite3.connect(folder / (db + ".sqlite")) as conn:
            conn.execute("CREATE TABLE a(id INTEGER, val TEXT)")
            conn.execute("CREATE TABLE b(id INTEGER, val TEXT)")
        for j in range(2):
            dev.append({
                "db_id": db,
                "question": f"Which different entries for each city {i} correspond to group {j}?",
                "query": "SECRET_GOLD_MUST_NOT_LEAK"
            })
    for old in EXCLUDED_DB:
        dev.append({
            "db_id": old,
            "question": "Which different entries for each city group?",
            "query": "SECRET"
        })
    train = [{"db_id": "train_db", "query": "SECRET"}]
    one = freeze(train, dev, tmp_path)
    two = freeze(train, dev, tmp_path)
    assert one == two
    assert one["total"] == 12 and one["physical_dbs"] == 6
    assert all(x["db_id"] not in EXCLUDED_DB for x in one["cases"])
    assert all("query" not in x and "SECRET" not in str(x) for x in one["cases"])


def test_freeze_requires_real_databases(tmp_path):
    with pytest.raises(ValueError, match="insufficient"):
        freeze([], [], tmp_path)


def test_aggregate_positive_means_true_pair_wins_with_no_regressions():
    shards = []
    for shard in range(3):
        cases = []
        for j in range(4):
            i = shard * 4 + j
            old = i < 4
            new = old or i in (4, 6)
            cases.append({
                "id": f"db{i // 2}:{i}", "db_id": f"db{i // 2}",
                "db_sha256": f"sha{i // 2}", "scorable": True, "plan_seconds": 1.0,
                "baseline": {"set_ex": old, "bag_ex": old, "calls": 1, "seconds": 2.0},
                "grain": {"set_ex": new, "bag_ex": new, "calls": 1, "seconds": 2.0},
            })
        shards.append({
            "protocol": PROTOCOL, "model_digest": "test", "cohort_sha256": "frozen",
            "shard": shard, "cases": cases
        })
    s = aggregate(shards, expected_digest="test")
    assert s["arms"]["baseline"]["set_ex"] == 4
    assert s["arms"]["grain"]["set_ex"] == 6
    assert len(s["rescues"]) == 2 and not s["regressions"]
    assert s["positive_screen_only"] and not s["promotion_allowed"]
    with pytest.raises(ValueError, match="matrix"):
        aggregate(shards[:2], expected_digest="test")
