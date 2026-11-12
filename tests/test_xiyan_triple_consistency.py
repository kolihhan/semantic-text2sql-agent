"""Gold-blind SQL candidate selection tests on a tiny independent SQLite fixture."""
import sqlite3
import pytest
from experiments.xiyan_triple_consistency import consensus_choice, aggregate_triple_shards


def make_database(tmp_path):
    path = tmp_path / "facts.sqlite"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            "CREATE TABLE facts(id INTEGER PRIMARY KEY, a TEXT);"
            "INSERT INTO facts VALUES(1,'a'),(2,'b'),(3,'c');"
        )
    return path


def test_three_way_consensus_can_recover_lexical_instead_of_mschema(tmp_path):
    db = make_database(tmp_path)
    # Both possible choices execute successfully. Third candidate breaks tie.
    out = consensus_choice(
        db, lexical_sql="SELECT a FROM facts WHERE id=1",
        mschema_sql="SELECT a FROM facts WHERE id=2",
        independent_sql="SELECT a FROM facts WHERE id=1",
    )
    assert out["chosen_arm"] == "lexical"
    assert out["reason"] == "execution_agreement"
    assert out["valid_candidates"] == 3


def test_three_way_consensus_can_choose_mschema(tmp_path):
    db = make_database(tmp_path)
    out = consensus_choice(
        db, lexical_sql="SELECT a FROM facts WHERE id=1",
        mschema_sql="SELECT a FROM facts WHERE id=2",
        independent_sql="SELECT a FROM facts WHERE id=2",
    )
    assert out["chosen_arm"] == "mschema"


def test_three_way_consensus_not_fooled_by_sql_surface(tmp_path):
    db = make_database(tmp_path)
    out = consensus_choice(
        db, lexical_sql="SELECT a FROM facts WHERE id=1",
        mschema_sql="SELECT a FROM facts WHERE id=2",
        independent_sql="SELECT a FROM facts WHERE 2=id",
    )
    assert out["chosen_arm"] == "mschema"
    assert out["valid_candidates"] == 3


def test_no_majority_falls_back_to_frozen_mschema(tmp_path):
    db = make_database(tmp_path)
    out = consensus_choice(
        db, lexical_sql="SELECT a FROM facts WHERE id=1",
        mschema_sql="SELECT a FROM facts WHERE id=2",
        independent_sql="SELECT a FROM facts WHERE id=3",
    )
    assert out["chosen_arm"] == "mschema"
    assert out["reason"] == "no_agreement_mschema_fallback"


def test_invalid_or_truncated_sql_does_not_create_false_consensus(tmp_path):
    db = make_database(tmp_path)
    out = consensus_choice(
        db, lexical_sql="SELECT no_column FROM facts",
        mschema_sql="SELECT a FROM facts WHERE id=1",
        independent_sql="SELECT a FROM facts WHERE id=1",
    )
    assert out["chosen_arm"] == "mschema"
    assert out["valid_candidates"] == 2
    out = consensus_choice(
        db, lexical_sql="SELECT a FROM facts WHERE id=1",
        mschema_sql="SELECT a FROM facts ORDER BY id",
        independent_sql="SELECT a FROM facts WHERE id=1",
        max_rows=2,
    )
    assert out["chosen_arm"] == "lexical"
    assert out["valid_candidates"] == 2


def test_read_only_guard_does_not_execute_update(tmp_path):
    db = make_database(tmp_path)
    out = consensus_choice(
        db, lexical_sql="UPDATE facts SET a='hacked'",
        mschema_sql="SELECT a FROM facts WHERE id=1",
        independent_sql="SELECT a FROM facts WHERE id=1",
    )
    assert out["chosen_arm"] == "mschema"
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT a FROM facts WHERE id=1").fetchone()[0] == "a"


def test_aggregate_validates_exact_100_and_frozen_mschema_baseline():
    shards = []
    for shard in range(4):
        cases = []
        for idx in range(shard, 100, 4):
            cases.append({
                "case_id": str(idx), "lexical_ex": idx < 58,
                "mschema_ex": idx < 59, "independent_ex": idx < 60,
                "consensus_ex": idx < 60, "chosen_arm": "independent",
                "reason": "execution_agreement",
            })
        shards.append({"schema": "xiyan-triple-v1", "shard_index": shard,
                       "model": "test", "cases": cases})
    result = aggregate_triple_shards(shards, min_correct=61)
    assert result["lexical_ex"] == 58 and result["mschema_ex"] == 59
    assert result["consensus_ex"] == 60
    assert result["gate_met"] is False
    with pytest.raises(ValueError):
        aggregate_triple_shards(shards[:3], min_correct=61)
    shards[0]["cases"][0]["mschema_ex"] = False
    with pytest.raises(ValueError, match="drift"):
        aggregate_triple_shards(shards, min_correct=61)
