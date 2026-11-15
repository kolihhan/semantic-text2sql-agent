import sqlite3
from evaluation.projection_bird100_replay import row_replay, aggregate


def make_fixture(tmp_path):
    db = tmp_path / "test.sqlite"
    with sqlite3.connect(db) as c:
        c.executescript(
            "CREATE TABLE T (name TEXT, qty INT);"
            "INSERT INTO T VALUES ('A',3),('A',4),('B',2);"
        )
    case = {
        "case_id": "1", "database_id": "t", "question": "Which name has the most quantity?",
        "guarded": {
            "status": "ok", "official_ex": False,
            "final_sql": "SELECT name, SUM(qty) AS n FROM T GROUP BY name ORDER BY n DESC LIMIT 1",
        },
    }
    return db, case


def test_replays_frozen_model_sql_and_only_removes_metric(tmp_path):
    db, case = make_fixture(tmp_path)
    r = row_replay(case, "SELECT name FROM T GROUP BY name ORDER BY SUM(qty) DESC LIMIT 1", db)
    assert r["before"]["set_ex"] is False
    assert r["after"]["set_ex"] is True
    assert r["triggered"] is True
    assert r["repaired_sql"] == "SELECT name FROM T GROUP BY name ORDER BY SUM(qty) DESC LIMIT 1"


def test_correct_untriggered_query_cannot_regress(tmp_path):
    db, case = make_fixture(tmp_path)
    case["guarded"]["official_ex"] = True
    case["guarded"]["final_sql"] = "SELECT name FROM T GROUP BY name ORDER BY SUM(qty) DESC LIMIT 1"
    r = row_replay(case, case["guarded"]["final_sql"], db)
    assert not r["triggered"]
    assert r["before"]["set_ex"] and r["after"]["set_ex"]


def test_original_ex_label_drift_is_fatal(tmp_path):
    import pytest
    db, case = make_fixture(tmp_path)
    case["guarded"]["official_ex"] = True
    with pytest.raises(ValueError, match="historical Guarded EX mismatch"):
        row_replay(case, "SELECT name FROM T GROUP BY name ORDER BY SUM(qty) DESC LIMIT 1", db)
