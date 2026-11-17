"""CPU-only BIRD EX core contract audit; not full official evaluator parity.

Pinned reference: AlibabaResearch/DAMO-ConvAI bird/llm/src/evaluation.py
commit 483554eae102996f5ec1f4feab4e78ef29c2a394.
"""
import sqlite3
import pytest
from evaluation.run_bird import _official_ex, _official_ex_results
from semantic_sql.execution import execute_readonly, ExecutionResult

@pytest.fixture
def database(tmp_path):
    path = tmp_path / "source.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE events (group_id INTEGER, label TEXT, score REAL);
            INSERT INTO events VALUES (1, 'Alpha', 1.0);
            INSERT INTO events VALUES (1, 'Alpha', 1.0);
            INSERT INTO events VALUES (2, 'alpha', 2.0);
            INSERT INTO events VALUES (3, NULL, NULL);
            CREATE TABLE tags (group_id INTEGER, tag TEXT);
            INSERT INTO tags VALUES (1, 'A');
            INSERT INTO tags VALUES (2, 'B');
        """)
    return path

def pinned_official_core(database, prediction, gold):
    """Mirror upstream execute_sql's cursor.execute/fetchall/set comparison."""
    try:
        with sqlite3.connect(database) as connection:
            cursor = connection.cursor()
            cursor.execute(prediction)
            pred_rows = cursor.fetchall()
            cursor.execute(gold)
            gold_rows = cursor.fetchall()
        return set(pred_rows) == set(gold_rows)
    except sqlite3.Error:
        return False

@pytest.mark.parametrize("prediction,gold", [
    ("SELECT label FROM events", "SELECT label FROM events"),
    ("SELECT label FROM events ORDER BY label DESC", "SELECT label FROM events"),
    ("SELECT DISTINCT group_id FROM events", "SELECT group_id FROM events"),
    ("SELECT label FROM events WHERE label='Alpha'", "SELECT label FROM events WHERE label='alpha'"),
    ("SELECT score FROM events WHERE score IS NULL", "SELECT score FROM events WHERE score IS NULL"),
    ("SELECT COUNT(*) FROM events", "SELECT COUNT(DISTINCT group_id) FROM events"),
    ("SELECT group_id FROM events WHERE group_id=999", "SELECT group_id FROM events WHERE group_id=998"),
    ("SELECT e.group_id FROM events e JOIN tags t ON e.group_id=t.group_id", "SELECT group_id FROM tags"),
    ("SELECT label, score FROM events", "SELECT score, label FROM events"),
    ("SELECT missing FROM events", "SELECT label FROM events"),
])
def test_safe_uncapped_local_matches_official_core(database, prediction, gold):
    assert _official_ex(database, prediction, gold, max_rows=50_000) == pinned_official_core(database, prediction, gold)

def test_set_ignores_multiplicity(database):
    sql = "SELECT group_id FROM events"
    gold = "SELECT DISTINCT group_id FROM events"
    assert pinned_official_core(database, sql, gold)
    predicted_rows = execute_readonly(database, sql, max_rows=50_000)
    gold_rows = execute_readonly(database, gold, max_rows=50_000)
    assert predicted_rows.rows != gold_rows.rows
    assert _official_ex_results(predicted_rows, gold_rows)

def test_cap_policy_explicitly_diverges(database):
    sql = "SELECT group_id FROM events"
    assert pinned_official_core(database, sql, sql)
    assert not _official_ex(database, sql, sql, max_rows=2)
    truncated = ExecutionResult(columns=("group_id",), rows=((1,), (1,)), truncated=True)
    assert not _official_ex_results(truncated, truncated)
