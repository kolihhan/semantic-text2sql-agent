"""A minimal opt-in postprocessing patch; LangGraph topology stays unchanged."""
from types import SimpleNamespace
import sqlite3

from semantic_sql.contracts import SQLCandidate, VerificationResult
from semantic_sql.inference import _execute
from semantic_sql.projection_repair import repair


QUESTION = "Which league has the most games in 2015?"
SQL = (
    "SELECT l.name, COUNT(g.id) AS games_count FROM league l "
    "JOIN games g ON g.league_id=l.id GROUP BY l.name "
    "ORDER BY games_count DESC LIMIT 1"
)


def test_single_structural_projection_change():
    revised, reason = repair(QUESTION, SQL)
    assert reason == "removed_unused_ranking_metrics"
    assert revised == (
        "SELECT l.name FROM league l JOIN games g ON g.league_id=l.id "
        "GROUP BY l.name ORDER BY COUNT(g.id) DESC LIMIT 1"
    )
    assert repair(QUESTION, revised)[0] == revised


def test_explicit_metric_requests_never_change():
    for question in (
        "List the top nine districts by the number of female account holders.",
        "Name the driver with the most winning. Mention his nationality and what is his maximum point scores.",
        "Which league has the most games and the count?",
    ):
        assert repair(question, SQL)[0] == SQL


def test_unsafe_shapes_remain_unchanged():
    examples = (
        SQL.replace(" LIMIT 1", ""),
        SQL.replace(" GROUP BY l.name", ""),
        SQL.replace(" ORDER BY", " HAVING COUNT(g.id)>0 ORDER BY"),
        "WITH a AS (SELECT * FROM league) " + SQL,
    )
    for example in examples:
        assert repair(QUESTION, example)[0] == example


def test_existing_execute_node_opt_in_and_default_off(tmp_path):
    db = tmp_path / "small.sqlite"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            "CREATE TABLE league(id INTEGER PRIMARY KEY, name TEXT);"
            "CREATE TABLE games(id INTEGER PRIMARY KEY, league_id INTEGER);"
            "INSERT INTO league VALUES (1,'Alpha'), (2,'Beta');"
            "INSERT INTO games VALUES (1,1), (2,1), (3,2);"
        )
    state = {
        "question": QUESTION, "candidate": SQLCandidate(SQL),
        "verification": VerificationResult(ok=True), "stages": (),
    }
    off = SimpleNamespace(context=SimpleNamespace(
        database=db, max_rows=100, projection_only=False,
    ))
    on = SimpleNamespace(context=SimpleNamespace(
        database=db, max_rows=100, projection_only=True,
    ))
    original = _execute(state, off)["result"]
    changed = _execute(state, on)["result"]
    assert original.rows == (("Alpha", 2),)
    assert original.candidate.sql == SQL
    assert not any(stage.name == "projection_patch" for stage in original.stages)
    assert changed.rows == (("Alpha",),)
    assert changed.columns == ("name",)
    assert "COUNT(g.id) DESC" in changed.candidate.sql
    assert changed.verification.ok
    assert any(stage.name == "projection_patch" for stage in changed.stages)
