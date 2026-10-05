import sqlite3

from semantic_sql.inference import run_guarded


class SequenceProvider:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.calls = 0

    def complete_text(self, *, system: str, user: str) -> str:
        self.calls += 1
        if not self.responses:
            raise AssertionError("unexpected model call")
        return self.responses.pop(0)


def test_semantic_analysis_runs_once_before_initial_sql(tmp_path):
    database = tmp_path / "db.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE orders (id INTEGER)")
        connection.execute("INSERT INTO orders VALUES (1)")

    provider = SequenceProvider(
        '{"outputs":["count of orders"],"filters":[],"relations":[],'
        '"aggregation":"count","grain":"one scalar","group_by":[],'
        '"ordering":"","limit":""}',
        "SELECT COUNT(*) FROM orders",
    )

    result = run_guarded(
        database=database,
        provider=provider,
        question="How many orders are there?",
        schema_context="orders(id)",
        semantic_analysis=True,
        max_repairs=1,
        max_rows=10,
    )

    assert result.status == "ok"
    assert result.rows == ((1,),)
    assert provider.calls == 2
    assert tuple(stage.name for stage in result.stages) == (
        "semantic_analysis", "sql", "verify", "execute",
    )
