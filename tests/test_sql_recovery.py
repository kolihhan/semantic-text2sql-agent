import json
import sqlite3

from semantic_sql.agent import SemanticSQLService
from semantic_sql.providers import OllamaProvider


class SQLResponses:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = 0

    def complete_sql(self, *, system, user):
        self.calls += 1
        return json.dumps(next(self.responses))

    def complete_text(self, *, system, user):
        raise AssertionError("SQL agent bypassed structured output")


def query(sql):
    return {"response_type": "query", "sql": sql, "message": None}


def refusal():
    return {"response_type": "unsupported", "sql": None, "message": "missing sample value"}


def database(tmp_path):
    p = tmp_path / "t.sqlite"
    with sqlite3.connect(p) as con:
        con.execute("CREATE TABLE items(id INTEGER)")
        con.execute("INSERT INTO items VALUES (1)")
    return p


def test_nonquery_response_recovers_through_same_guarded_path(tmp_path):
    p = SQLResponses(refusal(), query("SELECT id FROM items"))
    result = SemanticSQLService(database(tmp_path), provider=p, max_repairs=1).ask("List item ids")
    assert result.status == "ok"
    assert result.rows == ((1,),)
    assert result.candidate.attempt == 1
    assert p.calls == 2


def test_empty_result_recheck_only_accepts_verified_nonempty_query(tmp_path):
    p = SQLResponses(query("SELECT id FROM items WHERE id = -1"), query("SELECT id FROM items"))
    result = SemanticSQLService(database(tmp_path), provider=p, max_repairs=2).ask("List item ids")
    assert result.status == "ok"
    assert result.rows == ((1,),)
    assert result.candidate.attempt == 1
    assert [s.name for s in result.stages][-1] == "recheck"
    assert p.calls == 2


def test_empty_recheck_rejects_unsafe_change_and_preserves_original(tmp_path):
    db = database(tmp_path)
    p = SQLResponses(query("SELECT id FROM items WHERE id = -1"), query("DROP TABLE items"))
    result = SemanticSQLService(db, provider=p).ask("List item ids")
    assert result.status == "ok"
    assert result.rows == ()
    assert result.candidate.sql == "SELECT id FROM items WHERE id = -1"
    assert result.candidate.attempt == 0
    assert result.stages[-1].summary == "accepted=False"
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 1


def test_no_extra_call_when_query_succeeds_or_budget_exhausted(tmp_path):
    db = database(tmp_path)
    ready = SQLResponses(query("SELECT id FROM items"))
    result = SemanticSQLService(db, provider=ready).ask("List item ids")
    assert result.rows == ((1,),)
    assert ready.calls == 1
    exhausted = SQLResponses(query("SELECT id FROM items WHERE id=-1"))
    result = SemanticSQLService(db, provider=exhausted, max_repairs=0).ask("List item ids")
    assert result.status == "ok" and result.rows == ()
    assert exhausted.calls == 1


def test_structured_provider_sends_ollama_json_schema(monkeypatch):
    seen = {}
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def read(self): return json.dumps({"message": {"content": json.dumps(query("SELECT 1"))}}).encode()
    def fake_urlopen(req, timeout):
        seen.update(json.loads(req.data))
        return Response()
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = OllamaProvider(model="qwen3.5:9b-q4_K_M")
    assert json.loads(provider.complete_sql(system="SQL", user="SELECT 1"))["sql"] == "SELECT 1"
    assert seen["format"]["required"] == ["response_type", "sql", "message"]
    assert seen["format"]["additionalProperties"] is False
    assert seen["think"] is False
