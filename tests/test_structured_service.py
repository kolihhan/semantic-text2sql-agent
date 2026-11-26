"""Product integration: JSON generation, bounded recoveries, legacy compatibility."""
import json
from pathlib import Path

from semantic_sql.agent import SemanticSQLService
from semantic_sql.providers import DemoProvider, OllamaProvider

DB = Path(__file__).resolve().parents[1] / "demo" / "demo.db"


class JSONProvider:
    model = "qwen3.5:9b-q4_K_M"

    def __init__(self, *answers):
        self.answers = iter(answers)
        self.messages = []

    def complete_messages(self, *, messages, response_format, max_tokens):
        self.messages.append((messages, response_format, max_tokens))
        return {"text": json.dumps(next(self.answers)), "input_tokens": 40,
                "output_tokens": 20, "wall_s": 0.1, "done_reason": "stop"}


def query(sql):
    return {"response_type": "query", "sql": sql, "message": None}


def refusal():
    return {"response_type": "unsupported", "sql": None,
            "message": "No matching rows in the examples"}


def test_local_service_uses_structured_mode_by_default():
    provider = JSONProvider(query("SELECT COUNT(*) FROM orders"))
    result = SemanticSQLService(DB, provider=provider).ask("How many orders?")
    assert result.status == "ok"
    assert result.rows and result.rows[0][0] > 0
    assert result.candidate.sql == "SELECT COUNT(*) FROM orders"
    assert len(provider.messages) == 1
    msgs, schema, maximum = provider.messages[0]
    assert schema["json_schema"]["schema"]["properties"]["response_type"]
    assert maximum == 1024
    assert "Projection and ordering" in msgs[0]["content"]


def test_local_service_recovers_unnecessary_refusal():
    provider = JSONProvider(refusal(), query("SELECT COUNT(*) FROM orders"))
    result = SemanticSQLService(DB, provider=provider).ask("How many orders?")
    assert result.status == "ok"
    assert len(provider.messages) == 2
    assert "answerable" in provider.messages[1][0][-1]["content"]


def test_local_service_rechecks_empty_results_but_keeps_readonly():
    provider = JSONProvider(
        query("SELECT id FROM orders WHERE 1=0"),
        query("SELECT id FROM orders LIMIT 1"),
    )
    result = SemanticSQLService(DB, provider=provider).ask("List one order")
    assert result.status == "ok"
    assert result.rows
    assert result.candidate.sql == "SELECT id FROM orders LIMIT 1"
    assert len(provider.messages) == 2


def test_local_service_refuses_unsafe_sql_without_executing():
    provider = JSONProvider(query("DELETE FROM orders"))
    result = SemanticSQLService(DB, provider=provider).ask("delete orders")
    assert result.status == "verification_failed"
    assert not result.rows
    assert len(provider.messages) == 1


def test_offline_provider_stays_on_legacy_graph():
    result = SemanticSQLService(DB, provider=DemoProvider()).ask(
        "Show total sales from Czech Republic in 2024"
    )
    assert result.status == "ok"
    assert [stage.name for stage in result.stages] == ["sql", "verify", "execute"]


def test_ollama_structured_request_passes_json_schema_and_history(monkeypatch):
    captured = {}
    class Response:
        def __enter__(self): return self
        def __exit__(self, *exc): return None
        def read(self):
            return json.dumps({"message": {"content": json.dumps(query("SELECT 1"))},
                               "prompt_eval_count": 25, "eval_count": 8,
                               "done_reason": "stop"}).encode()

    def fake_urlopen(req, timeout):
        captured["payload"] = json.loads(req.data)
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = OllamaProvider(model="qwen3.5:9b-q4_K_M", temperature=0.0, seed=42)
    messages = [{"role": "system", "content": "SQL"},
                {"role": "user", "content": "query"},
                {"role": "assistant", "content": "bad query"},
                {"role": "user", "content": "repair"}]
    schema = {"type": "json_schema", "json_schema": {"schema": {
        "type": "object", "properties": {"sql": {"type": "string"}}}}}
    result = provider.complete_messages(messages=messages,response_format=schema,max_tokens=1024)
    assert result["text"] == json.dumps(query("SELECT 1"))
    assert result["input_tokens"] == 25
    assert result["output_tokens"] == 8
    assert captured["payload"]["messages"] == messages
    assert captured["payload"]["format"] == schema["json_schema"]["schema"]
    assert captured["payload"]["options"]["num_predict"] == 1024
    assert captured["payload"]["think"] is False


def test_structured_mode_fails_closed_when_provider_cannot_support_it():
    try:
        SemanticSQLService(DB, provider=DemoProvider(), mode="structured")
    except ValueError as exc:
        assert "structured" in str(exc).lower()
    else:
        raise AssertionError("No silent fallback to SQL-only")
