import json

from semantic_sql.providers import DemoProvider, OllamaProvider


def test_ollama_provider_posts_text_chat_request(monkeypatch):
    seen = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def read(self): return json.dumps({"message": {"content": "SELECT 1"}}).encode()

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data)
        seen["timeout"] = timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = OllamaProvider(model="qwen3.5:4b", base_url="http://localhost:11434", timeout_s=12)
    assert provider.complete_text(system="system", user="user") == "SELECT 1"
    assert seen["url"].endswith("/api/chat")
    assert seen["body"]["model"] == "qwen3.5:4b"
    assert seen["body"]["stream"] is False
    assert seen["body"]["think"] is False
    assert "format" not in seen["body"]
    assert seen["timeout"] == 12


def test_demo_provider_generates_the_offline_showcase_query():
    sql = DemoProvider().complete_text(
        system="Generate SQL",
        user="Question: Show total sales from Czech Republic in 2024\nSchema context: orders, customers",
    )
    assert "SUM(orders.total_amount)" in sql
    assert "customers.country_code = 'CZE'" in sql
