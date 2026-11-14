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


def test_ollama_explicit_seed_temperature_and_token_limit(monkeypatch):
    seen = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def read(self): return json.dumps({"message": {"content": "SELECT 2"}}).encode()

    def fake_urlopen(request, timeout):
        seen["body"] = json.loads(request.data)
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = OllamaProvider(
        model="test", temperature=0, seed=42, num_predict=1024,
    )
    assert provider.complete_text(system="", user="q") == "SELECT 2"
    assert seen["body"]["options"] == {
        "temperature": 0, "seed": 42, "num_predict": 1024,
    }


def test_ollama_default_sampling_options_remain_absent(monkeypatch):
    seen = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def read(self): return json.dumps({"message": {"content": "SELECT 3"}}).encode()

    def fake_urlopen(request, timeout):
        seen["body"] = json.loads(request.data)
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    OllamaProvider().complete_text(system="", user="q")
    assert "options" not in seen["body"]
