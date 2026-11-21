"""FreeLLMAPI transport contract: no real tokens, no inference, no benchmark scoring."""
import json

import pytest

from semantic_sql.providers import FreeLLMAPIProvider


def test_freellmapi_uses_explicit_model_and_no_auto_fallback(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): return None
        def read(self):
            return json.dumps({
                "model": "Qwen3.5-9B",
                "_routed_via": {"platform": "ovh", "model": "Qwen3.5-9B"},
                "choices": [{"message": {"content": "SELECT 1"}, "finish_reason": "stop"}],
            }).encode()

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data)
        captured["headers"] = dict(req.header_items())
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    p = FreeLLMAPIProvider(api_key="test-only", timeout_s=15)
    assert p.complete_text(system="system", user="question") == "SELECT 1"
    assert captured["url"] == "http://localhost:3001/v1/chat/completions"
    assert captured["body"]["model"] == "qwen3.5-9b"
    assert captured["body"]["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
    ]
    assert captured["body"]["stream"] is False
    assert captured["body"]["temperature"] == 0
    assert captured["body"]["seed"] == 42
    assert captured["body"]["reasoning_effort"] == "none"
    assert captured["body"]["max_tokens"] == 512
    assert captured["timeout"] == 15
    assert captured["headers"]["Authorization"] == "Bearer test-only"
    assert "auto" not in captured["body"]["model"].lower()


@pytest.mark.parametrize("returned,finish_reason", [
    ("google/gemma-4-31B-it", "stop"),
    ("", "stop"),
    ("Qwen3.5-9B", "length"),
])
def test_freellmapi_rejects_substitution_missing_identity_and_truncation(monkeypatch, returned, finish_reason):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): return None
        def read(self):
            return json.dumps({
                "model": returned,
                "_routed_via": {"platform": "ovh", "model": "Qwen3.5-9B"},
                "choices": [{"message": {"content": "SELECT 1"}, "finish_reason": finish_reason}],
            }).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: Response())
    with pytest.raises(ValueError, match="model identity|incomplete"):
        FreeLLMAPIProvider(api_key="test-only").complete_text(system="s", user="u")


def test_freellmapi_missing_key_fails_before_network(monkeypatch):
    monkeypatch.delenv("FREELLMAPI_API_KEY", raising=False)

    def unexpected_network(*_args, **_kwargs):
        raise AssertionError("must fail before network")

    monkeypatch.setattr("urllib.request.urlopen", unexpected_network)
    with pytest.raises(ValueError, match="FREELLMAPI_API_KEY"):
        FreeLLMAPIProvider().complete_text(system="s", user="u")


def test_freellmapi_refuses_an_unexpected_provider_even_with_matching_model(monkeypatch):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): return None
        def read(self):
            return json.dumps({
                "model": "Qwen3.5-9B",
                "_routed_via": {"platform": "kilo", "model": "Qwen3.5-9B"},
                "choices": [{"message": {"content": "SELECT 1"}, "finish_reason": "stop"}],
            }).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: Response())
    with pytest.raises(ValueError, match="routed provider"):
        FreeLLMAPIProvider(api_key="test-only").complete_text(system="s", user="u")
