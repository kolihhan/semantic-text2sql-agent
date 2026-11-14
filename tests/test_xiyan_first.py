"""Ponytail minimum: one real service path, one failure path, one boundary check."""
import json
from pathlib import Path

from semantic_sql.providers import OllamaProvider
from semantic_sql.xiyan_first import XiYanFirstSQLService

DB = Path(__file__).parents[1] / "demo" / "demo.db"


class Stub:
    def __init__(self, *sql):
        self.sql = iter(sql)
        self.calls = 0

    def complete_text(self, *, system, user):
        self.calls += 1
        return next(self.sql)


def test_primary_stays_primary_and_never_calls_fallback():
    primary, fallback = Stub("SELECT COUNT(*) FROM orders"), Stub()
    result = XiYanFirstSQLService(
        DB, xiyan_provider=primary, fallback_provider=fallback
    ).ask("How many orders?")
    assert result.status == "ok" and not result.truncated
    assert result.stages[0].summary == "xiyan_primary_valid"
    assert (primary.calls, fallback.calls) == (1, 0)


def test_preflight_failure_uses_existing_guarded_qwen_repair():
    primary = Stub("SELECT nonexistent FROM orders")
    fallback = Stub("SELECT also_nonexistent FROM orders", "SELECT COUNT(*) FROM orders")
    result = XiYanFirstSQLService(
        DB, xiyan_provider=primary, fallback_provider=fallback, fallback_repairs=1
    ).ask("How many orders?")
    assert result.status == "ok"
    assert result.candidate.sql == "SELECT COUNT(*) FROM orders"
    assert result.stages[0].summary == "qwen_fallback:verification_failed"
    assert (primary.calls, fallback.calls) == (1, 2)


def test_valid_but_wrong_sql_does_not_get_overruled():
    primary, fallback = Stub("SELECT MAX(total_amount) FROM orders"), Stub()
    result = XiYanFirstSQLService(
        DB, xiyan_provider=primary, fallback_provider=fallback
    ).ask("How many orders?")
    assert result.status == "ok"
    assert fallback.calls == 0


def test_sampling_options_are_opt_in(monkeypatch):
    sent = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def read(self): return b'{"message":{"content":"SELECT 1"}}'
    def request(req, timeout):
        sent.append(json.loads(req.data))
        return Response()
    monkeypatch.setattr("urllib.request.urlopen", request)
    assert OllamaProvider().complete_text(system="", user="x") == "SELECT 1"
    assert "options" not in sent[0]
    provider = OllamaProvider(temperature=0, seed=42, num_predict=1024)
    assert provider.complete_text(system="", user="x") == "SELECT 1"
    assert sent[1]["options"] == {"temperature": 0, "seed": 42, "num_predict": 1024}
