"""Token/cost instrumentation must preserve the fixed-Qwen inference contract.

No network, model, benchmark, holdout, or physical DB is touched by these tests.
"""
from __future__ import annotations

import json
import urllib.request

import pytest

from semantic_sql.providers import OllamaProvider


class MockReply:
    def __init__(self, payload: dict):
        self.payload=payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def _fake_urlopen(monkeypatch, responses):
    requests=[]
    iterator=iter(responses)

    def fake(req, timeout):
        assert isinstance(req, urllib.request.Request)
        assert req.get_method()=="POST"
        payload=json.loads(req.data.decode("utf-8"))
        assert payload["stream"] is False
        assert payload["think"] is False
        requests.append((payload,timeout))
        return MockReply(next(iterator))

    monkeypatch.setattr(urllib.request,"urlopen",fake)
    return requests


def test_default_no_usage_capture_output_and_request_unchanged(monkeypatch):
    requests=_fake_urlopen(monkeypatch,[{
        "message":{"content":"SELECT COUNT(*) FROM orders"},
        "eval_count":39,"prompt_eval_count":102,
    }])
    provider=OllamaProvider(model="qwen3.5:9b-q4_K_M",
                            temperature=0.0,seed=42,num_predict=512)
    assert provider.complete_text(system="SQL only",user="count orders") == "SELECT COUNT(*) FROM orders"
    assert requests[0][0]["model"]=="qwen3.5:9b-q4_K_M"
    assert requests[0][0]["options"]=={"temperature":0.0,"seed":42,"num_predict":512}
    assert provider.usage_records==[]
    assert provider.usage_totals["eval_count"] is None


def test_opt_in_real_ollama_tokens_and_nanosecond_durations(monkeypatch):
    payloads=[
        {"message":{"content":"SELECT 1"},
         "prompt_eval_count":100,"eval_count":20,
         "total_duration":1000000000,"load_duration":100,
         "prompt_eval_duration":300000000,"eval_duration":600000000},
        {"message":{"content":"SELECT 2"},
         "prompt_eval_count":160,"eval_count":10,
         "total_duration":1400000000,"load_duration":200,
         "prompt_eval_duration":500000000,"eval_duration":800000000},
    ]
    _fake_urlopen(monkeypatch,payloads)
    provider=OllamaProvider(capture_usage=True)
    assert provider.complete_text(system="SQL",user="first")=="SELECT 1"
    assert provider.complete_text(system="SQL",user="second")=="SELECT 2"
    assert len(provider.usage_records)==2
    assert provider.usage_totals=={
        "prompt_eval_count":260,"eval_count":30,
        "total_duration":2400000000,"load_duration":300,
        "prompt_eval_duration":800000000,"eval_duration":1400000000,
    }


def test_missing_server_counters_remain_unknown_not_fabricated(monkeypatch):
    _fake_urlopen(monkeypatch,[
        {"message":{"content":"SELECT 1"},
         "prompt_eval_count":10,"eval_count":8,"total_duration":1000},
        {"message":{"content":"SELECT 2"},
         "prompt_eval_count":11,"eval_count":None,"total_duration":1200},
    ])
    provider=OllamaProvider(capture_usage=True)
    provider.complete_text(system="SQL",user="first")
    provider.complete_text(system="SQL",user="second")
    assert provider.usage_totals["prompt_eval_count"]==21
    assert provider.usage_totals["eval_count"] is None
    assert provider.usage_totals["total_duration"]==2200
    assert provider.usage_totals["load_duration"] is None


def test_invalid_boolean_or_negative_counter_does_not_poison_model_output(monkeypatch):
    _fake_urlopen(monkeypatch,[{
        "message":{"content":"SELECT 3"},
        "prompt_eval_count":True,"eval_count":-1,"total_duration":1.2,
    }])
    provider=OllamaProvider(capture_usage=True)
    assert provider.complete_text(system="SQL",user="third")=="SELECT 3"
    assert provider.usage_totals["prompt_eval_count"] is None
    assert provider.usage_totals["eval_count"] is None
    assert provider.usage_totals["total_duration"] is None


def test_malformed_response_retains_original_error_and_does_not_log_as_success(monkeypatch):
    _fake_urlopen(monkeypatch,[{"message":{},"prompt_eval_count":8}])
    provider=OllamaProvider(capture_usage=True)
    with pytest.raises(KeyError):
        provider.complete_text(system="SQL",user="missing")
    assert provider.usage_records==[]
