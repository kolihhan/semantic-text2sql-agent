from pathlib import Path

from evaluation.xiyan_qwen_factorial_fresh40 import prompt_for
from semantic_sql.xiyan_first import XiYanFirstSQLService, xiyan_native_prompt

DB = Path(__file__).parents[1] / "demo" / "demo.db"


class StubProvider:
    def __init__(self, *outputs):
        self.outputs = iter(outputs)
        self.requests = []

    def complete_text(self, *, system: str, user: str) -> str:
        self.requests.append((system, user))
        answer = next(self.outputs)
        if isinstance(answer, Exception):
            raise answer
        return answer


def service(primary, fallback, **kwargs):
    return XiYanFirstSQLService(
        DB, xiyan_provider=primary, fallback_provider=fallback, **kwargs)


def test_prompt_is_byte_exact_match_for_evaluated_xiyan_prompt():
    expected = prompt_for(
        condition="xiyan", question="How many?", schema_context="TABLE X",
        evidence="ref",
    )
    actual = xiyan_native_prompt(
        question="How many?", schema_context="TABLE X", evidence="ref",
    )
    assert actual == expected


def test_primary_valid_executes_once_and_never_calls_qwen():
    first = StubProvider("SELECT COUNT(*) FROM orders")
    second = StubProvider()
    result = service(first, second).ask("How many orders?")
    assert result.status == "ok"
    assert result.candidate.sql == "SELECT COUNT(*) FROM orders"
    assert result.stages[0].summary == "xiyan_primary_valid"
    assert len(first.requests) == 1
    assert len(second.requests) == 0


def test_preflight_rejection_runs_qwen_guarded_with_repair():
    first = StubProvider("SELECT missing FROM orders")
    second = StubProvider("SELECT bad FROM orders", "SELECT COUNT(*) FROM orders")
    result = service(first, second, fallback_repairs=1).ask("How many orders?")
    assert result.status == "ok"
    assert result.candidate.sql == "SELECT COUNT(*) FROM orders"
    assert result.candidate.attempt == 1
    assert result.stages[0].summary == "qwen_fallback:verification_failed"
    assert len(first.requests) == 1
    assert len(second.requests) == 2


def test_a_valid_but_semantically_wrong_query_is_not_automatically_corrected():
    first = StubProvider("SELECT MAX(total_amount) FROM orders")
    second = StubProvider("SELECT COUNT(*) FROM orders")
    result = service(first, second).ask("How many orders?")
    assert result.status == "ok"
    assert result.candidate.sql == "SELECT MAX(total_amount) FROM orders"
    assert len(second.requests) == 0


def test_truncation_falls_back_instead_of_returning_incomplete_rows():
    first = StubProvider("SELECT id FROM orders ORDER BY id")
    second = StubProvider("SELECT COUNT(*) FROM orders")
    result = service(first, second, max_rows=1).ask("How many orders?")
    assert result.status == "ok"
    assert not result.truncated
    assert result.stages[0].summary == "qwen_fallback:primary_truncated"
    assert len(second.requests) == 1


def test_primary_model_transport_exception_fails_over_with_safe_trace():
    first = StubProvider(TimeoutError("PRIVATE SQL DATA"))
    second = StubProvider("SELECT COUNT(*) FROM orders")
    result = service(first, second).ask("How many orders?")
    assert result.status == "ok"
    assert result.stages[0].summary == "qwen_fallback:primary_exception:TimeoutError"
    assert "PRIVATE SQL DATA" not in str(result.stages)
    assert len(second.requests) == 1


def test_invalid_configuration_rejected():
    import pytest
    with pytest.raises(ValueError):
        service(StubProvider(), StubProvider(), max_rows=0)
