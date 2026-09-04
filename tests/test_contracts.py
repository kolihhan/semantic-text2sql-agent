from dataclasses import fields

from semantic_sql.contracts import AgentResult, SQLCandidate, StageRecord, VerificationResult


def test_agent_result_exposes_only_current_runtime_fields():
    names = {field.name for field in fields(AgentResult)}
    assert names == {"status", "question", "candidate", "verification", "rows", "columns", "truncated", "message", "stages"}
    assert {field.name for field in fields(SQLCandidate)} == {"sql", "attempt"}
    assert {field.name for field in fields(VerificationResult)} == {"ok", "issues"}
    assert {field.name for field in fields(StageRecord)} == {"name", "summary"}
