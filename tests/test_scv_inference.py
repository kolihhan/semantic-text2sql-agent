from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import semantic_sql.inference as inference
from semantic_sql.contract_verifier import SCVVerification, SCVViolation
from semantic_sql.semantic_contract import ContractExtraction, SemanticContract
from semantic_sql.sql_semantics import SQLSemantics, SQLSemanticsParse


class RecordingProvider:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete_text(self, *, system: str, user: str) -> str:
        self.calls.append((system, user))
        if not self.responses:
            raise AssertionError("unexpected model call")
        return self.responses.pop(0)


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "catalog.sqlite"
    with sqlite3.connect(path) as con:
        con.execute(
            'CREATE TABLE schools (id INTEGER PRIMARY KEY, "County Name" TEXT NOT NULL)'
        )
        con.execute('INSERT INTO schools VALUES (1, "Alpha")')
    return path


def _ok_contract() -> ContractExtraction:
    return ContractExtraction(
        status="ok",
        contract=SemanticContract(confidence="high"),
    )


def _patch_contract(
    monkeypatch: pytest.MonkeyPatch,
    extraction: ContractExtraction | None = None,
) -> None:
    monkeypatch.setattr(
        inference,
        "extract_semantic_contract",
        lambda **kwargs: extraction or _ok_contract(),
        raising=False,
    )


def _patch_parse(
    monkeypatch: pytest.MonkeyPatch,
    result: SQLSemanticsParse | None = None,
) -> None:
    monkeypatch.setattr(
        inference,
        "parse_sql_semantics",
        lambda sql: result or SQLSemanticsParse(status="ok", semantics=SQLSemantics()),
        raising=False,
    )


def test_scv_disabled_preserves_historical_guarded_graph_and_call_count(database: Path) -> None:
    provider = RecordingProvider()
    sql = 'SELECT "County Name" FROM schools'
    result = inference.run_guarded(
        database=database,
        provider=provider,
        question="Return county name.",
        schema_context='schools(id, "County Name")',
        initial_candidate=inference.SQLCandidate(sql=sql),
        max_repairs=0,
        semantic_contract_verification=False,
    )
    assert result.status == "ok"
    assert result.candidate is not None and result.candidate.sql == sql
    assert tuple(stage.name for stage in result.stages) == ("sql", "verify", "execute")
    assert provider.calls == []


def test_scv_contract_skip_executes_original_candidate_unchanged(
    database: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_contract(
        monkeypatch,
        ContractExtraction(status="skipped", contract=None, reason="low_confidence"),
    )
    provider = RecordingProvider()
    sql = 'SELECT "County Name" FROM schools'
    result = inference.run_guarded(
        database=database,
        provider=provider,
        question="Return county name.",
        schema_context='schools(id, "County Name")',
        initial_candidate=inference.SQLCandidate(sql=sql),
        max_repairs=0,
        semantic_contract_verification=True,
    )
    assert result.status == "ok"
    assert result.candidate is not None and result.candidate.sql == sql
    assert "SCV_SKIPPED_CONTRACT" in " ".join(stage.summary for stage in result.stages)
    assert "semantic_repair" not in tuple(stage.name for stage in result.stages)


def test_scv_ast_skip_executes_original_candidate_unchanged(
    database: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_contract(monkeypatch)
    _patch_parse(
        monkeypatch,
        SQLSemanticsParse(status="skipped", semantics=None, reason="parse_error"),
    )
    provider = RecordingProvider()
    sql = 'SELECT "County Name" FROM schools'
    result = inference.run_guarded(
        database=database,
        provider=provider,
        question="Return county name.",
        schema_context='schools(id, "County Name")',
        initial_candidate=inference.SQLCandidate(sql=sql),
        max_repairs=0,
        semantic_contract_verification=True,
    )
    assert result.status == "ok"
    assert result.candidate is not None and result.candidate.sql == sql
    assert "SCV_SKIPPED_AST" in " ".join(stage.summary for stage in result.stages)


def test_scv_pass_executes_without_semantic_repair(
    database: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_contract(monkeypatch)
    _patch_parse(monkeypatch)
    monkeypatch.setattr(
        inference,
        "verify_semantic_contract",
        lambda *args, **kwargs: SCVVerification(status="pass"),
        raising=False,
    )
    provider = RecordingProvider()
    result = inference.run_guarded(
        database=database,
        provider=provider,
        question="Return county name.",
        schema_context='schools(id, "County Name")',
        initial_candidate=inference.SQLCandidate(sql='SELECT "County Name" FROM schools'),
        max_repairs=0,
        semantic_contract_verification=True,
    )
    assert result.status == "ok"
    names = tuple(stage.name for stage in result.stages)
    assert names == ("sql", "verify", "semantic_contract", "semantic_verify", "execute")
    assert provider.calls == []


def test_scv_violation_triggers_one_targeted_repair(
    database: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_contract(monkeypatch)
    _patch_parse(monkeypatch)
    calls = {"verify": 0}

    violation = SCVViolation(
        code="PROJECTION_MISMATCH",
        expected="schools.County Name",
        actual="schools.id",
        evidence="requested county name is absent",
    )

    def fake_verify(*args, **kwargs):
        calls["verify"] += 1
        if calls["verify"] == 1:
            return SCVVerification(status="violation", violations=(violation,))
        return SCVVerification(status="pass")

    monkeypatch.setattr(inference, "verify_semantic_contract", fake_verify, raising=False)
    provider = RecordingProvider('SELECT "County Name" FROM schools')
    result = inference.run_guarded(
        database=database,
        provider=provider,
        question="Return county name.",
        schema_context='schools(id, "County Name")',
        initial_candidate=inference.SQLCandidate(sql="SELECT id FROM schools"),
        max_repairs=0,
        semantic_contract_verification=True,
        semantic_repair_budget=1,
    )
    assert result.status == "ok"
    assert result.candidate is not None
    assert result.candidate.sql == 'SELECT "County Name" FROM schools'
    names = tuple(stage.name for stage in result.stages)
    assert names.count("semantic_repair") == 1
    assert names.count("semantic_verify") == 2
    assert len(provider.calls) == 1
    assert "PROJECTION_MISMATCH" in provider.calls[0][1]
    assert "requested county name is absent" in provider.calls[0][1]


def test_scv_repair_failure_never_triggers_second_semantic_repair(
    database: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_contract(monkeypatch)
    _patch_parse(monkeypatch)
    violation = SCVViolation(
        code="PROJECTION_MISMATCH",
        expected="schools.County Name",
        actual="schools.id",
        evidence="requested county name is absent",
    )
    monkeypatch.setattr(
        inference,
        "verify_semantic_contract",
        lambda *args, **kwargs: SCVVerification(
            status="violation", violations=(violation,)
        ),
        raising=False,
    )
    provider = RecordingProvider("SELECT id FROM schools")
    result = inference.run_guarded(
        database=database,
        provider=provider,
        question="Return county name.",
        schema_context='schools(id, "County Name")',
        initial_candidate=inference.SQLCandidate(sql="SELECT id FROM schools"),
        max_repairs=0,
        semantic_contract_verification=True,
        semantic_repair_budget=1,
    )
    assert result.status == "verification_failed"
    assert tuple(stage.name for stage in result.stages).count("semantic_repair") == 1
    assert len(provider.calls) == 1


def test_scv_invalid_repair_is_preflighted_and_not_repaired_again(
    database: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_contract(monkeypatch)
    _patch_parse(monkeypatch)
    violation = SCVViolation(
        code="PROJECTION_MISMATCH",
        expected="schools.County Name",
        actual="schools.id",
        evidence="requested county name is absent",
    )
    monkeypatch.setattr(
        inference,
        "verify_semantic_contract",
        lambda *args, **kwargs: SCVVerification(
            status="violation", violations=(violation,)
        ),
        raising=False,
    )
    provider = RecordingProvider("SELECT missing_column FROM schools")
    result = inference.run_guarded(
        database=database,
        provider=provider,
        question="Return county name.",
        schema_context='schools(id, "County Name")',
        initial_candidate=inference.SQLCandidate(sql="SELECT id FROM schools"),
        max_repairs=2,
        semantic_contract_verification=True,
        semantic_repair_budget=1,
    )
    assert result.status == "verification_failed"
    assert len(provider.calls) == 1
    assert tuple(stage.name for stage in result.stages).count("semantic_repair") == 1


def test_scv_semantic_repair_budget_is_capped_at_one(
    database: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_contract(monkeypatch)
    _patch_parse(monkeypatch)
    violation = SCVViolation(
        code="PROJECTION_MISMATCH",
        expected="schools.County Name",
        actual="schools.id",
        evidence="requested county name is absent",
    )
    monkeypatch.setattr(
        inference,
        "verify_semantic_contract",
        lambda *args, **kwargs: SCVVerification(
            status="violation", violations=(violation,)
        ),
        raising=False,
    )
    provider = RecordingProvider("SELECT id FROM schools")
    result = inference.run_guarded(
        database=database,
        provider=provider,
        question="Return county name.",
        schema_context='schools(id, "County Name")',
        initial_candidate=inference.SQLCandidate(sql="SELECT id FROM schools"),
        max_repairs=0,
        semantic_contract_verification=True,
        semantic_repair_budget=3,
    )
    assert result.status == "verification_failed"
    assert tuple(stage.name for stage in result.stages).count("semantic_repair") == 1
    assert len(provider.calls) == 1
