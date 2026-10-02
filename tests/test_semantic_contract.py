from __future__ import annotations

import json

from semantic_sql.semantic_contract import (
    ContractExtraction,
    extract_semantic_contract,
    parse_semantic_contract,
)


class RecordingProvider:
    def __init__(self, response: str) -> None:
        self.response = response
        self.system = ""
        self.user = ""
        self.calls = 0

    def complete_text(self, *, system: str, user: str) -> str:
        self.calls += 1
        self.system = system
        self.user = user
        return self.response


def _valid_payload(**overrides):
    payload = {
        "confidence": "high",
        "projection": ["constructors.name"],
        "aggregation": {
            "function": "SUM",
            "target": "results.points",
            "entity_grain": "constructor",
            "distinct": False,
        },
        "group_by": ["constructor"],
        "ranking": {
            "metric": "SUM(results.points)",
            "direction": "DESC",
            "top_k": 1,
            "ties": "single",
        },
        "filters": [
            {"field": "races.year", "op": "BETWEEN", "value": [1980, 2010]}
        ],
        "join_entities": ["results", "races", "constructors"],
    }
    payload.update(overrides)
    return payload


def test_parse_semantic_contract_accepts_typed_high_confidence_contract() -> None:
    result = parse_semantic_contract(json.dumps(_valid_payload()))
    assert result.status == "ok"
    assert result.contract is not None
    assert result.contract.confidence == "high"
    assert result.contract.projection == ("constructors.name",)
    assert result.contract.group_by == ("constructor",)
    assert result.contract.join_entities == ("results", "races", "constructors")
    assert result.contract.aggregation is not None
    assert result.contract.aggregation.function == "SUM"
    assert result.contract.aggregation.distinct is False
    assert result.contract.ranking is not None
    assert result.contract.ranking.direction == "DESC"
    assert result.contract.ranking.top_k == 1


def test_parse_semantic_contract_normalizes_invalid_enum_fields_to_unresolved() -> None:
    payload = _valid_payload()
    payload["aggregation"]["function"] = "MEDIAN"
    payload["ranking"]["direction"] = "SIDEWAYS"
    payload["ranking"]["ties"] = "sometimes"
    result = parse_semantic_contract(json.dumps(payload))
    assert result.status == "ok"
    assert result.contract is not None
    assert result.contract.aggregation is not None
    assert result.contract.aggregation.function is None
    assert result.contract.ranking is not None
    assert result.contract.ranking.direction is None
    assert result.contract.ranking.ties == "unspecified"


def test_parse_semantic_contract_fails_open_on_malformed_json() -> None:
    result = parse_semantic_contract("{not-json")
    assert result == ContractExtraction(status="skipped", contract=None, reason="malformed_json")


def test_extract_contract_skips_low_confidence_contract() -> None:
    provider = RecordingProvider(json.dumps(_valid_payload(confidence="low")))
    result = extract_semantic_contract(
        provider=provider,
        question="Which constructor scored the most total points?",
        schema_context="constructors(id, name)\nresults(constructorId, points)",
    )
    assert result.status == "skipped"
    assert result.contract is None
    assert result.reason == "low_confidence"


def test_extract_contract_prompt_never_contains_candidate_sql() -> None:
    provider = RecordingProvider(json.dumps(_valid_payload()))
    candidate = "SELECT definitely_should_never_appear FROM secret_candidate"
    result = extract_semantic_contract(
        provider=provider,
        question="Which constructor scored the most total points?",
        schema_context="constructors(id, name)\nresults(constructorId, points)",
        evidence="points are stored in results.points",
    )
    assert result.status == "ok"
    assert provider.calls == 1
    assert candidate not in provider.user
    assert "Which constructor scored the most total points?" in provider.user
    assert "constructors(id, name)" in provider.user
    assert "points are stored in results.points" in provider.user
    assert "Candidate SQL" not in provider.user


def test_extract_contract_preserves_unresolved_fields_without_hardening_them() -> None:
    payload = _valid_payload(
        projection=["unresolved"],
        filters=[{"field": "unresolved", "op": "=", "value": "unresolved"}],
    )
    provider = RecordingProvider(json.dumps(payload))
    result = extract_semantic_contract(
        provider=provider,
        question="Return the requested value.",
        schema_context="items(id, value)",
    )
    assert result.status == "ok"
    assert result.contract is not None
    assert result.contract.projection == ()
    assert result.contract.filters == ()
