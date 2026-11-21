from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from .providers import ModelProvider


ContractConfidence = Literal["high", "medium", "low"]
AggregationFunction = Literal["COUNT", "SUM", "AVG", "MIN", "MAX"]
OrderDirection = Literal["ASC", "DESC"]
TiePolicy = Literal["single", "all", "unspecified"]

_ALLOWED_AGGREGATIONS = {"COUNT", "SUM", "AVG", "MIN", "MAX"}
_ALLOWED_DIRECTIONS = {"ASC", "DESC"}
_ALLOWED_TIES = {"single", "all", "unspecified"}
_ALLOWED_FILTER_OPS = {
    "=", "!=", "<", "<=", ">", ">=", "BETWEEN", "IN", "LIKE", "IS NULL", "IS NOT NULL",
}


@dataclass(frozen=True)
class AggregationSpec:
    function: AggregationFunction | None = None
    target: str | None = None
    entity_grain: str | None = None
    distinct: bool | None = None


@dataclass(frozen=True)
class RankingSpec:
    metric: str | None = None
    direction: OrderDirection | None = None
    top_k: int | None = None
    ties: TiePolicy = "unspecified"


@dataclass(frozen=True)
class FilterSpec:
    field: str
    op: str
    value: Any


@dataclass(frozen=True)
class SemanticContract:
    confidence: ContractConfidence
    projection: tuple[str, ...] = ()
    aggregation: AggregationSpec | None = None
    group_by: tuple[str, ...] = ()
    ranking: RankingSpec | None = None
    filters: tuple[FilterSpec, ...] = ()
    join_entities: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContractExtraction:
    status: Literal["ok", "skipped"]
    contract: SemanticContract | None
    reason: str = ""


def _clean_ref(value: object) -> str | None:
    text = str(value or "").strip()
    if not text or text.casefold() in {"unresolved", "unknown", "null", "none"}:
        return None
    return text


def _clean_refs(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    cleaned: list[str] = []
    for item in value:
        ref = _clean_ref(item)
        if ref is not None:
            cleaned.append(ref)
    return tuple(cleaned)


def _parse_aggregation(value: object) -> AggregationSpec | None:
    if not isinstance(value, dict):
        return None
    raw_function = str(value.get("function") or "").upper()
    function = raw_function if raw_function in _ALLOWED_AGGREGATIONS else None
    distinct = value.get("distinct")
    return AggregationSpec(
        function=function,  # type: ignore[arg-type]
        target=_clean_ref(value.get("target")),
        entity_grain=_clean_ref(value.get("entity_grain")),
        distinct=distinct if isinstance(distinct, bool) else None,
    )


def _parse_ranking(value: object) -> RankingSpec | None:
    if not isinstance(value, dict):
        return None
    raw_direction = str(value.get("direction") or "").upper()
    direction = raw_direction if raw_direction in _ALLOWED_DIRECTIONS else None
    raw_ties = str(value.get("ties") or "unspecified").lower()
    ties = raw_ties if raw_ties in _ALLOWED_TIES else "unspecified"
    top_k = value.get("top_k")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        top_k = None
    metric = _clean_ref(value.get("metric"))
    if metric is None and direction is None and top_k is None and ties == "unspecified":
        return None
    return RankingSpec(
        metric=metric,
        direction=direction,  # type: ignore[arg-type]
        top_k=top_k,
        ties=ties,  # type: ignore[arg-type]
    )


def _parse_filters(value: object) -> tuple[FilterSpec, ...]:
    if not isinstance(value, list):
        return ()
    filters: list[FilterSpec] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        field = _clean_ref(item.get("field"))
        raw_value = item.get("value")
        if field is None or (
            isinstance(raw_value, str)
            and raw_value.strip().casefold() in {"unresolved", "unknown", "null", "none"}
        ):
            continue
        op = str(item.get("op") or "").upper().strip()
        if op not in _ALLOWED_FILTER_OPS:
            continue
        filters.append(FilterSpec(field=field, op=op, value=raw_value))
    return tuple(filters)


def parse_semantic_contract(text: str) -> ContractExtraction:
    try:
        payload = json.loads(text.strip())
    except (TypeError, ValueError, json.JSONDecodeError):
        return ContractExtraction(status="skipped", contract=None, reason="malformed_json")
    if not isinstance(payload, dict):
        return ContractExtraction(status="skipped", contract=None, reason="wrong_shape")

    confidence = str(payload.get("confidence") or "low").lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    if confidence == "low":
        return ContractExtraction(status="skipped", contract=None, reason="low_confidence")

    contract = SemanticContract(
        confidence=confidence,  # type: ignore[arg-type]
        projection=_clean_refs(payload.get("projection")),
        aggregation=_parse_aggregation(payload.get("aggregation")),
        group_by=_clean_refs(payload.get("group_by")),
        ranking=_parse_ranking(payload.get("ranking")),
        filters=_parse_filters(payload.get("filters")),
        join_entities=_clean_refs(payload.get("join_entities")),
    )
    return ContractExtraction(status="ok", contract=contract)


_CONTRACT_SYSTEM = """Extract a compact semantic contract for a text-to-SQL question.
Return JSON only. Do not write SQL and do not inspect any candidate SQL.
Use only the question, evidence, and physical schema supplied by the user.

Schema:
{
  "confidence": "high" | "medium" | "low",
  "projection": ["resolved table.column/entity or unresolved"],
  "aggregation": {
    "function": "COUNT" | "SUM" | "AVG" | "MIN" | "MAX" | null,
    "target": "resolved field/entity or unresolved",
    "entity_grain": "resolved entity or unresolved",
    "distinct": true | false | null
  } | null,
  "group_by": ["resolved field/entity or unresolved"],
  "ranking": {
    "metric": "resolved expression/entity or unresolved",
    "direction": "ASC" | "DESC" | null,
    "top_k": integer | null,
    "ties": "single" | "all" | "unspecified"
  } | null,
  "filters": [
    {"field": "resolved field or unresolved", "op": "=", "value": "literal or unresolved"}
  ],
  "join_entities": ["resolved physical table/entity"]
}

If a field cannot be grounded confidently from the supplied schema, mark it unresolved rather than guessing.
Do not include reasoning, prose, markdown, or candidate SQL."""


def extract_semantic_contract(
    *,
    provider: ModelProvider,
    question: str,
    schema_context: str,
    evidence: str | None = None,
) -> ContractExtraction:
    prompt = (
        f"Question:\n{question}\n\n"
        f"Evidence:\n{evidence or '(none)'}\n\n"
        f"Physical schema context:\n{schema_context}\n"
    )
    try:
        response = provider.complete_text(system=_CONTRACT_SYSTEM, user=prompt)
    except Exception:
        return ContractExtraction(status="skipped", contract=None, reason="extractor_error")
    return parse_semantic_contract(response)
