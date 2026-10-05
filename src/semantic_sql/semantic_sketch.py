from __future__ import annotations

from dataclasses import asdict, dataclass
import json

from .providers import ModelProvider


@dataclass(frozen=True)
class SemanticSketch:
    outputs: tuple[str, ...] = ()
    filters: tuple[str, ...] = ()
    relations: tuple[str, ...] = ()
    aggregation: str = ""
    grain: str = ""
    group_by: tuple[str, ...] = ()
    ordering: str = ""
    limit: str = ""

    def as_prompt(self) -> str:
        payload = asdict(self)
        for key in ("outputs", "filters", "relations", "group_by"):
            payload[key] = list(payload[key])
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _items(value: object) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(text for item in value if (text := str(item).strip()))
    text = _text(value)
    return (text,) if text else ()


def analyze_semantics(
    question: str,
    schema_context: str,
    provider: ModelProvider,
    *,
    external_evidence: str | None = None,
) -> SemanticSketch:
    user = f"Question:\n{question}\n\nSchema context:\n{schema_context}"
    if external_evidence is not None:
        user += f"\n\nExternal evidence:\n{external_evidence}"

    response = provider.complete_text(
        system=(
            "Analyze the user's intended database answer before SQL generation. Do not write SQL. "
            "Use only the question, supplied schema context, and external evidence; do not invent facts. "
            "Return JSON only with keys outputs, filters, relations, aggregation, grain, group_by, ordering, limit. "
            "outputs, filters, relations, and group_by are arrays of short strings; all other values are strings. "
            "Use an empty array or empty string when a field does not apply."
        ),
        user=user,
    )

    try:
        payload = json.loads(response.strip())
        if not isinstance(payload, dict):
            raise ValueError("semantic sketch must be a JSON object")
        return SemanticSketch(
            outputs=_items(payload.get("outputs")),
            filters=_items(payload.get("filters")),
            relations=_items(payload.get("relations")),
            aggregation=_text(payload.get("aggregation")),
            grain=_text(payload.get("grain")),
            group_by=_items(payload.get("group_by")),
            ordering=_text(payload.get("ordering")),
            limit=_text(payload.get("limit")),
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return SemanticSketch()
