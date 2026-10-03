from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, median
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "evaluation"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bird_loader import load_bird_json
from grounding_metrics import score_grounding_pack, score_selected_tables
from schema_context import _description_rows, select_schema_context
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.direct import _strip_fence
from semantic_sql.execution import SQLExecutionError, execute_readonly
from semantic_sql.grounding import (
    QuestionDecomposition,
    build_grounding_pack,
    build_value_index,
    heuristic_decomposition,
)
from semantic_sql.providers import OllamaProvider


def resolve_database(database_root: Path, db_id: str) -> Path:
    direct = database_root / db_id / f"{db_id}.sqlite"
    if direct.is_file():
        return direct
    matches = [
        path
        for path in (database_root / db_id).glob("*.sqlite")
        if "__MACOSX" not in path.parts
    ]
    if len(matches) == 1:
        return matches[0]
    matches = [
        path
        for path in database_root.rglob("*.sqlite")
        if path.stem == db_id and "__MACOSX" not in path.parts
    ]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"SQLite database not found for {db_id}")


def score_query(
    database: Path,
    predicted_sql: str | None,
    gold_sql: str,
    *,
    max_rows: int = 50_000,
) -> dict[str, bool]:
    if not predicted_sql:
        return {
            "execution_success": False,
            "frozen_execution_match": False,
            "truncated": False,
        }
    try:
        predicted = execute_readonly(database, predicted_sql, max_rows=max_rows)
    except SQLExecutionError:
        return {
            "execution_success": False,
            "frozen_execution_match": False,
            "truncated": False,
        }
    try:
        gold = execute_readonly(database, gold_sql, max_rows=max_rows)
    except SQLExecutionError as exc:
        raise RuntimeError("gold SQL failed to execute") from exc

    truncated = bool(predicted.truncated or gold.truncated)
    return {
        "execution_success": True,
        "frozen_execution_match": (
            False if truncated else set(predicted.rows) == set(gold.rows)
        ),
        "truncated": truncated,
    }


def generate_xiyan(
    *,
    question: str,
    schema_context: str,
    evidence: str | None,
    provider: OllamaProvider,
) -> str:
    prompt = f"""你是一名SQLite专家，现在需要阅读并理解下面的【数据库schema】描述，以及可能用到的【参考信息】，并运用SQLite知识生成sql语句回答【用户问题】。
【用户问题】
{question}

【数据库schema】
{schema_context}

【参考信息】
{evidence or ''}

【用户问题】
{question}

只返回一条只读SQLite SQL，不要解释。
```sql"""
    return _strip_fence(provider.complete_text(system="", user=prompt))


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _strip_fence(stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("decomposition response has no JSON object")
    payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("decomposition response must be a JSON object")
    return payload


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " | ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def _as_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, list):
        return tuple(str(item).strip() for item in value if str(item).strip())
    text = str(value).strip()
    return (text,) if text else ()


def decompose_xiyan(
    *,
    question: str,
    provider: OllamaProvider,
) -> tuple[QuestionDecomposition, str, bool]:
    prompt = f"""把下面的数据库问题拆成只用于schema/value检索的结构。不要写SQL，不要猜表名或列名，不要添加问题中没有的实体或数值。

问题：
{question}

只返回JSON：
{{
  "entity": "主要实体/关系短语",
  "metric": "count/sum/average/ratio等；没有则空字符串",
  "filters": ["过滤条件短语"],
  "ranking": "most/least/top/first/last等；没有则空字符串",
  "time": ["时间条件短语"]
}}"""
    response = provider.complete_text(
        system=(
            "Decompose a database question for retrieval only. Return JSON only. "
            "Never invent schema identifiers or values."
        ),
        user=prompt,
    )
    try:
        payload = _extract_json_object(response)
        decomposition = QuestionDecomposition(
            entity=_as_text(payload.get("entity")),
            metric=_as_text(payload.get("metric")),
            filters=_as_tuple(payload.get("filters")),
            ranking=_as_text(payload.get("ranking")),
            time=_as_tuple(payload.get("time")),
            raw_question=question,
        )
        if not any(
            (
                decomposition.entity,
                decomposition.metric,
                decomposition.filters,
                decomposition.ranking,
                decomposition.time,
            )
        ):
            raise ValueError("empty structured decomposition")
        return decomposition, response, False
    except (TypeError, ValueError, json.JSONDecodeError):
        return heuristic_decomposition(question), response, True


def load_baseline_sql(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("baseline results must contain a cases list")

    result: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        case_id = str(row.get("case_id") or "")
        sql = row.get("xiyan_sql") or row.get("baseline_sql") or row.get("direct_sql")
        if case_id and isinstance(sql, str) and sql.strip():
            result[case_id] = sql.strip()
    return result


def _mean_metric(rows: list[dict[str, object]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return mean(values) if values else None


def aggregate_case_rows(
    rows: list[dict[str, object]],
    min_correct: int,
) -> dict[str, object]:
    if not rows:
        raise ValueError("cannot aggregate zero cases")

    baseline_correct = sum(bool(row["baseline_frozen_execution_match"]) for row in rows)
    dgsl_correct = sum(bool(row["dgsl_frozen_execution_match"]) for row in rows)
    wrong_to_correct = sum(
        (not bool(row["baseline_frozen_execution_match"]))
        and bool(row["dgsl_frozen_execution_match"])
        for row in rows
    )
    correct_to_wrong = sum(
        bool(row["baseline_frozen_execution_match"])
        and (not bool(row["dgsl_frozen_execution_match"]))
        for row in rows
    )

    baseline_metric_rows = [
        row["baseline_grounding"]
        for row in rows
        if isinstance(row.get("baseline_grounding"), dict)
    ]
    dgsl_metric_rows = [
        row["dgsl_grounding"]
        for row in rows
        if isinstance(row.get("dgsl_grounding"), dict)
    ]
    metric_names = (
        "table_recall",
        "table_precision",
        "column_recall",
        "column_precision",
        "fk_bridge_recall",
        "value_grounding_recall",
    )
    grounding = {
        metric: {
            "lexical": _mean_metric(baseline_metric_rows, metric),
            "dgsl": _mean_metric(dgsl_metric_rows, metric),
        }
        for metric in metric_names
    }

    decomposition_latencies = [
        float(row["decomposition_latency_s"])
        for row in rows
        if row.get("decomposition_latency_s") is not None
    ]
    generation_latencies = [
        float(row["dgsl_generation_latency_s"])
        for row in rows
        if row.get("dgsl_generation_latency_s") is not None
    ]

    return {
        "sample_size": len(rows),
        "baseline_correct": baseline_correct,
        "dgsl_correct": dgsl_correct,
        "net_correct_delta": dgsl_correct - baseline_correct,
        "wrong_to_correct": wrong_to_correct,
        "correct_to_wrong": correct_to_wrong,
        "min_correct_gate": min_correct,
        "gate_met": dgsl_correct >= min_correct,
        "grounding_metrics_macro": grounding,
        "median_decomposition_latency_s": (
            median(decomposition_latencies) if decomposition_latencies else None
        ),
        "median_dgsl_generation_latency_s": (
            median(generation_latencies) if generation_latencies else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--database-root", required=True)
    parser.add_argument("--frozen", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--baseline-results")
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--value-limit", type=int, default=256)
    parser.add_argument("--min-correct", type=int, default=61)
    args = parser.parse_args()

    source = Path(args.source)
    database_root = Path(args.database_root)
    frozen = json.loads(Path(args.frozen).read_text(encoding="utf-8"))
    frozen_cases = frozen["cases"]
    ordered_ids = [str(row["case_id"]) for row in frozen_cases]
    selected_ids = [
        case_id
        for index, case_id in enumerate(ordered_ids)
        if index % args.num_shards == args.shard_index
    ]

    cases, labels = load_bird_json(source)
    case_by_id = {str(case.case_id): case for case in cases}
    label_by_id = {str(label.case_id): label for label in labels}
    baseline_sql_by_id = load_baseline_sql(
        Path(args.baseline_results) if args.baseline_results else None
    )

    provider = OllamaProvider(model=args.model, timeout_s=900.0)
    cache: dict[str, tuple[Path, DatabaseCatalog, dict, tuple]] = {}
    rows: list[dict[str, object]] = []

    for index, case_id in enumerate(selected_ids, start=1):
        case = case_by_id[case_id]
        label = label_by_id[case_id]

        if case.database_id not in cache:
            database = resolve_database(database_root, case.database_id)
            catalog = DatabaseCatalog.from_sqlite(database)
            descriptions = {
                (table, column): description
                for table, column, description in _description_rows(
                    database_root,
                    case.database_id,
                )
            }
            value_index = build_value_index(
                catalog,
                max_values_per_column=args.value_limit,
            )
            cache[case.database_id] = (
                database,
                catalog,
                descriptions,
                value_index,
            )

        database, catalog, descriptions, value_index = cache[case.database_id]
        evidence = case.evidence.strip() or None

        lexical_context, lexical_diagnostics = select_schema_context(
            catalog,
            database_root,
            case.database_id,
            case.question,
        )
        baseline_sql = baseline_sql_by_id.get(case_id)
        baseline_generation_latency_s: float | None = None
        baseline_reused = baseline_sql is not None
        if baseline_sql is None:
            started = time.perf_counter()
            baseline_sql = generate_xiyan(
                question=case.question,
                schema_context=lexical_context,
                evidence=evidence,
                provider=provider,
            )
            baseline_generation_latency_s = time.perf_counter() - started
        baseline_score = score_query(database, baseline_sql, label.sql)
        baseline_grounding = score_selected_tables(
            lexical_diagnostics["selected_tables"],
            gold_sql=label.sql,
            catalog=catalog,
        )

        decomposition_started = time.perf_counter()
        decomposition, raw_decomposition, decomposition_fallback = decompose_xiyan(
            question=case.question,
            provider=provider,
        )
        decomposition_latency_s = time.perf_counter() - decomposition_started

        pack = build_grounding_pack(
            catalog,
            decomposition=decomposition,
            question=case.question,
            evidence=evidence,
            descriptions=descriptions,
            value_index=value_index,
        )
        dgsl_grounding = score_grounding_pack(
            pack,
            gold_sql=label.sql,
            catalog=catalog,
        )

        dgsl_started = time.perf_counter()
        dgsl_sql = generate_xiyan(
            question=case.question,
            schema_context=pack.context,
            evidence=evidence,
            provider=provider,
        )
        dgsl_generation_latency_s = time.perf_counter() - dgsl_started
        dgsl_score = score_query(database, dgsl_sql, label.sql)

        row = {
            "case_id": case_id,
            "database_id": case.database_id,
            "question": case.question,
            "baseline_reused": baseline_reused,
            "baseline_sql": baseline_sql,
            "baseline_generation_latency_s": baseline_generation_latency_s,
            "baseline_execution_success": baseline_score["execution_success"],
            "baseline_frozen_execution_match": baseline_score["frozen_execution_match"],
            "baseline_grounding": baseline_grounding,
            "decomposition": {
                "entity": decomposition.entity,
                "metric": decomposition.metric,
                "filters": list(decomposition.filters),
                "ranking": decomposition.ranking,
                "time": list(decomposition.time),
                "fallback": decomposition_fallback,
                "raw_response": raw_decomposition,
            },
            "decomposition_latency_s": decomposition_latency_s,
            "dgsl_context_sha256": sha256(pack.context.encode("utf-8")).hexdigest(),
            "dgsl_context": pack.context,
            "dgsl_diagnostics": pack.diagnostics(),
            "dgsl_grounding": dgsl_grounding,
            "dgsl_sql": dgsl_sql,
            "dgsl_generation_latency_s": dgsl_generation_latency_s,
            "dgsl_execution_success": dgsl_score["execution_success"],
            "dgsl_frozen_execution_match": dgsl_score["frozen_execution_match"],
            "dgsl_truncated": dgsl_score["truncated"],
        }
        rows.append(row)

        print(
            f"[shard {args.shard_index} {index:02d}/{len(selected_ids)}] "
            f"case={case_id} baseline={baseline_score['frozen_execution_match']} "
            f"dgsl={dgsl_score['frozen_execution_match']} "
            f"anchors={','.join(pack.anchor_tables[:4])} "
            f"values={len(pack.value_hits)}",
            flush=True,
        )

    payload = {
        "experiment": "XiYan lexical schema context vs DGSL v1 grounded schema context",
        "model": args.model,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "selected_case_ids": selected_ids,
        "gold_visible_to_generation_or_grounding": False,
        "gold_used_for_offline_metrics_only": True,
        "scorer": "project frozen set-equality execution match; not official BIRD EX",
        "value_limit_per_column": args.value_limit,
        "min_correct_gate": args.min_correct,
        "aggregate": aggregate_case_rows(rows, args.min_correct),
        "cases": rows,
    }
    Path(args.output).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
