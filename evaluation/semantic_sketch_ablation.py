from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.direct import generate_direct_sql
from semantic_sql.execution import SQLExecutionError, execute_readonly
from semantic_sql.providers import OllamaProvider
from semantic_sql.semantic_sketch import analyze_semantics

try:
    from .bird_loader import load_bird_json
    from .protocol import selected_case_ids, validate_source_hash
    from .run_bird import CountingProvider, _official_ex_results, _resolve_database
    from .schema_context import select_schema_context
except ImportError:  # direct script execution
    from bird_loader import load_bird_json
    from protocol import selected_case_ids, validate_source_hash
    from run_bird import CountingProvider, _official_ex_results, _resolve_database
    from schema_context import select_schema_context


def select_pilot_case_ids(cases: list[dict], *, limit: int = 20) -> list[str]:
    return [
        str(case["case_id"])
        for case in cases
        if not bool(case["direct"]["official_ex"])
    ][:limit]


def aggregate_pairs(rows: list[dict]) -> dict:
    total = len(rows)
    if total == 0:
        raise ValueError("cannot aggregate zero pairs")

    def arm(name: str) -> dict[str, float]:
        return {
            "official_bird_ex": sum(bool(row[name]["official_ex"]) for row in rows) / total,
            "execution_success": sum(bool(row[name]["execution_success"]) for row in rows) / total,
        }

    transitions = {
        "wrong_to_correct": sum(
            (not bool(row["baseline"]["official_ex"])) and bool(row["semantic"]["official_ex"])
            for row in rows
        ),
        "correct_to_wrong": sum(
            bool(row["baseline"]["official_ex"]) and (not bool(row["semantic"]["official_ex"]))
            for row in rows
        ),
        "execution_fail_to_success": sum(
            (not bool(row["baseline"]["execution_success"])) and bool(row["semantic"]["execution_success"])
            for row in rows
        ),
        "execution_success_to_fail": sum(
            bool(row["baseline"]["execution_success"]) and (not bool(row["semantic"]["execution_success"]))
            for row in rows
        ),
    }
    baseline = arm("baseline")
    semantic = arm("semantic")
    return {
        "sample_size": total,
        "baseline": baseline,
        "semantic": semantic,
        "transitions": transitions,
        "pilot_gate_pass": (
            transitions["wrong_to_correct"] > transitions["correct_to_wrong"]
            and semantic["execution_success"] >= baseline["execution_success"]
        ),
    }


def run_ablation(
    *,
    source_path: Path,
    database_root: Path,
    manifest_path: Path,
    baseline_path: Path,
    output_path: Path,
    model: str,
    shard_index: int = 0,
    num_shards: int = 1,
    base_url: str = "http://localhost:11434",
    timeout_s: float = 600.0,
    max_rows: int = 50_000,
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_source_hash(source_path, str(manifest["source_sha256"]))
    frozen_ids = selected_case_ids(manifest, "dev")
    selected_ids = [case_id for i, case_id in enumerate(frozen_ids) if i % num_shards == shard_index]

    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_by_id = {str(case["case_id"]): case for case in baseline_payload["cases"]}
    cases, labels = load_bird_json(source_path)
    case_by_id = {str(case.case_id): case for case in cases}
    label_by_id = {str(label.case_id): label for label in labels}

    cache: dict[str, tuple[Path, DatabaseCatalog]] = {}
    rows: list[dict] = []
    for index, case_id in enumerate(selected_ids, start=1):
        case = case_by_id[str(case_id)]
        label = label_by_id[str(case_id)]
        baseline_case = baseline_by_id[str(case_id)]
        if str(baseline_case["database_id"]) != str(case.database_id) or str(baseline_case["question"]) != str(case.question):
            raise RuntimeError(f"baseline identity mismatch for case {case_id}")

        if case.database_id not in cache:
            database = _resolve_database(database_root, case.database_id)
            cache[case.database_id] = (database, DatabaseCatalog.from_sqlite(database))
        database, catalog = cache[case.database_id]
        schema_context, schema_diagnostics = select_schema_context(
            catalog, database_root, case.database_id, case.question
        )
        evidence = case.evidence.strip() or None
        provider = CountingProvider(OllamaProvider(model=model, base_url=base_url, timeout_s=timeout_s))

        started = time.perf_counter()
        sketch = analyze_semantics(
            case.question,
            schema_context,
            provider,
            external_evidence=evidence,
        )
        candidate = generate_direct_sql(
            case.question,
            schema_context,
            provider,
            external_evidence=evidence,
            semantic_sketch=sketch,
        )
        latency = time.perf_counter() - started

        semantic_error = None
        semantic_execution = None
        try:
            semantic_execution = execute_readonly(database, candidate.sql, max_rows=max_rows)
        except SQLExecutionError as exc:
            semantic_error = str(exc)
        try:
            gold_execution = execute_readonly(database, label.sql, max_rows=max_rows)
        except SQLExecutionError as exc:
            raise RuntimeError(f"gold SQL failed for case {case_id}") from exc

        truncated = bool(
            (semantic_execution is not None and semantic_execution.truncated)
            or gold_execution.truncated
        )
        semantic_success = semantic_execution is not None and not truncated
        semantic_ex = _official_ex_results(semantic_execution, gold_execution)
        baseline_direct = baseline_case["direct"]
        row = {
            "case_id": str(case_id),
            "database_id": case.database_id,
            "question": case.question,
            "schema_context": schema_diagnostics,
            "baseline": {
                "sql": baseline_direct["final_sql"],
                "official_ex": bool(baseline_direct["official_ex"]),
                "execution_success": bool(baseline_direct["execution_success"]),
            },
            "semantic": {
                "sql": candidate.sql,
                "official_ex": bool(semantic_ex),
                "execution_success": bool(semantic_success),
                "truncated": truncated,
                "error": semantic_error,
                "model_calls": provider.total_calls,
                "latency_s": latency,
                "sketch": asdict(sketch),
            },
        }
        rows.append(row)
        print(
            f"[semantic {index:02d}/{len(selected_ids)}] case={case_id} "
            f"baseline={row['baseline']['official_ex']} semantic={semantic_ex}",
            flush=True,
        )

    payload = {
        "status": "complete",
        "model": model,
        "sample_size": len(rows),
        "shard_index": shard_index,
        "num_shards": num_shards,
        "metrics": aggregate_pairs(rows),
        "cases": rows,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Semantic-sketch Text-to-SQL ablation")
    parser.add_argument("--source", required=True)
    parser.add_argument("--database-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-rows", type=int, default=50_000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    payload = run_ablation(
        source_path=Path(args.source),
        database_root=Path(args.database_root),
        manifest_path=Path(args.manifest),
        baseline_path=Path(args.baseline),
        output_path=Path(args.output),
        model=args.model,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        base_url=args.base_url,
        timeout_s=args.timeout,
        max_rows=args.max_rows,
    )
    print(json.dumps(payload["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
