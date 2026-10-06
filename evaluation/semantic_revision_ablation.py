from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.contracts import SQLCandidate
from semantic_sql.execution import ExecutionResult, SQLExecutionError, execute_readonly
from semantic_sql.inference import run_guarded
from semantic_sql.providers import OllamaProvider

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


def select_smoke_case_ids(cases: list[dict], *, limit: int = 4) -> list[str]:
    return [
        str(case["case_id"])
        for case in cases
        if bool(case.get("guarded", {}).get("execution_success"))
        and bool(case.get("guarded", {}).get("final_sql"))
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
            (not bool(row["baseline"]["official_ex"])) and bool(row["revised"]["official_ex"])
            for row in rows
        ),
        "correct_to_wrong": sum(
            bool(row["baseline"]["official_ex"]) and (not bool(row["revised"]["official_ex"]))
            for row in rows
        ),
        "execution_fail_to_success": sum(
            (not bool(row["baseline"]["execution_success"])) and bool(row["revised"]["execution_success"])
            for row in rows
        ),
        "execution_success_to_fail": sum(
            bool(row["baseline"]["execution_success"]) and (not bool(row["revised"]["execution_success"]))
            for row in rows
        ),
    }
    return {
        "sample_size": total,
        "baseline": arm("baseline"),
        "revised": arm("revised"),
        "transitions": transitions,
        "reviewed": sum(bool(row["revision"]["reviewed"]) for row in rows),
        "changed": sum(bool(row["revision"]["changed"]) for row in rows),
        "applied": sum(bool(row["revision"]["applied"]) for row in rows),
        "fallbacks": sum(bool(row["revision"]["fallback"]) for row in rows),
        "model_calls": sum(int(row["revision"]["model_calls"]) for row in rows),
    }


def _first_stage_summary(result, name: str) -> str | None:
    for stage in result.stages:
        if stage.name == name:
            return stage.summary
    return None


def _execution(database: Path, sql: str, *, max_rows: int) -> tuple[ExecutionResult | None, str | None]:
    try:
        return execute_readonly(database, sql, max_rows=max_rows), None
    except SQLExecutionError as exc:
        return None, str(exc)


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
    if baseline_payload.get("model") != model:
        raise RuntimeError("baseline model mismatch")
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
        baseline_sql = str(baseline_case["guarded"]["final_sql"] or "").strip()
        if not baseline_sql:
            raise RuntimeError(f"missing guarded SQL for case {case_id}")

        baseline_execution, baseline_error = _execution(database, baseline_sql, max_rows=max_rows)
        try:
            gold_execution = execute_readonly(database, label.sql, max_rows=max_rows)
        except SQLExecutionError as exc:
            raise RuntimeError(f"gold SQL failed for case {case_id}") from exc
        if gold_execution.truncated:
            raise RuntimeError(f"gold SQL truncated for case {case_id}")

        baseline_success = baseline_execution is not None and not baseline_execution.truncated
        baseline_ex = _official_ex_results(baseline_execution, gold_execution)

        reviewed = False
        changed = False
        applied = False
        fallback = False
        fallback_reason = None
        issue_type = "SKIPPED"
        model_calls = 0
        latency_s = 0.0
        proposed_sql = baseline_sql
        final_sql = baseline_sql
        final_execution = baseline_execution

        if baseline_success:
            provider = CountingProvider(OllamaProvider(model=model, base_url=base_url, timeout_s=timeout_s))
            reviewed = True
            started = time.perf_counter()
            result = run_guarded(
                database=database,
                provider=provider,
                question=case.question,
                schema_context=schema_context,
                evidence=evidence,
                initial_candidate=SQLCandidate(baseline_sql),
                max_repairs=0,
                max_rows=max_rows,
                semantic_revision=True,
            )
            latency_s = time.perf_counter() - started
            model_calls = provider.total_calls
            if model_calls > 1:
                raise RuntimeError(f"semantic revision used more than one model call for case {case_id}")

            proposed_sql = result.candidate.sql if result.candidate else baseline_sql
            changed = proposed_sql != baseline_sql
            summary = _first_stage_summary(result, "semantic_revision")
            if summary:
                issue_type = summary.split(":", 1)[0].strip() or "OTHER"
            elif model_calls == 0:
                issue_type = "PREFLIGHT_SKIPPED"
            else:
                issue_type = "OTHER"

            revised_execution = None
            if result.status == "ok":
                revised_execution = ExecutionResult(
                    columns=result.columns,
                    rows=result.rows,
                    truncated=result.truncated,
                )
            if revised_execution is not None and not revised_execution.truncated:
                final_sql = proposed_sql
                final_execution = revised_execution
                applied = changed
            elif changed:
                fallback = True
                fallback_reason = "revision_not_executable_or_truncated"
        else:
            fallback_reason = "baseline_not_executable_or_truncated"

        revised_success = final_execution is not None and not final_execution.truncated
        revised_ex = _official_ex_results(final_execution, gold_execution)
        row = {
            "case_id": str(case_id),
            "database_id": case.database_id,
            "question": case.question,
            "schema_context": schema_diagnostics,
            "baseline": {
                "sql": baseline_sql,
                "official_ex": bool(baseline_ex),
                "execution_success": bool(baseline_success),
                "truncated": bool(baseline_execution is not None and baseline_execution.truncated),
                "error": baseline_error,
            },
            "revised": {
                "sql": final_sql,
                "official_ex": bool(revised_ex),
                "execution_success": bool(revised_success),
            },
            "revision": {
                "reviewed": reviewed,
                "changed": changed,
                "applied": applied,
                "fallback": fallback,
                "fallback_reason": fallback_reason,
                "issue_type": issue_type,
                "proposed_sql": proposed_sql,
                "model_calls": model_calls,
                "latency_s": latency_s,
            },
        }
        rows.append(row)
        print(
            f"[revision {index:03d}/{len(selected_ids)}] case={case_id} "
            f"baseline={baseline_ex} revised={revised_ex} changed={changed} fallback={fallback}",
            flush=True,
        )

    payload = {
        "status": "complete",
        "experiment": "conservative post-SQL semantic revision over frozen Qwen 9B guarded outputs",
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
    parser = argparse.ArgumentParser(description="Conservative semantic-revision Text-to-SQL ablation")
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
