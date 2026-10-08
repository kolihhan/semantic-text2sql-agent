from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from time import perf_counter

from evaluation.bird_loader import load_bird_json
from evaluation.run_bird import CountingProvider, _official_ex_results, _resolve_database
from evaluation.schema_context import select_schema_context
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.contracts import SQLCandidate
from semantic_sql.execution import ExecutionResult, SQLExecutionError, execute_readonly
from semantic_sql.inference import (
    ZERO_RESULT_FILTER_REVISION_TREATMENT_ID,
    run_guarded_zero_result_revision,
)
from semantic_sql.providers import OllamaProvider


_ROWS_RE = re.compile(r"^rows=(\d+)$")


def _final_row_count(guarded: dict) -> int | None:
    rows = None
    for stage in guarded.get("stages", []):
        if stage.get("name") != "execute":
            continue
        match = _ROWS_RE.match(str(stage.get("summary", "")))
        if match:
            rows = int(match.group(1))
    return rows


def _load_baseline(paths: list[Path]) -> list[dict]:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    cases = [case for payload in payloads for case in payload.get("cases", [])]
    if len(cases) != 100:
        raise ValueError(f"expected exact frozen-100 baseline, got {len(cases)} cases")
    ids = [str(case.get("case_id")) for case in cases]
    if len(set(ids)) != 100:
        raise ValueError("baseline case IDs are not unique")
    if any(case.get("initial_sql_identical") is not True for case in cases):
        raise ValueError("baseline paired initial-SQL identity is invalid")
    return cases


def _selected_zero_result_cases(cases: list[dict]) -> list[dict]:
    return [
        case for case in cases
        if case.get("guarded", {}).get("execution_success") is True
        and _final_row_count(case["guarded"]) == 0
    ]


def run_experiment(
    *,
    source_path: Path,
    database_root: Path,
    baseline_paths: list[Path],
    output_path: Path,
    model: str,
    base_url: str,
    timeout_s: float,
    max_rows: int,
    expected_selected: int | None = None,
) -> dict:
    baseline = _load_baseline(baseline_paths)
    selected = _selected_zero_result_cases(baseline)
    if expected_selected is not None and len(selected) != expected_selected:
        raise ValueError(f"expected {expected_selected} zero-result cases, got {len(selected)}")

    source_cases, labels = load_bird_json(source_path)
    source_by_id = {str(case.case_id): case for case in source_cases}
    labels_by_id = {str(label.case_id): label for label in labels}
    cache: dict[str, tuple[Path, DatabaseCatalog]] = {}
    results = []

    for index, baseline_case in enumerate(selected, 1):
        case_id = str(baseline_case["case_id"])
        case = source_by_id[case_id]
        label = labels_by_id[case_id]
        if case.database_id not in cache:
            database = _resolve_database(database_root, case.database_id)
            cache[case.database_id] = (database, DatabaseCatalog.from_sqlite(database))
        database, catalog = cache[case.database_id]
        schema_context, diagnostics = select_schema_context(
            catalog, database_root, case.database_id, case.question
        )
        baseline_sql = str(baseline_case["guarded"]["final_sql"])
        provider = CountingProvider(
            OllamaProvider(model=model, base_url=base_url, timeout_s=timeout_s)
        )
        started = perf_counter()
        result = run_guarded_zero_result_revision(
            database=database,
            provider=provider,
            question=case.question,
            schema_context=schema_context,
            evidence=case.evidence.strip() or None,
            initial_candidate=SQLCandidate(baseline_sql),
            max_repairs=0,
            max_rows=max_rows,
        )
        latency_s = perf_counter() - started

        try:
            gold_execution = execute_readonly(database, label.sql, max_rows=max_rows)
        except SQLExecutionError as exc:
            raise RuntimeError(f"gold SQL failed for {case_id}") from exc
        if gold_execution.truncated:
            raise RuntimeError(f"gold SQL truncated for selected case {case_id}")
        treatment_execution = None
        if result.status == "ok":
            treatment_execution = ExecutionResult(
                columns=result.columns,
                rows=result.rows,
                truncated=result.truncated,
            )
        treatment_ex = _official_ex_results(treatment_execution, gold_execution)
        accepted = any(
            stage.name == "empty_result_revision" and "accepted" in stage.summary
            for stage in result.stages
        )
        results.append({
            "case_id": case_id,
            "database_id": case.database_id,
            "question": case.question,
            "baseline_sql": baseline_sql,
            "baseline_official_ex": bool(baseline_case["guarded"]["official_ex"]),
            "treatment_sql": result.candidate.sql if result.candidate else None,
            "treatment_official_ex": treatment_ex,
            "accepted_revision": accepted,
            "result_rows": len(result.rows),
            "model_calls": provider.total_calls,
            "latency_s": latency_s,
            "schema_context": diagnostics,
            "stages": [{"name": stage.name, "summary": stage.summary} for stage in result.stages],
        })
        print(
            f"[{index:02d}/{len(selected):02d}] {case_id} "
            f"accepted={accepted} ex={treatment_ex} calls={provider.total_calls}",
            flush=True,
        )

    by_id = {row["case_id"]: row for row in results}
    before_correct = sum(bool(case["guarded"]["official_ex"]) for case in baseline)
    after_correct = sum(
        bool(by_id[str(case["case_id"])]["treatment_official_ex"])
        if str(case["case_id"]) in by_id
        else bool(case["guarded"]["official_ex"])
        for case in baseline
    )
    wrong_to_correct = sum(
        (not row["baseline_official_ex"]) and row["treatment_official_ex"]
        for row in results
    )
    correct_to_wrong = sum(
        row["baseline_official_ex"] and (not row["treatment_official_ex"])
        for row in results
    )
    payload = {
        "schema_version": "qwen9b-zero-result-filter-revision/v1",
        "treatment": ZERO_RESULT_FILTER_REVISION_TREATMENT_ID,
        "model": model,
        "baseline_cases": len(baseline),
        "selected_by_runtime_signal": len(selected),
        "selection_rule": "guarded execution_success=true and final execute rows=0; gold is not used for selection",
        "selected_case_ids": [str(case["case_id"]) for case in selected],
        "before_correct": before_correct,
        "after_correct": after_correct,
        "before_ex": before_correct / len(baseline),
        "after_ex": after_correct / len(baseline),
        "wrong_to_correct": wrong_to_correct,
        "correct_to_wrong": correct_to_wrong,
        "accepted_revisions": sum(row["accepted_revision"] for row in results),
        "model_calls": sum(row["model_calls"] for row in results),
        "cases": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.5:9b-q4_K_M")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-rows", type=int, default=50_000)
    parser.add_argument("--expected-selected", type=int, default=None)
    args = parser.parse_args()
    payload = run_experiment(
        source_path=args.source,
        database_root=args.database_root,
        baseline_paths=args.baseline,
        output_path=args.output,
        model=args.model,
        base_url=args.base_url,
        timeout_s=args.timeout,
        max_rows=args.max_rows,
        expected_selected=args.expected_selected,
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "cases"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
