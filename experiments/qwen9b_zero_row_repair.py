from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "evaluation"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bird_loader import load_bird_json
from schema_context import _description_rows, select_schema_context
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.direct import _strip_fence
from semantic_sql.execution import ExecutionResult, SQLExecutionError, execute_readonly
from semantic_sql.grounding import build_grounding_pack, build_value_index, heuristic_decomposition
from semantic_sql.providers import OllamaProvider
from semantic_sql.verifier import verify_sql_preflight
from run_bird import _ollama_model_digest, _resolve_database


def should_attempt_repair(*, execution_success: bool, truncated: bool, row_count: int) -> bool:
    return execution_success and not truncated and row_count == 0


def aggregate_rows(rows: Iterable[dict]) -> dict[str, int | bool]:
    rows = list(rows)
    baseline_correct = sum(bool(row["baseline_ex"]) for row in rows)
    treatment_correct = sum(bool(row["treatment_ex"]) for row in rows)
    wrong_to_correct = sum(
        (not bool(row["baseline_ex"])) and bool(row["treatment_ex"])
        for row in rows
    )
    correct_to_wrong = sum(
        bool(row["baseline_ex"]) and (not bool(row["treatment_ex"]))
        for row in rows
    )
    return {
        "cases": len(rows),
        "baseline_correct": baseline_correct,
        "treatment_correct": treatment_correct,
        "repair_attempted": sum(bool(row["repair_attempted"]) for row in rows),
        "repair_changed": sum(bool(row["repair_changed"]) for row in rows),
        "wrong_to_correct": wrong_to_correct,
        "correct_to_wrong": correct_to_wrong,
        "promote": treatment_correct > baseline_correct and correct_to_wrong == 0,
    }


def _load_baseline(paths: list[Path]) -> tuple[list[dict], str, str]:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if len(payloads) != 4:
        raise ValueError(f"expected four frozen shard artifacts, got {len(payloads)}")
    models = {str(payload.get("model")) for payload in payloads}
    digests = {str(payload.get("model_digest")) for payload in payloads}
    if len(models) != 1 or len(digests) != 1:
        raise ValueError("baseline shards disagree on model identity")
    cases = [case for payload in payloads for case in payload.get("cases", [])]
    ids = [str(case.get("case_id")) for case in cases]
    if len(cases) != 100 or len(set(ids)) != 100:
        raise ValueError("baseline artifacts must contain exactly 100 unique frozen cases")
    return cases, next(iter(models)), next(iter(digests))


def _matches(predicted: ExecutionResult | None, gold: ExecutionResult | None) -> bool:
    if predicted is None or gold is None or predicted.truncated or gold.truncated:
        return False
    return set(predicted.rows) == set(gold.rows)


def _repair_sql(
    *,
    question: str,
    evidence: str | None,
    original_sql: str,
    grounding_context: str,
    provider: OllamaProvider,
) -> str:
    prompt = f"""Question:
{question}

External evidence:
{evidence or ''}

Grounded schema and bounded database values:
{grounding_context}

Current SQL:
{original_sql}

The current SQL is valid SQLite and executed successfully, but returned zero rows.
Review only FILTER/VALUE grounding: wrong filter column, wrong literal, wrong spelling/case/value form, or a missing filter required by the question.
Do not rewrite the query merely for style. If there is no concrete filter/value mismatch supported by the grounded context, return the original SQL unchanged.
Return exactly one read-only SQLite SELECT or CTE and no explanation."""
    response = provider.complete_text(
        system=(
            "Repair a zero-row SQLite query only when the supplied grounded database evidence supports a concrete "
            "filter/value correction. Preserve the requested projection, aggregation, joins, ordering, and limits unless "
            "a filter/value correction necessarily touches the filter column. Return SQL only."
        ),
        user=prompt,
    )
    return _strip_fence(response).strip()


def run_experiment(
    *,
    source_path: Path,
    database_root: Path,
    baseline_paths: list[Path],
    output_path: Path,
    model: str,
    base_url: str = "http://127.0.0.1:11434",
    timeout_s: float = 600.0,
    max_rows: int = 50_000,
) -> dict:
    baseline_cases, baseline_model, baseline_digest = _load_baseline(baseline_paths)
    if baseline_model != model:
        raise ValueError(f"model mismatch: baseline={baseline_model} requested={model}")
    current_digest = _ollama_model_digest(base_url, model)
    if current_digest != baseline_digest:
        raise ValueError(
            f"model digest drift: baseline={baseline_digest} current={current_digest}"
        )

    cases, labels = load_bird_json(source_path)
    cases_by_id = {case.case_id: case for case in cases}
    labels_by_id = {label.case_id: label for label in labels}
    provider = OllamaProvider(model=model, base_url=base_url, timeout_s=timeout_s)

    cache: dict[str, tuple[Path, DatabaseCatalog, tuple, dict[tuple[str, str], str]]] = {}
    output_rows: list[dict] = []

    for index, baseline in enumerate(baseline_cases, 1):
        case_id = str(baseline["case_id"])
        case = cases_by_id[case_id]
        label = labels_by_id[case_id]
        if str(baseline["database_id"]) != case.database_id or str(baseline["question"]) != case.question:
            raise ValueError(f"baseline/source identity mismatch: {case_id}")

        if case.database_id not in cache:
            database = _resolve_database(database_root, case.database_id)
            catalog = DatabaseCatalog.from_sqlite(database)
            value_index = build_value_index(catalog)
            descriptions = {
                (table, column): description
                for table, column, description in _description_rows(database_root, case.database_id)
            }
            cache[case.database_id] = (database, catalog, value_index, descriptions)
        database, catalog, value_index, descriptions = cache[case.database_id]

        original_sql = str(baseline["guarded"].get("final_sql") or "").strip()
        baseline_execution: ExecutionResult | None = None
        baseline_error: str | None = None
        try:
            baseline_execution = execute_readonly(database, original_sql, max_rows=max_rows)
        except SQLExecutionError as exc:
            baseline_error = str(exc)

        try:
            gold_execution = execute_readonly(database, label.sql, max_rows=max_rows)
        except SQLExecutionError as exc:
            raise RuntimeError(f"gold SQL failed for {case_id}") from exc

        baseline_ex = _matches(baseline_execution, gold_execution)
        if baseline_ex != bool(baseline["guarded"].get("official_ex")):
            raise ValueError(f"baseline EX mismatch for {case_id}")

        attempted = should_attempt_repair(
            execution_success=baseline_execution is not None,
            truncated=bool(baseline_execution.truncated) if baseline_execution else False,
            row_count=len(baseline_execution.rows) if baseline_execution else 0,
        )
        treatment_sql = original_sql
        repair_changed = False
        repair_accepted = False
        repair_reason = "not_zero_row"
        value_hits: list[dict] = []

        if attempted:
            _schema_context, _schema_diag = select_schema_context(
                catalog, database_root, case.database_id, case.question
            )
            pack = build_grounding_pack(
                catalog,
                decomposition=heuristic_decomposition(case.question),
                question=case.question,
                evidence=case.evidence.strip() or None,
                descriptions=descriptions,
                value_index=value_index,
            )
            value_hits = [
                {"table": hit.table, "column": hit.column, "value": hit.value, "score": hit.score}
                for hit in pack.value_hits
            ]
            proposed = _repair_sql(
                question=case.question,
                evidence=case.evidence.strip() or None,
                original_sql=original_sql,
                grounding_context=pack.context,
                provider=provider,
            )
            repair_changed = proposed != original_sql
            if not proposed:
                repair_reason = "empty_model_output"
            else:
                verification = verify_sql_preflight(database, proposed)
                if not verification.ok:
                    repair_reason = "preflight_rejected"
                else:
                    try:
                        proposed_execution = execute_readonly(database, proposed, max_rows=max_rows)
                    except SQLExecutionError:
                        repair_reason = "execution_failed"
                    else:
                        if proposed_execution.truncated:
                            repair_reason = "truncated"
                        else:
                            treatment_sql = proposed
                            repair_accepted = True
                            repair_reason = "accepted"

        try:
            treatment_execution = execute_readonly(database, treatment_sql, max_rows=max_rows)
        except SQLExecutionError:
            treatment_execution = None
        treatment_ex = _matches(treatment_execution, gold_execution)

        output_rows.append({
            "case_id": case_id,
            "database_id": case.database_id,
            "baseline_sql": original_sql,
            "treatment_sql": treatment_sql,
            "baseline_ex": baseline_ex,
            "treatment_ex": treatment_ex,
            "baseline_row_count": len(baseline_execution.rows) if baseline_execution else None,
            "baseline_truncated": bool(baseline_execution.truncated) if baseline_execution else False,
            "baseline_error": baseline_error,
            "repair_attempted": attempted,
            "repair_changed": repair_changed,
            "repair_accepted": repair_accepted,
            "repair_reason": repair_reason,
            "value_hits": value_hits,
        })
        print(
            f"[{index:03d}/100] {case_id} zero={attempted} changed={repair_changed} "
            f"baseline={baseline_ex} treatment={treatment_ex}",
            flush=True,
        )

    summary = aggregate_rows(output_rows)
    if summary["baseline_correct"] != 39:
        raise ValueError(f"expected frozen Guarded baseline 39/100, got {summary['baseline_correct']}")
    payload = {
        "status": "complete",
        "experiment": "qwen9b-zero-row-filter-value-repair-v1",
        "model": model,
        "model_digest": current_digest,
        "baseline_run_id": 37307108955,
        "sample_size": 100,
        "scope": "reuse frozen Guarded SQL; one FILTER/VALUE repair call only after valid zero-row execution",
        "summary": summary,
        "cases": output_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.5:9b-q4_K_M")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-rows", type=int, default=50_000)
    args = parser.parse_args(argv)
    baseline_paths = [Path(path) for path in sorted(glob.glob(str(args.baseline_dir / "qwen9b-shard-*.json")))]
    payload = run_experiment(
        source_path=args.source,
        database_root=args.database_root,
        baseline_paths=baseline_paths,
        output_path=args.output,
        model=args.model,
        base_url=args.base_url,
        timeout_s=args.timeout,
        max_rows=args.max_rows,
    )
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
