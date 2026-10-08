from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "evaluation"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bird_loader import load_bird_json
from schema_context import select_schema_context
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.direct import _strip_fence
from semantic_sql.execution import SQLExecutionError, ExecutionResult, execute_readonly
from semantic_sql.grounding import build_grounding_pack, build_value_index, heuristic_decomposition
from semantic_sql.providers import OllamaProvider
from semantic_sql.verifier import verify_sql_preflight


def should_attempt_zero_row_repair(*, execution_success: bool, row_count: int, truncated: bool) -> bool:
    return execution_success and row_count == 0 and not truncated


def _resolve_database(database_root: Path, db_id: str) -> Path:
    direct = database_root / db_id / f"{db_id}.sqlite"
    if direct.is_file():
        return direct
    matches = [p for p in (database_root / db_id).glob("*.sqlite") if "__MACOSX" not in p.parts]
    if len(matches) == 1:
        return matches[0]
    matches = [p for p in database_root.rglob("*.sqlite") if p.stem == db_id and "__MACOSX" not in p.parts]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"SQLite database not found for {db_id}")


def _official_ex(predicted: ExecutionResult | None, gold: ExecutionResult | None) -> bool:
    if predicted is None or gold is None or predicted.truncated or gold.truncated:
        return False
    return set(predicted.rows) == set(gold.rows)


def _load_baseline_cases(pattern: str) -> list[dict]:
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        raise FileNotFoundError(f"no baseline shard files matched {pattern}")
    cases = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        cases.extend(payload["cases"])
    by_id = {str(row["case_id"]): row for row in cases}
    if len(cases) != 100 or len(by_id) != 100:
        raise ValueError(f"expected exactly 100 unique baseline cases, got {len(cases)} / {len(by_id)}")
    return cases


def _repair_prompt(*, question: str, evidence: str | None, candidate_sql: str, grounding_context: str) -> tuple[str, str]:
    system = (
        "Repair one read-only SQLite query only when its zero-row result is likely caused by a semantic mismatch. "
        "Check filter literals, filter columns, join keys, and overly restrictive conditions against the grounded database values/schema. "
        "Zero rows can be legitimate, so preserve the original query when no concrete mismatch is supported. "
        "Return exactly one read-only SELECT or CTE and no explanation."
    )
    user = (
        f"Question:\n{question}\n\n"
        f"Grounding:\n{grounding_context}\n\n"
        + (f"External evidence:\n{evidence}\n\n" if evidence else "")
        + f"Executable candidate that returned zero rows:\n{candidate_sql}\n"
    )
    return system, user


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--database-root", required=True)
    parser.add_argument("--baseline-glob", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source_path = Path(args.source)
    database_root = Path(args.database_root)
    baseline_cases = _load_baseline_cases(args.baseline_glob)
    cases, labels = load_bird_json(source_path)
    case_by_id = {str(row.case_id): row for row in cases}
    label_by_id = {str(row.case_id): row for row in labels}
    provider = OllamaProvider(model=args.model, timeout_s=900.0)

    cache: dict[str, tuple[Path, DatabaseCatalog, tuple]] = {}
    rows: list[dict[str, object]] = []
    for index, baseline in enumerate(baseline_cases, 1):
        case_id = str(baseline["case_id"])
        case = case_by_id[case_id]
        label = label_by_id[case_id]
        if case.database_id not in cache:
            database = _resolve_database(database_root, case.database_id)
            catalog = DatabaseCatalog.from_sqlite(database)
            cache[case.database_id] = (database, catalog, build_value_index(catalog))
        database, catalog, value_index = cache[case.database_id]

        original_sql = str(baseline["guarded"]["final_sql"] or "")
        original_execution = None
        original_error = None
        try:
            original_execution = execute_readonly(database, original_sql, max_rows=50_000)
        except SQLExecutionError as exc:
            original_error = str(exc)

        try:
            gold_execution = execute_readonly(database, label.sql, max_rows=50_000)
        except SQLExecutionError as exc:
            raise RuntimeError(f"gold SQL failed for {case_id}: {exc}") from exc

        eligible = should_attempt_zero_row_repair(
            execution_success=original_execution is not None,
            row_count=len(original_execution.rows) if original_execution is not None else 0,
            truncated=bool(original_execution.truncated) if original_execution is not None else False,
        )
        final_sql = original_sql
        final_execution = original_execution
        attempted = False
        accepted_repair = False
        grounding_values: list[dict[str, object]] = []

        if eligible:
            attempted = True
            schema_context, _ = select_schema_context(catalog, database_root, case.database_id, case.question)
            decomposition = heuristic_decomposition(case.question)
            pack = build_grounding_pack(
                catalog,
                decomposition=decomposition,
                question=case.question,
                evidence=case.evidence.strip() or None,
                value_index=value_index,
            )
            grounding_values = [
                {"table": hit.table, "column": hit.column, "value": hit.value, "score": hit.score}
                for hit in pack.value_hits
            ]
            grounding_context = pack.context if pack.context.strip() else schema_context
            system, user = _repair_prompt(
                question=case.question,
                evidence=case.evidence.strip() or None,
                candidate_sql=original_sql,
                grounding_context=grounding_context,
            )
            proposed_sql = _strip_fence(provider.complete_text(system=system, user=user))
            verification = verify_sql_preflight(database, proposed_sql)
            if verification.ok:
                try:
                    proposed_execution = execute_readonly(database, proposed_sql, max_rows=50_000)
                except SQLExecutionError:
                    proposed_execution = None
                if proposed_execution is not None and not proposed_execution.truncated:
                    final_sql = proposed_sql
                    final_execution = proposed_execution
                    accepted_repair = proposed_sql != original_sql

        baseline_ex = _official_ex(original_execution, gold_execution)
        treatment_ex = _official_ex(final_execution, gold_execution)
        rows.append({
            "case_id": case_id,
            "database_id": case.database_id,
            "question": case.question,
            "baseline_sql": original_sql,
            "baseline_execution_success": original_execution is not None,
            "baseline_row_count": len(original_execution.rows) if original_execution is not None else None,
            "baseline_truncated": bool(original_execution.truncated) if original_execution is not None else None,
            "baseline_official_ex": baseline_ex,
            "baseline_error": original_error,
            "eligible_zero_row": eligible,
            "repair_attempted": attempted,
            "repair_accepted": accepted_repair,
            "matched_values": grounding_values,
            "treatment_sql": final_sql,
            "treatment_execution_success": final_execution is not None,
            "treatment_row_count": len(final_execution.rows) if final_execution is not None else None,
            "treatment_official_ex": treatment_ex,
        })
        print(
            f"[{index:03d}/100] {case_id} eligible={eligible} "
            f"baseline={baseline_ex} treatment={treatment_ex}",
            flush=True,
        )

    baseline_correct = sum(bool(row["baseline_official_ex"]) for row in rows)
    treatment_correct = sum(bool(row["treatment_official_ex"]) for row in rows)
    payload = {
        "experiment": "Qwen3.5 9B zero-row conditional value repair",
        "model": args.model,
        "sample_size": 100,
        "gold_visible_to_model": False,
        "trigger": "guarded SQL executes successfully, returns exactly zero rows, and is not truncated",
        "baseline_correct": baseline_correct,
        "treatment_correct": treatment_correct,
        "wrong_to_correct": sum((not row["baseline_official_ex"]) and row["treatment_official_ex"] for row in rows),
        "correct_to_wrong": sum(row["baseline_official_ex"] and (not row["treatment_official_ex"]) for row in rows),
        "eligible_cases": sum(bool(row["eligible_zero_row"]) for row in rows),
        "accepted_repairs": sum(bool(row["repair_accepted"]) for row in rows),
        "gate_met": treatment_correct > baseline_correct and not any(
            row["baseline_official_ex"] and (not row["treatment_official_ex"]) for row in rows
        ),
        "cases": rows,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "cases"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
