from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "evaluation"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bird_loader import load_bird_json
from schema_context import select_schema_context
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.contracts import SQLCandidate
from semantic_sql.direct import _strip_fence
from semantic_sql.execution import SQLExecutionError, execute_readonly
from semantic_sql.inference import run_guarded
from semantic_sql.providers import OllamaProvider


class CountingProvider:
    def __init__(self, inner: OllamaProvider) -> None:
        self.inner = inner
        self.calls = 0

    def complete_text(self, *, system: str, user: str) -> str:
        self.calls += 1
        return self.inner.complete_text(system=system, user=user)


def resolve_database(database_root: Path, db_id: str) -> Path:
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


def official_ex(database: Path, predicted_sql: str | None, gold_sql: str) -> bool:
    if not predicted_sql:
        return False
    try:
        predicted = execute_readonly(database, predicted_sql, max_rows=50_000)
        gold = execute_readonly(database, gold_sql, max_rows=50_000)
    except SQLExecutionError:
        return False
    if predicted.truncated or gold.truncated:
        return False
    return set(predicted.rows) == set(gold.rows)


def deterministic_sample(case_ids: list[str], sample_size: int) -> list[str]:
    return sorted(case_ids, key=lambda case_id: hashlib.sha256(case_id.encode()).hexdigest())[:sample_size]


def generate_xiyan(*, question: str, schema_context: str, evidence: str | None, provider: CountingProvider) -> str:
    prompt = f"""You are an expert SQLite text-to-SQL model.
Generate exactly one read-only SQLite query that answers the user question using only the supplied schema and evidence.
Preserve entity identity, requested output columns, aggregation granularity, joins, filters, ordering, top-k semantics, and literal values.

Question:
{question}

Schema:
{schema_context}

Evidence:
{evidence or '(none)'}

Return SQL only. No markdown or explanation."""
    return _strip_fence(provider.complete_text(system="", user=prompt))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--database-root", required=True)
    parser.add_argument("--baseline", default="runs/bird-paired-dev-v4/paired.json")
    parser.add_argument("--model", required=True)
    parser.add_argument("--sample-size", type=int, default=12)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.source)
    database_root = Path(args.database_root)
    baseline_payload = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    baseline_cases = {str(row["case_id"]): row for row in baseline_payload["cases"]}
    selected_ids = deterministic_sample(list(baseline_cases), min(args.sample_size, len(baseline_cases)))

    cases, labels = load_bird_json(source)
    case_by_id = {case.case_id: case for case in cases}
    label_by_id = {label.case_id: label for label in labels}

    missing = [case_id for case_id in selected_ids if case_id not in case_by_id or case_id not in label_by_id]
    if missing:
        raise RuntimeError(f"frozen case IDs missing from BIRD source: {missing}")

    provider = CountingProvider(OllamaProvider(model=args.model, timeout_s=900.0))
    catalogs: dict[str, tuple[Path, DatabaseCatalog]] = {}
    rows: list[dict[str, object]] = []

    for index, case_id in enumerate(selected_ids, start=1):
        case = case_by_id[case_id]
        label = label_by_id[case_id]
        if case.database_id not in catalogs:
            database = resolve_database(database_root, case.database_id)
            catalogs[case.database_id] = (database, DatabaseCatalog.from_sqlite(database))
        database, catalog = catalogs[case.database_id]
        schema_context, _ = select_schema_context(catalog, database_root, case.database_id, case.question)
        evidence = case.evidence.strip() or None

        calls_before = provider.calls
        started = time.perf_counter()
        initial_sql = generate_xiyan(
            question=case.question,
            schema_context=schema_context,
            evidence=evidence,
            provider=provider,
        )
        result = run_guarded(
            database=database,
            provider=provider,
            question=case.question,
            schema_context=schema_context,
            evidence=evidence,
            initial_candidate=SQLCandidate(sql=initial_sql, attempt=0),
            max_repairs=2,
            max_rows=50_000,
        )
        latency_s = time.perf_counter() - started
        final_sql = result.candidate.sql if result.candidate else None
        treatment_ex = official_ex(database, final_sql, label.sql)
        treatment_exec = result.status == "ok"

        baseline = baseline_cases[case_id]["guarded"]
        row = {
            "case_id": case_id,
            "database_id": case.database_id,
            "question": case.question,
            "baseline": {
                "official_ex": bool(baseline["official_ex"]),
                "execution_success": bool(baseline["execution_success"]),
                "final_sql": baseline.get("final_sql"),
                "model_calls": int(baseline["model_calls"]),
                "latency_s": float(baseline["latency_s"]),
            },
            "treatment": {
                "official_ex": treatment_ex,
                "execution_success": treatment_exec,
                "initial_sql": initial_sql,
                "final_sql": final_sql,
                "status": result.status,
                "model_calls": provider.calls - calls_before,
                "latency_s": latency_s,
                "repair_attempts": final_sql is not None and sum(stage.name == "repair" for stage in result.stages),
            },
        }
        rows.append(row)
        print(
            f"[{index:02d}/{len(selected_ids)}] case={case_id} "
            f"baseline_ex={row['baseline']['official_ex']} treatment_ex={treatment_ex} "
            f"calls={row['treatment']['model_calls']}",
            flush=True,
        )

    wrong_to_correct = sum(not r["baseline"]["official_ex"] and r["treatment"]["official_ex"] for r in rows)
    correct_to_wrong = sum(r["baseline"]["official_ex"] and not r["treatment"]["official_ex"] for r in rows)
    baseline_correct = sum(r["baseline"]["official_ex"] for r in rows)
    treatment_correct = sum(r["treatment"]["official_ex"] for r in rows)
    baseline_exec = sum(r["baseline"]["execution_success"] for r in rows)
    treatment_exec = sum(r["treatment"]["execution_success"] for r in rows)

    payload = {
        "experiment": "frozen-100 subset: qwen guarded baseline vs XiYan guarded treatment",
        "sample_rule": "lowest SHA256(case_id) among the existing frozen 100 BIRD cases",
        "sample_size": len(rows),
        "selected_case_ids": selected_ids,
        "baseline_artifact": args.baseline,
        "baseline_model": baseline_payload.get("model"),
        "treatment_model": args.model,
        "gold_visible_to_models": False,
        "same_guarded_flow": True,
        "summary": {
            "baseline_correct": baseline_correct,
            "treatment_correct": treatment_correct,
            "baseline_official_ex": baseline_correct / len(rows),
            "treatment_official_ex": treatment_correct / len(rows),
            "baseline_execution_success": baseline_exec / len(rows),
            "treatment_execution_success": treatment_exec / len(rows),
            "wrong_to_correct": wrong_to_correct,
            "correct_to_wrong": correct_to_wrong,
            "net_correct_delta": wrong_to_correct - correct_to_wrong,
            "treatment_total_model_calls": sum(r["treatment"]["model_calls"] for r in rows),
        },
        "cases": rows,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
