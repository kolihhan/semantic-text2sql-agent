from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median
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


def resolve_database(database_root: Path, db_id: str) -> Path:
    direct = database_root / db_id / f"{db_id}.sqlite"
    if direct.is_file():
        return direct
    matches = [
        p for p in (database_root / db_id).glob("*.sqlite")
        if "__MACOSX" not in p.parts
    ]
    if len(matches) == 1:
        return matches[0]
    matches = [
        p for p in database_root.rglob("*.sqlite")
        if p.stem == db_id and "__MACOSX" not in p.parts
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


def aggregate_case_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    sample_size = len(rows)
    if sample_size == 0:
        raise ValueError("cannot aggregate zero cases")
    if len({str(row["case_id"]) for row in rows}) != sample_size:
        raise ValueError("duplicate case_id in aggregate input")

    baseline_correct = sum(bool(row["baseline_frozen_execution_match"]) for row in rows)
    scv_correct = sum(bool(row["scv_frozen_execution_match"]) for row in rows)
    baseline_exec = sum(bool(row["baseline_execution_success"]) for row in rows)
    scv_exec = sum(bool(row["scv_execution_success"]) for row in rows)
    wrong_to_correct = sum(
        (not bool(row["baseline_frozen_execution_match"]))
        and bool(row["scv_frozen_execution_match"])
        for row in rows
    )
    correct_to_wrong = sum(
        bool(row["baseline_frozen_execution_match"])
        and (not bool(row["scv_frozen_execution_match"]))
        for row in rows
    )
    changed_sql = sum(bool(row.get("sql_changed")) for row in rows)

    status_counts = Counter(str(row.get("scv_status") or "UNKNOWN") for row in rows)
    violation_counts: Counter[str] = Counter()
    for row in rows:
        for code in row.get("scv_violation_codes") or []:
            violation_counts[str(code)] += 1

    latencies = [float(row["scv_latency_s"]) for row in rows if row.get("scv_latency_s") is not None]

    return {
        "sample_size": sample_size,
        "baseline": {
            "frozen_execution_match": baseline_correct / sample_size,
            "execution_success": baseline_exec / sample_size,
            "correct": baseline_correct,
        },
        "scv": {
            "frozen_execution_match": scv_correct / sample_size,
            "execution_success": scv_exec / sample_size,
            "correct": scv_correct,
            "changed_sql": changed_sql,
        },
        "transitions": {
            "wrong_to_correct": wrong_to_correct,
            "correct_to_wrong": correct_to_wrong,
            "net_correct_delta": scv_correct - baseline_correct,
        },
        "scv_status_counts": dict(sorted(status_counts.items())),
        "violation_counts": dict(sorted(violation_counts.items())),
        "median_scv_latency_s": median(latencies) if latencies else None,
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


def _stage_summaries(result, name: str) -> list[str]:
    return [stage.summary for stage in result.stages if stage.name == name]


def _scv_diagnostics(result) -> tuple[str, str | None, list[str]]:
    contract_summaries = _stage_summaries(result, "semantic_contract")
    verify_summaries = _stage_summaries(result, "semantic_verify")
    repaired = bool(_stage_summaries(result, "semantic_repair"))

    contract_summary = contract_summaries[-1] if contract_summaries else None
    first_violation = next(
        (summary for summary in verify_summaries if summary.startswith("SCV_VIOLATION:")),
        None,
    )
    violation_codes = (
        [code for code in first_violation.split(":", 1)[1].split(",") if code]
        if first_violation
        else []
    )

    combined = " ".join([*(contract_summaries or []), *(verify_summaries or [])])
    if "SCV_SKIPPED_CONTRACT" in combined:
        status = "SCV_SKIPPED_CONTRACT"
    elif "SCV_SKIPPED_AST" in combined:
        status = "SCV_SKIPPED_AST"
    elif result.status == "ok" and repaired:
        status = "SCV_REPAIR_PASS"
    elif result.status == "ok":
        status = "SCV_PASS"
    elif repaired:
        status = "SCV_REPAIR_FAILED"
    elif first_violation:
        status = "SCV_VIOLATION"
    else:
        status = "SCV_FAILED"
    return status, contract_summary, violation_codes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--database-root", required=True)
    parser.add_argument("--frozen", required=True)
    parser.add_argument("--generator-model", required=True)
    parser.add_argument("--semantic-model", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output", required=True)
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

    generator_provider = OllamaProvider(model=args.generator_model, timeout_s=900.0)
    semantic_provider = OllamaProvider(model=args.semantic_model, timeout_s=900.0)
    catalogs: dict[str, tuple[Path, DatabaseCatalog]] = {}
    rows: list[dict[str, object]] = []

    for index, case_id in enumerate(selected_ids, start=1):
        case = case_by_id[case_id]
        label = label_by_id[case_id]
        if case.database_id not in catalogs:
            database = resolve_database(database_root, case.database_id)
            catalogs[case.database_id] = (
                database,
                DatabaseCatalog.from_sqlite(database),
            )
        database, catalog = catalogs[case.database_id]
        schema_context, schema_diagnostics = select_schema_context(
            catalog,
            database_root,
            case.database_id,
            case.question,
        )
        evidence = case.evidence.strip() or None

        generation_started = time.perf_counter()
        direct_sql = generate_xiyan(
            question=case.question,
            schema_context=schema_context,
            evidence=evidence,
            provider=generator_provider,
        )
        generation_latency_s = time.perf_counter() - generation_started
        baseline = score_query(database, direct_sql, label.sql)

        scv_started = time.perf_counter()
        scv_result = run_guarded(
            database=database,
            provider=semantic_provider,
            question=case.question,
            schema_context=schema_context,
            evidence=evidence,
            initial_candidate=SQLCandidate(sql=direct_sql, attempt=0),
            max_repairs=0,
            max_rows=50_000,
            semantic_contract_verification=True,
            semantic_repair_budget=1,
        )
        scv_latency_s = time.perf_counter() - scv_started
        scv_sql = scv_result.candidate.sql if scv_result.candidate else None
        scv_score = score_query(database, scv_sql, label.sql)
        scv_status, contract_summary, violation_codes = _scv_diagnostics(scv_result)

        row = {
            "case_id": case_id,
            "database_id": case.database_id,
            "question": case.question,
            "schema_selector": schema_diagnostics.get("selector"),
            "schema_context_sha256": schema_diagnostics.get("schema_context_sha256"),
            "xiyan_sql": direct_sql,
            "generation_latency_s": generation_latency_s,
            "baseline_execution_success": baseline["execution_success"],
            "baseline_frozen_execution_match": baseline["frozen_execution_match"],
            "baseline_truncated": baseline["truncated"],
            "contract_summary": contract_summary,
            "scv_status": scv_status,
            "scv_violation_codes": violation_codes,
            "scv_sql": scv_sql,
            "scv_execution_success": scv_score["execution_success"],
            "scv_frozen_execution_match": scv_score["frozen_execution_match"],
            "scv_truncated": scv_score["truncated"],
            "sql_changed": (scv_sql or "") != direct_sql,
            "scv_latency_s": scv_latency_s,
        }
        rows.append(row)
        print(
            f"[shard {args.shard_index} {index:02d}/{len(selected_ids)}] "
            f"case={case_id} baseline={baseline['frozen_execution_match']} "
            f"scv={scv_score['frozen_execution_match']} status={scv_status}",
            flush=True,
        )

    payload = {
        "experiment": "XiYan candidate vs same XiYan candidate + Semantic Contract Verifier",
        "generator_model": args.generator_model,
        "semantic_model": args.semantic_model,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "selected_case_ids": selected_ids,
        "gold_visible_to_generation_or_scv": False,
        "scorer": "project frozen set-equality execution match; not official BIRD EX",
        "semantic_repair_budget": 1,
        "cases": rows,
    }
    Path(args.output).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
