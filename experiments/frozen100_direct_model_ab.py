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
from semantic_sql.direct import _strip_fence, generate_direct_sql
from semantic_sql.execution import SQLExecutionError, execute_readonly
from semantic_sql.providers import OllamaProvider


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def official_ex(database: Path, predicted_sql: str, gold_sql: str) -> tuple[bool, bool]:
    try:
        predicted = execute_readonly(database, predicted_sql, max_rows=50_000)
    except SQLExecutionError:
        return False, False
    try:
        gold = execute_readonly(database, gold_sql, max_rows=50_000)
    except SQLExecutionError as exc:
        raise RuntimeError(f"gold SQL failed: {exc}") from exc
    if predicted.truncated or gold.truncated:
        raise RuntimeError("truncated execution invalidates EX")
    return True, set(predicted.rows) == set(gold.rows)


def deterministic_sample(case_ids: list[str], sample_size: int) -> list[str]:
    ranked = sorted(case_ids, key=lambda case_id: hashlib.sha256(case_id.encode("utf-8")).hexdigest())
    return ranked[:sample_size]


def generate_xiyan(
    *, question: str, schema_context: str, evidence: str | None, provider: OllamaProvider
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


def score(database: Path, sql: str, gold_sql: str) -> dict[str, bool]:
    execution_success, ex = official_ex(database, sql, gold_sql)
    return {"execution_success": execution_success, "official_ex": ex}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--database-root", required=True)
    parser.add_argument("--baseline-model", default="qwen3.5:4b")
    parser.add_argument("--treatment-model", required=True)
    parser.add_argument("--sample-size", type=int, default=12)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.source)
    database_root = Path(args.database_root)
    output = Path(args.output)
    cases, labels = load_bird_json(source)
    case_by_id = {c.case_id: c for c in cases}
    label_by_id = {l.case_id: l for l in labels}
    selected_ids = deterministic_sample(list(case_by_id), min(args.sample_size, len(case_by_id)))

    baseline_provider = OllamaProvider(model=args.baseline_model, timeout_s=900.0)
    treatment_provider = OllamaProvider(model=args.treatment_model, timeout_s=900.0)
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

        started = time.perf_counter()
        baseline_sql = generate_direct_sql(
            case.question, schema_context, baseline_provider, external_evidence=evidence
        ).sql
        baseline_latency_s = time.perf_counter() - started
        baseline_score = score(database, baseline_sql, label.sql)

        started = time.perf_counter()
        treatment_sql = generate_xiyan(
            question=case.question,
            schema_context=schema_context,
            evidence=evidence,
            provider=treatment_provider,
        )
        treatment_latency_s = time.perf_counter() - started
        treatment_score = score(database, treatment_sql, label.sql)

        row = {
            "case_id": case_id,
            "database_id": case.database_id,
            "question": case.question,
            "baseline": {"sql": baseline_sql, **baseline_score, "latency_s": baseline_latency_s},
            "treatment": {"sql": treatment_sql, **treatment_score, "latency_s": treatment_latency_s},
        }
        rows.append(row)
        print(
            f"[{index:02d}/{len(selected_ids)}] {case_id} "
            f"baseline_ex={baseline_score['official_ex']} treatment_ex={treatment_score['official_ex']}",
            flush=True,
        )

    wrong_to_correct = sum(
        (not bool(row["baseline"]["official_ex"])) and bool(row["treatment"]["official_ex"])
        for row in rows
    )
    correct_to_wrong = sum(
        bool(row["baseline"]["official_ex"]) and (not bool(row["treatment"]["official_ex"]))
        for row in rows
    )
    baseline_ex = sum(bool(row["baseline"]["official_ex"]) for row in rows) / len(rows)
    treatment_ex = sum(bool(row["treatment"]["official_ex"]) for row in rows) / len(rows)
    baseline_exec = sum(bool(row["baseline"]["execution_success"]) for row in rows) / len(rows)
    treatment_exec = sum(bool(row["treatment"]["execution_success"]) for row in rows) / len(rows)

    payload = {
        "experiment": "deterministic paired BIRD model/prompt spike",
        "source_sha256": sha256(source),
        "sample_rule": "lowest SHA256(case_id), independent of model outputs",
        "sample_size": len(rows),
        "selected_case_ids": selected_ids,
        "baseline_model": args.baseline_model,
        "treatment_model": args.treatment_model,
        "gold_visible_to_models": False,
        "summary": {
            "baseline_official_ex": baseline_ex,
            "treatment_official_ex": treatment_ex,
            "baseline_execution_success": baseline_exec,
            "treatment_execution_success": treatment_exec,
            "wrong_to_correct": wrong_to_correct,
            "correct_to_wrong": correct_to_wrong,
            "net_correct_delta": wrong_to_correct - correct_to_wrong,
        },
        "cases": rows,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
