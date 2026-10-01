from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "evaluation"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bird_loader import load_bird_json
from schema_context import select_schema_context
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.direct import _strip_fence
from semantic_sql.providers import OllamaProvider
from semantic_sql.verifier import verify_sql_preflight


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


def _execute_all_readonly(database: Path, sql: str) -> tuple[tuple[object, ...], ...]:
    uri = f"file:{database.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute("PRAGMA query_only = ON")
        cursor = connection.execute(sql)
        return tuple(tuple(row) for row in cursor.fetchall())


def exact_score(database: Path, predicted_sql: str, gold_sql: str) -> dict[str, bool]:
    verification = verify_sql_preflight(database, predicted_sql)
    if not verification.ok:
        return {"execution_success": False, "official_ex": False}
    try:
        predicted_rows = _execute_all_readonly(database, predicted_sql)
    except sqlite3.Error:
        return {"execution_success": False, "official_ex": False}
    try:
        gold_rows = _execute_all_readonly(database, gold_sql)
    except sqlite3.Error as exc:
        raise RuntimeError(f"gold SQL failed for {database.name}: {exc}") from exc
    return {
        "execution_success": True,
        "official_ex": set(predicted_rows) == set(gold_rows),
    }


def generate_xiyan(*, question: str, schema_context: str, evidence: str | None, provider: OllamaProvider) -> str:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--database-root", required=True)
    parser.add_argument("--frozen", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.source)
    database_root = Path(args.database_root)
    frozen = json.loads(Path(args.frozen).read_text(encoding="utf-8"))
    frozen_cases = frozen["cases"]
    frozen_by_id = {str(row["case_id"]): row for row in frozen_cases}
    ordered_ids = [str(row["case_id"]) for row in frozen_cases]
    selected_ids = [
        case_id for index, case_id in enumerate(ordered_ids)
        if index % args.num_shards == args.shard_index
    ]

    cases, labels = load_bird_json(source)
    case_by_id = {str(c.case_id): c for c in cases}
    label_by_id = {str(l.case_id): l for l in labels}

    provider = OllamaProvider(model=args.model, timeout_s=900.0)
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
        sql = generate_xiyan(
            question=case.question,
            schema_context=schema_context,
            evidence=evidence,
            provider=provider,
        )
        latency_s = time.perf_counter() - started
        treatment = exact_score(database, sql, label.sql)
        baseline = frozen_by_id[case_id]
        row = {
            "case_id": case_id,
            "database_id": case.database_id,
            "question": case.question,
            "qwen_direct_ex": bool(baseline["direct"]["official_ex"]),
            "qwen_guarded_ex": bool(baseline["guarded"]["official_ex"]),
            "xiyan_sql": sql,
            "xiyan_execution_success": treatment["execution_success"],
            "xiyan_official_ex": treatment["official_ex"],
            "xiyan_latency_s": latency_s,
        }
        rows.append(row)
        print(
            f"[shard {args.shard_index} {index:02d}/{len(selected_ids)}] "
            f"case={case_id} xiyan_ex={treatment['official_ex']}",
            flush=True,
        )

    payload = {
        "experiment": "frozen-100 XiYan direct generator evaluation",
        "model": args.model,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "selected_case_ids": selected_ids,
        "gold_visible_to_model": False,
        "benchmark_execution": "read-only fetch-all; no product row cap",
        "cases": rows,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
