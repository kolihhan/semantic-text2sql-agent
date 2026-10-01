from __future__ import annotations

import argparse
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


def score(database: Path, predicted_sql: str | None, gold_sql: str) -> dict[str, bool]:
    if not predicted_sql:
        return {"execution_success": False, "official_ex": False}
    try:
        predicted = execute_readonly(database, predicted_sql, max_rows=50_000)
    except SQLExecutionError:
        return {"execution_success": False, "official_ex": False}
    try:
        gold = execute_readonly(database, gold_sql, max_rows=50_000)
    except SQLExecutionError:
        raise RuntimeError("gold SQL failed to execute")
    # Keep the historical project scoring contract: truncated results cannot
    # establish execution match, but they must not abort the entire shard.
    if predicted.truncated or gold.truncated:
        return {"execution_success": True, "official_ex": False}
    return {
        "execution_success": True,
        "official_ex": set(predicted.rows) == set(gold.rows),
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


def semantic_issue(result) -> str | None:
    for stage in result.stages:
        if stage.name == "semantic_revision":
            return stage.summary
    return None


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
        direct_sql = generate_xiyan(
            question=case.question,
            schema_context=schema_context,
            evidence=evidence,
            provider=provider,
        )
        generation_latency_s = time.perf_counter() - started
        direct = score(database, direct_sql, label.sql)

        revision_started = time.perf_counter()
        revised_result = run_guarded(
            database=database,
            provider=provider,
            question=case.question,
            schema_context=schema_context,
            evidence=evidence,
            initial_candidate=SQLCandidate(sql=direct_sql, attempt=0),
            max_repairs=0,
            max_rows=50_000,
            semantic_revision=True,
        )
        revision_latency_s = time.perf_counter() - revision_started
        revised_sql = revised_result.candidate.sql if revised_result.candidate else None
        revised = score(database, revised_sql, label.sql)

        row = {
            "case_id": case_id,
            "database_id": case.database_id,
            "question": case.question,
            "xiyan_sql": direct_sql,
            "xiyan_execution_success": direct["execution_success"],
            "xiyan_official_ex": direct["official_ex"],
            "xiyan_generation_latency_s": generation_latency_s,
            "cgsr_sql": revised_sql,
            "cgsr_status": revised_result.status,
            "cgsr_issue": semantic_issue(revised_result),
            "cgsr_execution_success": revised["execution_success"],
            "cgsr_official_ex": revised["official_ex"],
            "cgsr_revision_latency_s": revision_latency_s,
        }
        rows.append(row)
        print(
            f"[shard {args.shard_index} {index:02d}/{len(selected_ids)}] "
            f"case={case_id} direct={direct['official_ex']} cgsr={revised['official_ex']} "
            f"issue={row['cgsr_issue']}",
            flush=True,
        )

    payload = {
        "experiment": "XiYan direct vs XiYan + clause-guided semantic revision",
        "model": args.model,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "selected_case_ids": selected_ids,
        "gold_visible_to_model": False,
        "max_repairs": 0,
        "semantic_revision": True,
        "cases": rows,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
