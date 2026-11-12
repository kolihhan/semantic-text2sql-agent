"""One-shot XiYan 7B M-Schema experiment against frozen XiYan lexical SQL.

Baseline is read from the original 58/100 Actions artifacts, never regenerated.
Gold SQL is used *only after* candidate generation for offline execution scoring.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import time

from experiments.xiyan_dgsl_frozen100 import (
    generate_xiyan, load_baseline_sql, resolve_database, score_query,
)
from evaluation.bird_loader import load_bird_json
from evaluation.schema_context import _description_rows, select_schema_context
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.providers import OllamaProvider


def _q(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _safe_example(value: object) -> str | None:
    if value is None or isinstance(value, (bytes, bytearray, memoryview)):
        return None
    raw = " ".join(str(value).split())
    if not 1 <= len(raw) <= 40:
        return None
    return raw.replace("[", "(").replace("]", ")")


def build_mschema(
    *,
    catalog: DatabaseCatalog,
    database_id: str,
    selected_tables: list[str],
    descriptions: dict[tuple[str, str], str],
    max_sample_columns: int = 24,
    sample_values: int = 2,
) -> str:
    """Full column recall for selected lexical tables; M-Schema-inspired layout."""
    if not selected_tables or any(table not in catalog.tables for table in selected_tables):
        raise ValueError("selected schema tables are missing or unknown")
    if not 0 <= max_sample_columns <= 100 or not 0 <= sample_values <= 5:
        raise ValueError("bounded value samples required")

    lines = [f"Database: {database_id}", "Dialect: SQLite"]
    sampled_columns = 0
    with sqlite3.connect(catalog.database) as conn:
        for table in sorted(set(selected_tables)):
            lines.append(f"Table: {_q(table)}")
            info = conn.execute(f"PRAGMA table_info({_q(table)})").fetchall()
            if {str(row[1]) for row in info} != set(catalog.tables[table].columns):
                raise ValueError("SQLite schema changed between catalog and M-Schema")
            for row in info:
                col = str(row[1])
                col_type = str(row[2] or "UNKNOWN").upper()
                parts = [f"({_q(col)}:{col_type}"]
                if row[5]:
                    parts.append("Primary Key")
                desc = descriptions.get((table, col), "").strip()
                if desc:
                    parts.append("Description: " + " ".join(desc.split())[:120])
                if sample_values and sampled_columns < max_sample_columns and (
                    "CHAR" in col_type or "TEXT" in col_type or col_type in ("VARCHAR", "NVARCHAR")
                ):
                    seen: list[str] = []
                    for value in catalog.sample_values(table, col, limit=sample_values):
                        example = _safe_example(value)
                        if example is not None and example not in seen:
                            seen.append(example)
                    if seen:
                        parts.append("Examples: [" + ", ".join(seen) + "]")
                        sampled_columns += 1
                lines.append(", ".join(parts) + ")")
    edges = [
        (src, col, dst, dst_col)
        for src, col, dst, dst_col in catalog.foreign_keys
        if src in selected_tables and dst in selected_tables
    ]
    if edges:
        lines.append("Foreign keys:")
        lines.extend(f"{_q(src)}.{_q(col)} = {_q(dst)}.{_q(dst_col)}" for src, col, dst, dst_col in edges)
    return "\n".join(lines)


def run_shard(
    *,
    source: Path, database_root: Path, frozen: Path, baseline: Path,
    model: str, shard_index: int, num_shards: int,
) -> dict:
    if not 0 <= shard_index < num_shards or num_shards < 1:
        raise ValueError("invalid shard")
    prior = json.loads(frozen.read_text(encoding="utf-8"))
    ordered = [str(r["case_id"]) for r in prior["cases"]]
    if len(ordered) != 100 or len(set(ordered)) != 100:
        raise ValueError("frozen sample must contain 100 unique cases")
    selected = [case_id for idx, case_id in enumerate(ordered) if idx % num_shards == shard_index]
    cases, labels = load_bird_json(source)
    case_by_id = {str(c.case_id): c for c in cases}
    gold_by_id = {str(c.case_id): c for c in labels}
    baseline_sql = load_baseline_sql(baseline)
    if any(case_id not in baseline_sql for case_id in selected):
        raise ValueError("missing original frozen XiYan baseline SQL, refusing regeneration")
    provider = OllamaProvider(model=model, timeout_s=900.0)
    cache: dict[str, tuple[Path, DatabaseCatalog, dict[tuple[str,str],str]]] = {}
    rows = []
    for i, case_id in enumerate(selected, 1):
        case = case_by_id[case_id]
        label = gold_by_id[case_id]
        if case.database_id not in cache:
            path = resolve_database(database_root, case.database_id)
            catalog = DatabaseCatalog.from_sqlite(path)
            descriptions = {(t, c): d for t, c, d in _description_rows(database_root, case.database_id)}
            cache[case.database_id] = (path, catalog, descriptions)
        path, catalog, descriptions = cache[case.database_id]
        _, lexical = select_schema_context(catalog, database_root, case.database_id, case.question)
        mschema = build_mschema(
            catalog=catalog,
            database_id=case.database_id,
            selected_tables=lexical["selected_tables"],
            descriptions=descriptions,
        )
        start = time.perf_counter()
        candidate = generate_xiyan(
            question=case.question,
            schema_context=mschema,
            evidence=case.evidence.strip() or None,
            provider=provider,
        )
        elapsed = time.perf_counter() - start
        baseline_eval = score_query(path, baseline_sql[case_id], label.sql)
        treatment_eval = score_query(path, candidate, label.sql)
        rows.append({
            "case_id": case_id,
            "database_id": case.database_id,
            "baseline_sql_sha256": sha256(baseline_sql[case_id].encode()).hexdigest(),
            "mschema_sha256": sha256(mschema.encode()).hexdigest(),
            "mschema_characters": len(mschema),
            "candidate_sql": candidate,
            "generation_latency_s": elapsed,
            "baseline_execution_success": baseline_eval["execution_success"],
            "baseline_ex": baseline_eval["frozen_execution_match"],
            "mschema_execution_success": treatment_eval["execution_success"],
            "mschema_ex": treatment_eval["frozen_execution_match"],
            "mschema_truncated": treatment_eval["truncated"],
            "baseline_truncated": baseline_eval["truncated"],
        })
        print(
            f"[shard={shard_index} {i}/{len(selected)}] case={case_id} "
            f"lexical={baseline_eval['frozen_execution_match']} "
            f"mschema={treatment_eval['frozen_execution_match']}", flush=True,
        )
    return {
        "schema": "xiyan-mschema-frozen100-v1",
        "model": model,
        "num_shards": num_shards,
        "shard_index": shard_index,
        "cases": rows,
        "gold_access": "offline scoring only",
        "baseline": "reused frozen SQL from original XiYan 58/100 run",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_shard(
        source=args.source, database_root=args.database_root, frozen=args.frozen,
        baseline=args.baseline_results, model=args.model,
        shard_index=args.shard_index, num_shards=args.num_shards,
    )
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"WRITTEN shard={args.shard_index} rows={len(result['cases'])}")


if __name__ == "__main__":
    main()
