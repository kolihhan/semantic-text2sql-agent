from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "evaluation", ROOT / "experiments"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bird_loader import load_bird_json
from schema_context import select_schema_context
from frozen100_xiyan_eval import exact_score, resolve_database
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.contracts import SQLCandidate
from semantic_sql.inference import run_guarded
from semantic_sql.providers import OllamaProvider


def _semantic_stage(result):
    for stage in result.stages:
        if stage.name == "semantic_revision":
            return stage
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--database-root", required=True)
    parser.add_argument("--xiyan-inputs", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.source)
    database_root = Path(args.database_root)
    inputs = json.loads(Path(args.xiyan_inputs).read_text(encoding="utf-8"))
    input_rows = inputs["cases"]
    selected = [
        row for index, row in enumerate(input_rows)
        if index % args.num_shards == args.shard_index
    ]

    cases, labels = load_bird_json(source)
    case_by_id = {str(c.case_id): c for c in cases}
    label_by_id = {str(l.case_id): l for l in labels}

    provider = OllamaProvider(model=args.model, timeout_s=900.0)
    catalogs: dict[str, tuple[Path, DatabaseCatalog]] = {}
    output_rows: list[dict[str, object]] = []

    for index, base in enumerate(selected, start=1):
        case_id = str(base["case_id"])
        case = case_by_id[case_id]
        label = label_by_id[case_id]
        if case.database_id not in catalogs:
            database = resolve_database(database_root, case.database_id)
            catalogs[case.database_id] = (database, DatabaseCatalog.from_sqlite(database))
        database, catalog = catalogs[case.database_id]
        schema_context, _ = select_schema_context(catalog, database_root, case.database_id, case.question)
        evidence = case.evidence.strip() or None
        xiyan_sql = str(base["xiyan_sql"])
        baseline = exact_score(database, xiyan_sql, label.sql)

        started = time.perf_counter()
        result = run_guarded(
            database=database,
            provider=provider,
            question=case.question,
            schema_context=schema_context,
            evidence=evidence,
            initial_candidate=SQLCandidate(sql=xiyan_sql, attempt=0),
            max_repairs=0,
            max_rows=50_000,
            semantic_revision=True,
        )
        latency_s = time.perf_counter() - started
        revised_sql = result.candidate.sql if result.candidate is not None else xiyan_sql
        treatment = exact_score(database, revised_sql, label.sql)
        semantic = _semantic_stage(result)
        summary = semantic.summary if semantic is not None else "NOT_RUN: changed=false"
        issue_type = summary.split(":", 1)[0]
        changed = "changed=true" in summary

        row = {
            "case_id": case_id,
            "database_id": case.database_id,
            "question": case.question,
            "xiyan_sql": xiyan_sql,
            "xiyan_execution_success": baseline["execution_success"],
            "xiyan_official_ex": baseline["official_ex"],
            "cgsr_sql": revised_sql,
            "cgsr_status": result.status,
            "cgsr_execution_success": treatment["execution_success"],
            "cgsr_official_ex": treatment["official_ex"],
            "cgsr_review_ran": semantic is not None,
            "cgsr_changed": changed,
            "cgsr_issue_type": issue_type,
            "cgsr_latency_s": latency_s,
        }
        output_rows.append(row)
        print(
            f"[CGSR shard {args.shard_index} {index:02d}/{len(selected)}] "
            f"case={case_id} {baseline['official_ex']}->{treatment['official_ex']} "
            f"issue={issue_type} changed={changed}",
            flush=True,
        )

    counts = Counter(str(row["cgsr_issue_type"]) for row in output_rows)
    payload = {
        "experiment": "frozen-100 XiYan + one-pass CGSR paired ablation",
        "model": args.model,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "gold_visible_to_model": False,
        "max_repairs": 0,
        "semantic_revision": True,
        "issue_counts": dict(counts),
        "cases": output_rows,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
