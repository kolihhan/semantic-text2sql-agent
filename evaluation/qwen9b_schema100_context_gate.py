"""Source-only transfer feasibility gate for original Qwen3.5-9B BIRD100.

Reuses byte-for-byte fixed CamelCase schema tokenizer in PR #41.
Counts how many of the ORIGINAL 100 questions have changed model-visible
schema context, before any model inference or access to reference SQL.
Never claims an EX gain from schema selection alone.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from evaluation.qwen_schema_grounding_pair import checked_source, contexts, case_id
from evaluation.run_bird import _resolve_database
from semantic_sql.catalog import DatabaseCatalog

MODEL = "qwen3.5:9b-q4_K_M"
MODEL_DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
ORIGINAL_IDS = 100


def select_affected(source: Path, root: Path, reference: Path) -> dict:
    rows = checked_source(source)
    frozen = json.loads(reference.read_text())
    old = frozen.get("cases")
    if frozen.get("status") != "complete" or frozen.get("sample_size") != ORIGINAL_IDS or len(old or []) != ORIGINAL_IDS:
        raise ValueError("original 100-case cohort missing")
    by_id = {case_id(item, i): item for i, item in enumerate(rows)}
    if len(by_id) != len(rows):
        raise ValueError("source cases not uniquely indexed")
    catalog_cache = {}
    affected, unchanged = [], []
    case_ids = set()
    for previous in old:
        ident = str(previous["case_id"])
        if ident in case_ids:
            raise ValueError("duplicate frozen original BIRD case id")
        case_ids.add(ident)
        row = by_id.get(ident)
        if (row is None or row.get("db_id") != previous.get("database_id")
            or row.get("question") != previous.get("question")):
            raise ValueError(f"original question/database identity drift: {ident}")
        db = str(row["db_id"])
        if db not in catalog_cache:
            catalog_cache[db] = DatabaseCatalog.from_sqlite(_resolve_database(root, db))
        old_ctx, new_ctx = contexts(catalog_cache[db], root, db, str(row["question"]))
        item = {
            "case_id": ident, "database_id": db,
            "legacy_context_sha256": old_ctx[1]["schema_context_sha256"],
            "fixed_context_sha256": new_ctx[1]["schema_context_sha256"],
            "original_selected_tables": old_ctx[1]["selected_tables"],
            "fixed_selected_tables": new_ctx[1]["selected_tables"],
        }
        (affected if old_ctx[0] != new_ctx[0] else unchanged).append(item)
    if len(case_ids) != ORIGINAL_IDS or len(affected) + len(unchanged) != ORIGINAL_IDS:
        raise AssertionError("cohort coverage changed")
    affected_dbs = Counter(x["database_id"] for x in affected)
    return {
        "protocol": "Qwen9B-original100-schema-tokenizer-transfer-source-preflight-v1",
        "unchanged_model": MODEL, "unchanged_model_digest": MODEL_DIGEST,
        "source_sha256": sha256(source.read_bytes()).hexdigest(),
        "original_100_case_ids_sha256": sha256(
            json.dumps(sorted(case_ids), separators=(",", ":")).encode()).hexdigest(),
        "total_cases": ORIGINAL_IDS,
        "affected_contexts": len(affected),
        "affected_databases": len(affected_dbs),
        "affected_by_database": dict(sorted(affected_dbs.items())),
        "unchanged_cases": len(unchanged),
        "affected": affected,
        "gold_used_to_select": False,
        "model_calls": 0,
        "no_model_swap": True, "no_SQL_training": True,
        "BIRD100_original_EX": 39,
        "BIRD100_new_EX": None,
        "gate_for_paired_generation": (
            "eligible" if len(affected) >= 3 and len(affected_dbs) >= 3 else
            "reject_insufficient_original100_affected_diversity"
        ),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--reference", default=Path("runs/bird-paired-dev-v5/paired.json"), type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("refuse overwrite original100 gate")
    result = select_affected(args.source, args.root, args.reference)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2)+"\n")
    print("QWEN9B_ORIGINAL_BIRD100_SCHEMA_TRANSFER_FEASIBILITY", json.dumps(
        {k:v for k,v in result.items() if k!="affected"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
