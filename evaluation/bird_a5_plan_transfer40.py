"""Frozen A5 semantic-plan transfer test: independent BIRD Train40 / 20 DBs.

Reuse the exact A5 synthetic-study prompt and same native XiYan generator.
Only the A5 arm adds an independently generated bounded semantic plan to
the lexical schema context. No gold or answer-dependent routing.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from urllib.parse import quote
from urllib.request import urlopen

from evaluation.bird_generator_transfer import (
    MIRROR, DATASET, REVISION, TRAIN_SHA, XIYAN_DIGEST,
    check_database_bytes, load_checked, remote_manifest, select_cases,
)
from evaluation.run_bird import _ollama_model_digest
from evaluation.schema_context import select_schema_context
from evaluation.xiyan_qwen_factorial_fresh40 import MODEL_IDS, SOURCE_SHA256, generate, prompt_for
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.contracts import SQLCandidate
from semantic_sql.execution import execute_readonly
from semantic_sql.inference import run_guarded

ARMS = ("a0", "a5")
PLAN_KEYS = ("answer_entity", "relation_role", "aggregation_grain", "filters")
SHARDS = 10


def plan_prompt(question: str, schema: str) -> str:
    # A5 v1 frozen English plan, including adversarial role/duplication rules.
    return (
        "Create a short semantic query PLAN, not an SQL query. "
        "Use EXACTLY a single JSON object with four string fields: "
        "answer_entity, relation_role, aggregation_grain, filters. "
        "For any role or value not determined by the question and schema, "
        "write UNKNOWN. Distinguish who INITIATED a relation from who RECEIVED it; "
        "count entities with DISTINCT when a join can duplicate them; "
        "do not replace AND by OR; do not change explicit numeric thresholds. "
        "Never write SQL, never invent table names, and never claim to know gold SQL.\n"
        f"USER QUESTION:\n{question}\n\nSQLITE DATABASE SCHEMA:\n{schema}\n\n"
        "RETURN JSON ONLY:"
    )


def parse_plan(raw: str) -> dict | None:
    text = raw.strip()
    if len(text) > 4000:
        return None
    try:
        item = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(item, dict) or set(item) != set(PLAN_KEYS):
        return None
    if any(not isinstance(item[key], str) or not item[key].strip()
           or len(item[key]) > 180 for key in PLAN_KEYS):
        return None
    return item


def run_shard(train_path: Path, dev_path: Path, shard: int, work: Path) -> dict:
    if shard not in range(SHARDS):
        raise ValueError("invalid BIRD Train40 shard")
    train = load_checked(train_path, TRAIN_SHA)
    dev = load_checked(dev_path, SOURCE_SHA256)
    manifest = remote_manifest()
    cohort = select_cases(train, dev, manifest, guarded=True)
    if len(cohort) != 40:
        raise ValueError("frozen train cohort changed")
    selected = cohort[4 * shard:4 * (shard + 1)]
    if len(selected) != 4:
        raise ValueError("need exactly four questions")
    model = MODEL_IDS["xiyan"]
    digest = _ollama_model_digest("http://127.0.0.1:11434", model)
    if digest != XIYAN_DIGEST:
        raise ValueError("XiYan digest drift")
    dbs = {}
    for dbid in sorted({r["database_id"] for r in selected}):
        path = work / dbid / f"{dbid}.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{MIRROR}/datasets/{DATASET}/resolve/{REVISION}/raw/bird/train_databases"
        with urlopen(f"{url}/{quote(dbid)}/{quote(dbid)}.sqlite", timeout=120) as response:
            data = response.read(manifest[dbid]["size"] + 1)
        if not check_database_bytes(data, manifest[dbid]):
            raise ValueError("BIRD database checksum mismatch")
        path.write_bytes(data)
        dbs[dbid] = (path, DatabaseCatalog.from_sqlite(path))

    records = []
    # Generate every candidate first; only then access reference SQL.
    for item in selected:
        dbid = item["database_id"]
        db, catalog = dbs[dbid]
        schema, _ = select_schema_context(catalog, work, dbid, item["question"])
        evidence = item["evidence"] or None
        start = perf_counter()
        sys, usr = prompt_for(
            condition="xiyan", question=item["question"],
            schema_context=schema, evidence=evidence,
        )
        baseline = generate(model=model, system=sys, user=usr)
        raw_plan = generate(
            model=model, system="Only JSON plans, no SQL.",
            user=plan_prompt(item["question"], schema),
        )
        plan = parse_plan(raw_plan)
        if plan is None:
            challenger = baseline
        else:
            grounded = (
                schema + "\nProposed question-semantic plan (NOT a fact, "
                "verify every assumption against the user's question):\n"
                + json.dumps(plan, sort_keys=True, ensure_ascii=False)
            )
            system2, user2 = prompt_for(
                condition="xiyan", question=item["question"],
                schema_context=grounded, evidence=evidence,
            )
            challenger = generate(model=model, system=system2, user=user2)
        elapsed = round(perf_counter() - start, 3)
        arms = {}
        for key, sql in (("a0", baseline), ("a5", challenger)):
            result = run_guarded(
                database=db, provider=None, question=item["question"],
                schema_context=schema, evidence=evidence,
                initial_candidate=SQLCandidate(sql),
                max_repairs=0, max_rows=50_000,
            )
            arms[key] = {
                "sql": sql, "rows": result.rows if result.status == "ok" and not result.truncated else (),
                "execution_success": result.status == "ok" and not result.truncated,
                "preflight": result.verification.ok,
            }
        records.append({
            "case_id": item["case_id"], "database_id": dbid,
            "question": item["question"], "plan_valid": plan is not None,
            "generation_latency_s": elapsed, "arms": arms,
        })
        print("A5_TRANSFER_GENERATED", shard, item["case_id"], flush=True)

    for record in records:
        db, _ = dbs[record["database_id"]]
        gold = execute_readonly(
            db, train[record["case_id"]]["SQL"], max_rows=50_000,
        )
        if gold.truncated:
            raise ValueError("truncated reference rows")
        for arm in record["arms"].values():
            result_rows = arm.pop("rows")
            arm["ex"] = bool(arm["execution_success"] and set(result_rows) == set(gold.rows))
            arm["bag"] = bool(arm["execution_success"] and Counter(result_rows) == Counter(gold.rows))
    return {
        "shard": shard, "model": digest,
        "cohort_sha256": sha256(json.dumps([
            [r["case_id"], r["database_id"]] for r in cohort
        ], separators=(",", ":")).encode()).hexdigest(),
        "cases": records,
    }


def aggregate(shards: list[dict]) -> dict:
    if len(shards) != SHARDS or sorted(x["shard"] for x in shards) != list(range(SHARDS)):
        raise ValueError("need ten unique shards")
    first = shards[0]
    if any(s["model"] != first["model"]
           or s["cohort_sha256"] != first["cohort_sha256"]
           or len(s["cases"]) != 4 for s in shards):
        raise ValueError("cohort/model identity mismatch")
    rows = [r for s in sorted(shards, key=lambda x: x["shard"]) for r in s["cases"]]
    if len(rows) != 40 or len({r["case_id"] for r in rows}) != 40:
        raise ValueError("incomplete 40 questions")
    stats = {
        arm: {
            "ex": sum(r["arms"][arm]["ex"] for r in rows),
            "bag": sum(r["arms"][arm]["bag"] for r in rows),
            "execution_success": sum(r["arms"][arm]["execution_success"] for r in rows),
        } for arm in ARMS
    }
    rescues = sum(not r["arms"]["a0"]["ex"] and r["arms"]["a5"]["ex"] for r in rows)
    regressions = sum(r["arms"]["a0"]["ex"] and not r["arms"]["a5"]["ex"] for r in rows)
    return {
        "count": 40, "model": first["model"],
        "cohort_sha256": first["cohort_sha256"],
        "arms": stats, "rescues": rescues, "regressions": regressions,
        "net": rescues - regressions,
        "valid_plans": sum(r["plan_valid"] for r in rows),
        "changed_sql": sum(r["arms"]["a0"]["sql"] != r["arms"]["a5"]["sql"] for r in rows),
        "mean_total_generation_s": round(sum(r["generation_latency_s"] for r in rows) / 40, 3),
        "boundary": "previously studied BIRD Train40; not a pristine holdout or a BIRD DEV uplift",
        "cases": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path)
    p.add_argument("--dev", type=Path)
    p.add_argument("--shard", type=int)
    p.add_argument("--work", type=Path, default=Path("a5-transfer-work"))
    p.add_argument("--inputs", type=Path, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.inputs:
        result = aggregate([json.loads(p.read_text()) for p in args.inputs])
        print("A5_TRANSFER_FINAL", json.dumps({k:v for k,v in result.items() if k != "cases"}), flush=True)
    else:
        if args.train is None or args.dev is None or args.shard is None:
            p.error("--train, --dev, --shard required")
        result = run_shard(args.train, args.dev, args.shard, args.work)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
