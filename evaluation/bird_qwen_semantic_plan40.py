"""One question-driven semantic-plan intervention on frozen BIRD Train40.

Two arms share Qwen3.5-9B, lexical schema, evidence, verifier and <=2
existing guarded repairs. Only the challenger generates a bounded
natural-language plan *before* SQL. No XiYan inference and no SQL training.
All gold SQL is read only after both arms finish a shard.
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

from evaluation.bird_a5_plan_transfer40 import plan_prompt, parse_plan
from evaluation.bird_generator_transfer import (
    MIRROR, DATASET, REVISION, TRAIN_SHA, check_database_bytes,
    load_checked, remote_manifest, select_cases,
)
from evaluation.run_bird import _ollama_model_digest
from evaluation.schema_context import select_schema_context
from evaluation.xiyan_qwen_factorial_fresh40 import (
    MODEL_IDS, QWEN_DIGEST, SOURCE_SHA256, generate,
)
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.execution import execute_readonly
from semantic_sql.inference import run_guarded

N_SHARDS = 10
ARMS = ("original_qwen_guarded", "planned_qwen_guarded")


class PinnedQwen:
    """Reuse the existing pinned Ollama generator for guarded repair too."""
    def complete_text(self, *, system: str, user: str) -> str:
        return generate(model=MODEL_IDS["qwen"], system=system, user=user)


def run_shard(train_path: Path, dev_path: Path, shard: int, work: Path) -> dict:
    if shard not in range(N_SHARDS):
        raise ValueError("invalid shard")
    train = load_checked(train_path, TRAIN_SHA)
    dev = load_checked(dev_path, SOURCE_SHA256)
    manifest = remote_manifest()
    cases = select_cases(train, dev, manifest, guarded=True)
    if len(cases) != 40:
        raise ValueError("frozen BIRD Train40 changed")
    chosen = cases[4 * shard:4 * (shard + 1)]
    if len(chosen) != 4:
        raise ValueError("incomplete shard")
    model_digest = _ollama_model_digest("http://127.0.0.1:11434", MODEL_IDS["qwen"])
    if model_digest != QWEN_DIGEST:
        raise ValueError("Qwen9B checkpoint changed")

    dbs = {}
    for dbid in sorted({r["database_id"] for r in chosen}):
        path = work / dbid / f"{dbid}.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{MIRROR}/datasets/{DATASET}/resolve/{REVISION}/raw/bird/train_databases"
        with urlopen(f"{url}/{quote(dbid)}/{quote(dbid)}.sqlite", timeout=120) as response:
            data = response.read(manifest[dbid]["size"] + 1)
        if not check_database_bytes(data, manifest[dbid]):
            raise ValueError("BIRD database digest changed")
        path.write_bytes(data)
        dbs[dbid] = (path, DatabaseCatalog.from_sqlite(path))

    provider = PinnedQwen()
    records = []
    for case in chosen:
        db, catalog = dbs[case["database_id"]]
        schema, _ = select_schema_context(catalog, work, case["database_id"], case["question"])
        evidence = case["evidence"] or None
        start = perf_counter()
        baseline = run_guarded(
            database=db, provider=provider, question=case["question"],
            schema_context=schema, evidence=evidence,
            max_repairs=2, max_rows=50_000,
        )
        raw_plan = provider.complete_text(
            system="Output only a concise JSON semantic plan, not SQL.",
            user=plan_prompt(case["question"], schema),
        )
        plan = parse_plan(raw_plan)
        grounded = schema
        if plan is not None:
            grounded += (
                "\nUnverified semantic plan. Check every item against the question; "
                "the original question wins over any plan disagreement:\n"
                + json.dumps(plan, sort_keys=True, ensure_ascii=False)
            )
        revised = run_guarded(
            database=db, provider=provider, question=case["question"],
            schema_context=grounded, evidence=evidence,
            max_repairs=2, max_rows=50_000,
        )
        rec = {
            "case_id": case["case_id"],
            "db": case["database_id"],
            "plan_valid": plan is not None,
            "changed_sql": baseline.candidate != revised.candidate,
            "latency_s": round(perf_counter() - start, 3),
            "arms": {},
        }
        for name, result in zip(ARMS, (baseline, revised)):
            good = result.status == "ok" and not result.truncated
            rec["arms"][name] = {
                "rows": result.rows if good else (),
                "execution_success": good,
                "attempt": result.candidate.attempt if result.candidate else -1,
            }
        records.append(rec)
        print("QWEN_SEMANTIC_PLAN_SHARD", shard, case["case_id"], flush=True)

    # Gold cannot affect generation or candidate selection.
    for rec in records:
        db, _ = dbs[rec["db"]]
        gold = execute_readonly(db, train[rec["case_id"]]["SQL"], max_rows=50_000)
        if gold.truncated:
            raise ValueError("truncated gold SQL result")
        for arm in rec["arms"].values():
            rows = arm.pop("rows")
            arm["ex"] = bool(arm["execution_success"] and set(rows) == set(gold.rows))
            arm["bag"] = bool(arm["execution_success"] and Counter(rows) == Counter(gold.rows))
    return {
        "shard": shard,
        "digest": model_digest,
        "cohort_hash": sha256(json.dumps(
            [(x["case_id"], x["database_id"]) for x in cases],
            separators=(",", ":"),
        ).encode()).hexdigest(),
        "cases": records,
    }


def aggregate(shards: list[dict]) -> dict:
    if len(shards) != N_SHARDS or sorted(x["shard"] for x in shards) != list(range(N_SHARDS)):
        raise ValueError("ten unique completed shards required")
    if len({x["digest"] for x in shards}) != 1 or len({x["cohort_hash"] for x in shards}) != 1:
        raise ValueError("model or cohort changed")
    cases = [r for s in shards for r in s["cases"]]
    if len(cases) != 40 or len({r["case_id"] for r in cases}) != 40:
        raise ValueError("need 40 unique cases")
    before, after = ARMS
    rescues = sum(not r["arms"][before]["ex"] and r["arms"][after]["ex"] for r in cases)
    regressions = sum(r["arms"][before]["ex"] and not r["arms"][after]["ex"] for r in cases)
    result = {
        "n": len(cases), "model": "Qwen3.5-9B", "model_digest": shards[0]["digest"],
        "cohort_hash": shards[0]["cohort_hash"],
        "arms": {arm: {
            "ex": sum(r["arms"][arm]["ex"] for r in cases),
            "bag": sum(r["arms"][arm]["bag"] for r in cases),
            "execution_success": sum(r["arms"][arm]["execution_success"] for r in cases),
            "repairs": sum(max(0, r["arms"][arm]["attempt"]) for r in cases),
        } for arm in ARMS},
        "rescues": rescues, "regressions": regressions,
        "net": rescues - regressions,
        "valid_plans": sum(r["plan_valid"] for r in cases),
        "mean_total_latency_s": round(sum(r["latency_s"] for r in cases) / 40, 3),
        "no_training": True, "no_gold_routing": True,
        "boundary": "Previously explored BIRD Train40 / 20 DBs; not pristine holdout or DEV100 improvement",
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path)
    parser.add_argument("--dev", type=Path)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--work", type=Path, default=Path("qwen-plan-work"))
    parser.add_argument("--inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.inputs:
        result = aggregate([json.loads(p.read_text()) for p in args.inputs])
        print("QWEN_SEMANTIC_PLAN_FINAL", json.dumps(result, sort_keys=True), flush=True)
    else:
        if args.train is None or args.dev is None or args.shard is None:
            parser.error("--train, --dev, and --shard required")
        result = run_shard(args.train, args.dev, args.shard, args.work)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
