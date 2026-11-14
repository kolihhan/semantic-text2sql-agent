"""Frozen BIRD TRAIN 40 question three-arm XiYan schema representation ablation.

Same XiYan GGUF, native prompt, seed, execution harness, exact DB/question
identity; only schema presentation changes: lexical, typed M-Schema-like, or
typed M-Schema-like with bounded query-relevant samples. No training or gold
access until generation and read-only execution of all arms are complete.
This cohort was already used for prior generator transfer, so the result
is a mechanism ablation, not a pristine holdout.
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
from evaluation.mschema_like import to_mschema_like
from evaluation.run_bird import _ollama_model_digest
from evaluation.schema_context import select_schema_context
from evaluation.xiyan_qwen_factorial_fresh40 import (
    MODEL_IDS, SOURCE_SHA256, generate, prompt_for,
)
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.contracts import SQLCandidate
from semantic_sql.execution import execute_readonly
from semantic_sql.inference import run_guarded

ARMS = ("lexical", "mschema_types", "mschema_examples")
N_SHARDS = 10


def run_shard(train_path: Path, dev_path: Path, shard: int, work: Path) -> dict:
    if shard not in range(N_SHARDS):
        raise ValueError("invalid BIRD Train40 shard")
    train = load_checked(train_path, TRAIN_SHA)
    dev = load_checked(dev_path, SOURCE_SHA256)
    manifest = remote_manifest()
    cohort = select_cases(train, dev, manifest, guarded=True)
    if len(cohort) != 40 or len({r["case_id"] for r in cohort}) != 40:
        raise ValueError("frozen BIRD Train cohort drift")
    selected = cohort[4 * shard:4 * (shard + 1)]
    if len(selected) != 4:
        raise ValueError("need exactly four cases per shard")
    digest = _ollama_model_digest("http://127.0.0.1:11434", MODEL_IDS["xiyan"])
    if digest != XIYAN_DIGEST:
        raise ValueError(f"XiYan model drift: {digest}")
    dbs = {}
    for db_id in sorted({r["database_id"] for r in selected}):
        path = work / db_id / f"{db_id}.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{MIRROR}/datasets/{DATASET}/resolve/{REVISION}/raw/bird/train_databases"
        with urlopen(f"{url}/{quote(db_id)}/{quote(db_id)}.sqlite", timeout=120) as response:
            contents = response.read(manifest[db_id]["size"] + 1)
        if not check_database_bytes(contents, manifest[db_id]):
            raise ValueError("physical SQLite database checksum drift")
        path.write_bytes(contents)
        dbs[db_id] = (path, DatabaseCatalog.from_sqlite(path))
    rows = []
    for row in selected:
        dbid = row["database_id"]
        _, catalog = dbs[dbid]
        lexical, meta = select_schema_context(catalog, work, dbid, row["question"])
        presentations = {
            "lexical": lexical,
            "mschema_types": to_mschema_like(
                catalog, db_id=dbid, selected_tables=meta["selected_tables"],
                question=row["question"], lexical_context=lexical, with_examples=False,
            ),
            "mschema_examples": to_mschema_like(
                catalog, db_id=dbid, selected_tables=meta["selected_tables"],
                question=row["question"], lexical_context=lexical, with_examples=True,
            ),
        }
        rows.append({
            **{k: row[k] for k in ("case_id", "database_id", "question")},
            "evidence": row["evidence"],
            "_presentations": presentations,
            "present_chars": {a: len(v) for a, v in presentations.items()},
            "selection_hash": meta["schema_context_sha256"],
            "arms": {},
        })
    for arm in ARMS:
        for row in rows:
            db, _ = dbs[row["database_id"]]
            system, user = prompt_for(
                condition="xiyan", question=row["question"],
                schema_context=row["_presentations"][arm], evidence=row["evidence"] or None,
            )
            begin = perf_counter()
            sql = generate(model=MODEL_IDS["xiyan"], system=system, user=user)
            result = run_guarded(
                database=db,
                provider=None,  # zero-repair path uses initial candidate only
                question=row["question"], schema_context=row["_presentations"][arm],
                evidence=row["evidence"] or None, initial_candidate=SQLCandidate(sql),
                max_repairs=0, max_rows=50_000,
            )
            row["arms"][arm] = {
                "sql": sql,
                "execution_success": result.status == "ok" and not result.truncated,
                "preflight": result.verification.ok,
                "rows": result.rows if result.status == "ok" and not result.truncated else (),
                "latency_s": round(perf_counter() - begin, 3),
            }
            print("MSCHEMA_GENERATED", shard, row["case_id"], arm, flush=True)

    # Score only after all three candidates are fixed.
    for row in rows:
        row.pop("_presentations")
        db, _ = dbs[row["database_id"]]
        gold = execute_readonly(db, train[row["case_id"]]["SQL"], max_rows=50_000)
        if gold.truncated:
            raise ValueError("BIRD gold rows truncated")
        for arm in ARMS:
            result = row["arms"][arm]
            out = result.pop("rows")
            result["ex"] = bool(result["execution_success"] and set(out) == set(gold.rows))
            result["bag_equal"] = bool(result["execution_success"] and Counter(out) == Counter(gold.rows))
    return {
        "shard": shard, "cases": rows, "model": digest,
        "cohort_sha256": sha256(json.dumps([
            [r["case_id"], r["database_id"]] for r in cohort
        ], separators=(",", ":")).encode()).hexdigest(),
    }


def aggregate(shards: list[dict]) -> dict:
    if len(shards) != N_SHARDS or sorted(s["shard"] for s in shards) != list(range(N_SHARDS)):
        raise ValueError("ten unique shards needed")
    first = shards[0]
    if any(s["model"] != first["model"] or s["cohort_sha256"] != first["cohort_sha256"]
           or len(s["cases"]) != 4 for s in shards):
        raise ValueError("cohort or model drift")
    rows = [r for s in sorted(shards, key=lambda s: s["shard"]) for r in s["cases"]]
    if len(rows) != 40 or len({r["case_id"] for r in rows}) != 40:
        raise ValueError("incomplete 40-case cohort")
    if any(set(r["arms"]) != set(ARMS) for r in rows):
        raise ValueError("arms missing")
    stats = {}
    for arm in ARMS:
        vals = [r["arms"][arm] for r in rows]
        stats[arm] = {
            "ex": sum(r["ex"] for r in vals),
            "bag": sum(r["bag_equal"] for r in vals),
            "execution_success": sum(r["execution_success"] for r in vals),
            "preflight": sum(r["preflight"] for r in vals),
            "mean_latency_s": round(sum(r["latency_s"] for r in vals)/40, 3),
            "mean_schema_chars": round(sum(r["present_chars"][arm] for r in rows)/40, 1),
        }
    pairs = {}
    for base, challenger in (
        ("lexical", "mschema_types"),
        ("mschema_types", "mschema_examples"),
        ("lexical", "mschema_examples"),
    ):
        pairs[f"{base}->{challenger}"] = {
            "rescues": sum(not r["arms"][base]["ex"] and r["arms"][challenger]["ex"] for r in rows),
            "regressions": sum(r["arms"][base]["ex"] and not r["arms"][challenger]["ex"] for r in rows),
        }
    return {
        "count": 40, "model_digest": first["model"],
        "cohort_sha256": first["cohort_sha256"],
        "schema_only_intervention": True,
        "gold_available_only_after_generation": True,
        "arms": stats, "paired": pairs,
        "claim_boundary": "BIRD Train40 used before; no pristine holdout or default promotion",
        "cases": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path)
    p.add_argument("--dev", type=Path)
    p.add_argument("--shard", type=int)
    p.add_argument("--work", type=Path, default=Path("bird-mschema-work"))
    p.add_argument("--inputs", type=Path, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.inputs:
        result = aggregate([json.loads(path.read_text()) for path in args.inputs])
        print("MSCHEMA_AGG", json.dumps({k:v for k,v in result.items() if k != "cases"}), flush=True)
    else:
        if args.train is None or args.dev is None or args.shard is None:
            p.error("--train, --dev and --shard are required")
        result = run_shard(args.train, args.dev, args.shard, args.work)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
