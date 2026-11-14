"""Frozen, matched BIRD DEV100 generator comparison; no model training or gold-based selection.

Qwen uses the original guarded generation/repair path. XiYan uses its native prompt,
followed by identical read-only preflight/execution without repair.
This is a retrospective DEV comparison, not a clean hidden-test estimate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from evaluation.bird_generator_transfer import (
    XIYAN_DIGEST, _ModelProvider, guarded_qwen,
)
from evaluation.run_bird import _ollama_model_digest, _resolve_database
from evaluation.schema_context import select_schema_context
from evaluation.xiyan_qwen_factorial_fresh40 import (
    MODEL_IDS, QWEN_DIGEST, generate, prompt_for,
)
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.contracts import SQLCandidate
from semantic_sql.execution import execute_readonly
from semantic_sql.inference import run_guarded

SOURCE_SHA = "ffd8018378ddb1a8794753e0a31cfc81862ff7318a5184c22f3dc4ce03a03feb"
REFERENCE_SOURCE_SHA = "0310b7ca2dfb6247234e2073f93e7a75106ee41ba55231f75d06d2ef68efe2ec"
N_SHARDS = 10


def load_cohort(source_path: Path, reference_path: Path) -> tuple[list[dict], list[dict]]:
    if sha256(source_path.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError("source version drift")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if (len(source) != 1534 or reference.get("status") != "complete"
        or reference.get("sample_size") != 100
        or reference.get("source_sha256") != REFERENCE_SOURCE_SHA):
        raise ValueError("frozen reference identity mismatch")
    indexed = {str(r.get("question_id") or r.get("id") or i): r
               for i, r in enumerate(source)}
    if len(indexed) != len(source):
        raise ValueError("duplicate BIRD source IDs")
    frozen = []
    for ref in reference["cases"]:
        case_id = str(ref["case_id"])
        row = indexed.get(case_id)
        if row is None or str(row["db_id"]) != str(ref["database_id"]) or row["question"] != ref["question"]:
            raise ValueError(f"frozen question drift: {case_id}")
        frozen.append({"case_id": case_id, "database_id": str(row["db_id"]),
                       "question": row["question"], "evidence": row.get("evidence") or ""})
    if len(frozen) != 100 or len({r["case_id"] for r in frozen}) != 100:
        raise ValueError("incomplete frozen cohort")
    return source, frozen


def digest_cohort(rows: list[dict]) -> str:
    return sha256(json.dumps(
        [(r["case_id"], r["database_id"], r["question"]) for r in rows],
        ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def run_shard(source_path: Path, reference_path: Path, db_root: Path, shard: int) -> dict:
    if shard not in range(N_SHARDS):
        raise ValueError("invalid shard")
    source, cohort = load_cohort(source_path, reference_path)
    chosen = cohort[shard::N_SHARDS]
    if len(chosen) != 10:
        raise ValueError("must evaluate ten matched cases per shard")
    actual = {key: _ollama_model_digest("http://127.0.0.1:11434", model)
              for key, model in MODEL_IDS.items()}
    expected = {"qwen": QWEN_DIGEST, "xiyan": XIYAN_DIGEST}
    if actual != expected:
        raise ValueError(f"model digest drift: {actual}")
    dbs = {}
    work = []
    for item in chosen:
        dbid = item["database_id"]
        if dbid not in dbs:
            path = _resolve_database(db_root, dbid)
            dbs[dbid] = (path, DatabaseCatalog.from_sqlite(path))
        path, catalog = dbs[dbid]
        schema, meta = select_schema_context(catalog, db_root, dbid, item["question"])
        work.append({**item, "schema_sha256": meta["schema_context_sha256"],
                     "_schema": schema, "arms": {}})
    # Batch by model; keep the same physical DB, question, schema, and evidence.
    # Gold SQL is not accessed until both candidate arms have completed.
    for model_key in ("qwen", "xiyan"):
        for row in work:
            db, _ = dbs[row["database_id"]]
            evidence = row["evidence"].strip() or None
            if model_key == "qwen":
                arm = guarded_qwen(db, row["question"], row["_schema"], evidence)
            else:
                system, user = prompt_for(
                    condition="xiyan", question=row["question"],
                    schema_context=row["_schema"], evidence=evidence)
                begin = perf_counter()
                sql = generate(model=MODEL_IDS["xiyan"], system=system, user=user)
                result = run_guarded(
                    database=db, provider=_ModelProvider(MODEL_IDS["xiyan"], generate),
                    question=row["question"], schema_context=row["_schema"],
                    evidence=evidence, initial_candidate=SQLCandidate(sql),
                    max_repairs=0, max_rows=50_000,
                )
                arm = {"sql": sql, "model_calls": 1,
                       "latency_s": round(perf_counter() - begin, 3), "result": result}
            row["arms"][model_key] = arm
            print("GENERATED", shard, row["case_id"], model_key, flush=True)

    by_id = {str(r.get("question_id") or r.get("id") or i): r
             for i, r in enumerate(source)}
    for row in work:
        row.pop("_schema")
        db, _ = dbs[row["database_id"]]
        gold = execute_readonly(db, by_id[row["case_id"]]["SQL"], max_rows=50_000)
        if gold.truncated:
            raise ValueError(f"gold truncated {row['case_id']}")
        for arm in row["arms"].values():
            result = arm.pop("result")
            executed = result.status == "ok" and not result.truncated
            arm.update(
                ex=bool(executed and set(result.rows) == set(gold.rows)),
                bag_equal=bool(executed and Counter(result.rows) == Counter(gold.rows)),
                execution_success=executed, preflight=result.verification.ok,
                stop_reason="truncated" if result.truncated else result.status,
            )
    return {"shard": shard, "count": 10, "cohort_sha256": digest_cohort(cohort),
            "models": actual, "cases": work}


def aggregate(inputs: list[dict]) -> dict:
    if len(inputs) != N_SHARDS or sorted(s["shard"] for s in inputs) != list(range(N_SHARDS)):
        raise ValueError("missing or duplicate shard")
    first = inputs[0]
    if any(s["cohort_sha256"] != first["cohort_sha256"] or s["models"] != first["models"]
           or s["count"] != 10 or len(s["cases"]) != 10 for s in inputs):
        raise ValueError("cohort/model/protocol drift")
    rows = [r for shard in inputs for r in shard["cases"]]
    if len(rows) != 100 or len({r["case_id"] for r in rows}) != 100:
        raise ValueError("missing/duplicate frozen cases")
    stats = {}
    for key in ("qwen", "xiyan"):
        arms = [r["arms"][key] for r in rows]
        stats[key] = {
            "ex": sum(r["ex"] for r in arms),
            "bag_equal": sum(r["bag_equal"] for r in arms),
            "execution_success": sum(r["execution_success"] for r in arms),
            "preflight": sum(r["preflight"] for r in arms),
            "model_calls": sum(r["model_calls"] for r in arms),
            "mean_latency_s": round(sum(r["latency_s"] for r in arms)/100, 3),
        }
    def paired(metric: str) -> tuple[int, int]:
        return (
            sum(r["arms"]["xiyan"][metric] and not r["arms"]["qwen"][metric] for r in rows),
            sum(r["arms"]["qwen"][metric] and not r["arms"]["xiyan"][metric] for r in rows),
        )
    return {
        "count": 100, "cohort_sha256": first["cohort_sha256"],
        "models": first["models"],
        "protocol": "original Qwen guarded vs XiYan native+no repair; same frozen DEV100",
        "arms": stats, "ex_rescues_regressions": paired("ex"),
        "bag_rescues_regressions": paired("bag_equal"), "cases": rows,
        "claim_boundary": "public BIRD DEV; retrospective cohort; no hidden-set or production claim",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path)
    p.add_argument("--reference", type=Path, default=Path("runs/bird-paired-dev-v5/paired.json"))
    p.add_argument("--database-root", type=Path)
    p.add_argument("--shard", type=int)
    p.add_argument("--inputs", type=Path, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.inputs:
        result = aggregate([json.loads(p.read_text()) for p in args.inputs])
        print(json.dumps({k: v for k, v in result.items() if k != "cases"}), flush=True)
    else:
        if args.source is None or args.database_root is None or args.shard is None:
            p.error("--source, --database-root, --shard required")
        result = run_shard(args.source, args.reference, args.database_root, args.shard)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
