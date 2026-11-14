"""True end-to-end XiYan-first + conditional Qwen Guarded on frozen BIRD DEV100.

The gold SQL is never read during model generation or route selection. This
reuses the shipped service, benchmark selection, schema grounding and SQL
execution. Public DEV100 is not a pristine hidden validation set.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from evaluation.run_bird import _ollama_model_digest, _resolve_database, CountingProvider
from evaluation.schema_context import select_schema_context
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.execution import execute_readonly
from semantic_sql.providers import OllamaProvider
from semantic_sql.xiyan_first import XiYanFirstSQLService

N_SHARDS = 10
DEV_SHA = "ffd8018378ddb1a8794753e0a31cfc81862ff7318a5184c22f3dc4ce03a03feb"
REFERENCE_SHA = "0310b7ca2dfb6247234e2073f93e7a75106ee41ba55231f75d06d2ef68efe2ec"
MODELS = {
    "xiyan": ("hf.co/wanhin/XiYanSQL-QwenCoder-7B-2504-gguf:Q4_K_M",
              "d3ed251f77a4f187cdef40777dcd3f5935bd0f80a268ce6e56393c8954fde8f8"),
    "qwen": ("qwen3.5:9b-q4_K_M",
             "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"),
}


def frozen_cases(source: Path, reference: Path) -> tuple[list[dict], dict]:
    if sha256(source.read_bytes()).hexdigest() != DEV_SHA:
        raise ValueError("BIRD DEV question bytes drift")
    rows = json.loads(source.read_text(encoding="utf-8"))
    manifest = json.loads(reference.read_text(encoding="utf-8"))
    if not (len(rows) == 1534 and manifest.get("status") == "complete"
            and manifest.get("sample_size") == 100
            and manifest.get("source_sha256") == REFERENCE_SHA):
        raise ValueError("BIRD DEV frozen reference drift")
    by_id = {str(x.get("question_id") or x.get("id") or i): x for i, x in enumerate(rows)}
    cases = []
    for original in manifest["cases"]:
        cid = str(original["case_id"])
        row = by_id[cid]
        if row["db_id"] != original["database_id"] or row["question"] != original["question"]:
            raise ValueError("question/db identity drift")
        cases.append({
            "case_id": cid, "db_id": row["db_id"],
            "question": row["question"], "evidence": row.get("evidence") or "",
        })
    if len(cases) != 100 or len({c["case_id"] for c in cases}) != 100:
        raise ValueError("missing frozen BIRD DEV cases")
    digest = sha256(json.dumps(
        [(c["case_id"], c["db_id"], c["question"]) for c in cases],
        ensure_ascii=False, separators=(",", ":"),
    ).encode()).hexdigest()
    return cases, {"digest": digest, "source_by_id": by_id}


def run_shard(source: Path, reference: Path, db_root: Path, shard: int) -> dict:
    if shard not in range(N_SHARDS):
        raise ValueError("invalid shard index")
    cases, frozen = frozen_cases(source, reference)
    models = {key: _ollama_model_digest("http://127.0.0.1:11434", info[0])
              for key, info in MODELS.items()}
    if models != {key: info[1] for key, info in MODELS.items()}:
        raise ValueError("model digest changed")
    scores = []
    db_cache = {}
    for item in cases[shard::N_SHARDS]:
        db_id = item["db_id"]
        if db_id not in db_cache:
            path = _resolve_database(db_root, db_id)
            db_cache[db_id] = (path, DatabaseCatalog.from_sqlite(path))
        db, catalog = db_cache[db_id]
        schema, _meta = select_schema_context(catalog, db_root, db_id, item["question"])
        settings = dict(timeout_s=900, temperature=0, seed=42, num_predict=1024)
        xi = CountingProvider(OllamaProvider(model=MODELS["xiyan"][0], **settings))
        qw = CountingProvider(OllamaProvider(model=MODELS["qwen"][0], **settings))
        service = XiYanFirstSQLService(
            db, xiyan_provider=xi, fallback_provider=qw,
            max_rows=50_000, fallback_repairs=2,
        )
        start = perf_counter()
        result = service.ask(
            item["question"], evidence=item["evidence"] or None,
            schema_context=schema,
        )
        latency = perf_counter() - start
        # Gold is touched only after the entire live route finishes.
        gold = execute_readonly(
            db, frozen["source_by_id"][item["case_id"]]["SQL"], max_rows=50_000,
        )
        if gold.truncated:
            raise ValueError("gold output truncated")
        valid = result.status == "ok" and not result.truncated
        route = result.stages[0].summary
        scores.append({
            "case_id": item["case_id"],
            "ex": bool(valid and set(result.rows) == set(gold.rows)),
            "bag": bool(valid and Counter(result.rows) == Counter(gold.rows)),
            "execution_success": valid,
            "route": route,
            "xiyan_calls": xi.total_calls, "qwen_calls": qw.total_calls,
            "latency_s": round(latency, 3),
        })
        print("LIVE_BIRD", shard, item["case_id"], route, round(latency, 3), flush=True)
    if len(scores) != 10:
        raise ValueError("incomplete live BIRD shard")
    return {"shard": shard, "cohort_sha": frozen["digest"],
            "models": models, "cases": scores}


def aggregate(shards: list[dict]) -> dict:
    if len(shards) != 10 or sorted(s["shard"] for s in shards) != list(range(10)):
        raise ValueError("need ten unique successful shards")
    first = shards[0]
    if any(s["models"] != first["models"]
           or s["cohort_sha"] != first["cohort_sha"]
           or len(s["cases"]) != 10 for s in shards):
        raise ValueError("frozen cohort or model drift")
    rows = [r for s in shards for r in s["cases"]]
    if len(rows) != 100 or len({r["case_id"] for r in rows}) != 100:
        raise ValueError("100 unique frozen DEV cases required")
    return {
        "count": len(rows), "models": first["models"],
        "cohort_sha": first["cohort_sha"],
        "ex": sum(r["ex"] for r in rows),
        "bag": sum(r["bag"] for r in rows),
        "execution_success": sum(r["execution_success"] for r in rows),
        "routes": dict(Counter(
            "xiyan_primary_valid" if r["route"] == "xiyan_primary_valid"
            else "qwen_fallback" for r in rows
        )),
        "model_calls": {name: sum(r[name + "_calls"] for r in rows)
                        for name in ("xiyan", "qwen")},
        "mean_latency_s": round(sum(r["latency_s"] for r in rows)/100, 3),
        "claim_boundary": "real conditional inference on public frozen BIRD DEV100; not hidden performance",
        "cases": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path)
    p.add_argument("--reference", type=Path, default=Path("runs/bird-paired-dev-v5/paired.json"))
    p.add_argument("--db-root", type=Path)
    p.add_argument("--shard", type=int)
    p.add_argument("--inputs", type=Path, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.inputs:
        out = aggregate([json.loads(path.read_text()) for path in args.inputs])
        print("LIVE_BIRD_FINAL", json.dumps({k:v for k,v in out.items() if k != "cases"}), flush=True)
    else:
        if args.source is None or args.db_root is None or args.shard is None:
            p.error("--source, --db-root, --shard required")
        out = run_shard(args.source, args.reference, args.db_root, args.shard)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
