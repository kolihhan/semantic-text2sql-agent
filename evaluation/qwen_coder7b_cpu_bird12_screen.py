"""CPU Qwen2.5-Coder-7B generation screen against frozen original Qwen9B BIRD12.

No SQL training. Original Qwen9B SQL and 12-case cohort are recovered from
the completed physical source-freeze artifact; no benchmark answers enter
generation, and no SQL patch is promoted without source-shift replication.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256, file_digest
import json
import os
from pathlib import Path
from time import monotonic

from evaluation import colab_qwen3_14b_bird12_screen as frozen_source
from evaluation.run_bird import _resolve_database, _ollama_model_digest
from evaluation.schema_context import select_schema_context
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.execution import execute_readonly, SQLExecutionError
from semantic_sql.inference import run_guarded
from semantic_sql.providers import OllamaProvider

MODEL = "qwen2.5-coder:7b-instruct-q4_K_M"
PUBLISHED_DIGEST_PREFIX = "dae161e27b0e"
PROTOCOL = "qwen-only-coder7b-vs-frozen-qwen35-9b-guarded-bird12-cpu-v1"
COHORT_SHA = frozen_source.COHORT_SHA
MAX_ROWS = 50_000
SHARDS = 3


def require_model() -> str:
    digest = str(_ollama_model_digest("http://127.0.0.1:11434", MODEL))
    if not digest.split(":")[-1].startswith(PUBLISHED_DIGEST_PREFIX):
        raise RuntimeError("untrusted Qwen2.5-Coder tag: Ollama digest changed")
    return digest


def prepare_source(source_root: Path):
    sha = os.environ.get("GITHUB_SHA", "")
    if len(sha) != 40:
        raise ValueError("GitHub evaluator source revision must be frozen")
    frozen_source.SOURCE_CODE_SHA = sha
    frozen_source.COHORT = source_root / "fresh-native-batched-bird12-manifest.json"
    frozen_source.ORIGINAL = source_root / "native-batched-bird12-final.json"
    frozen_source.WORK = Path("bird-coder7b-source")
    rows, prior = frozen_source.source_inputs()
    raw, dbroot = frozen_source.download_data()
    records = json.loads(raw.read_text(encoding="utf-8"))
    by_id = {str(x.get("question_id") or x.get("id") or i): x
             for i, x in enumerate(records)}
    return rows, prior, by_id, dbroot


def run_shard(source_root: Path, shard: int) -> dict:
    if shard not in range(SHARDS):
        raise ValueError("source shard must be in [0,2]")
    manifest, old, source, dbroot = prepare_source(source_root)
    model_digest = require_model()
    selected = manifest[4 * shard: 4 * shard + 4]
    if len(selected) != 4 or len({r["database_id"] for r in selected}) != 2:
        raise ValueError("frozen paired cohort was changed")
    model = OllamaProvider(model=MODEL, temperature=0, seed=42,
                           num_predict=700, timeout_s=900)
    output = []
    for record in selected:
        cid, dbid = record["case_id"], record["database_id"]
        original = source[cid]
        if original["db_id"] != dbid or " ".join(original["question"].split()) != record["question"]:
            raise ValueError("source question identity changed")
        database = _resolve_database(dbroot, dbid)
        catalog = DatabaseCatalog.from_sqlite(database)
        schema, _ = select_schema_context(catalog, dbroot, dbid, record["question"])
        t0 = monotonic()
        got = run_guarded(
            database=database, provider=model, question=record["question"],
            schema_context=schema, evidence=record["evidence"] or None,
            max_repairs=2, max_rows=MAX_ROWS,
        )
        candidate = got.candidate.sql if got.candidate else None
        with database.open("rb") as handle:
            dbsha = file_digest(handle, "sha256").hexdigest()
        output.append({
            "case_id": cid, "database_id": dbid,
            "source_schema_sha256": sha256(schema.encode()).hexdigest(),
            "source_sqlite_sha256": dbsha,
            "original9b": {
                "sql": old[cid]["arms"]["original"]["sql"],
                "set_ex": old[cid]["arms"]["original"]["set_ex"],
                "bag_ex": old[cid]["arms"]["original"]["bag_ex"],
            },
            "coder7b": {
                "sql": candidate, "status": got.status,
                "truncated": got.truncated, "seconds": round(monotonic() - t0, 2),
                "_rows": got.rows if got.status == "ok" and not got.truncated else None,
            },
        })
        print("CODER7B_SQL_GENERATED", shard, cid, got.status, flush=True)

    # Freeze EVERY Qwen7B SQL string for the shard before reading reference SQL.
    frozen = Path("coder7b-predictions-before-gold-" + str(shard) + ".json")
    frozen.write_text(json.dumps(output, default=str, indent=2) + "\n")
    for record in output:
        dbid, cid = record["database_id"], record["case_id"]
        db = _resolve_database(dbroot, dbid)
        try:
            reference = execute_readonly(db, str(source[cid]["SQL"]), max_rows=MAX_ROWS)
            scorable = not reference.truncated
        except (SQLExecutionError, ValueError):
            reference, scorable = None, False
        result = record["coder7b"]
        rows = result.pop("_rows")
        if not scorable:
            result["set_ex"] = result["bag_ex"] = None
        else:
            good = rows is not None
            result["set_ex"] = bool(good and set(map(tuple, rows)) == set(map(tuple, reference.rows)))
            result["bag_ex"] = bool(good and Counter(map(tuple, rows)) == Counter(map(tuple, reference.rows)))
    return {
        "protocol": PROTOCOL, "model": MODEL, "digest": model_digest,
        "source_cohort_sha": COHORT_SHA,
        "source_bird_sha": frozen_source.BIRD_SHA,
        "no_training": True, "no_default_changes": True,
        "shard": shard, "cases": output,
    }


def aggregate(shards: list[dict]) -> dict:
    if (len(shards) != SHARDS or
        sorted(x.get("shard") for x in shards) != list(range(SHARDS)) or
        len({x.get("digest") for x in shards}) != 1 or
        any(x.get("protocol") != PROTOCOL or
            x.get("source_cohort_sha") != COHORT_SHA or
            x.get("no_training") is not True or
            len(x.get("cases", [])) != 4 for x in shards)):
        raise ValueError("source/model-shard mismatch or incomplete physical experiment")
    cases = [r for x in sorted(shards, key=lambda x: x["shard"]) for r in x["cases"]]
    if len(cases) != 12 or len({x["case_id"] for x in cases}) != 12 or len({x["database_id"] for x in cases}) != 6:
        raise ValueError("incomplete or repeated source-disjoint BIRD12 cases")
    if any(x["coder7b"]["set_ex"] is None or x["coder7b"]["bag_ex"] is None for x in cases):
        raise ValueError("missing physical gold execution for a case")
    oldset = sum(bool(x["original9b"]["set_ex"]) for x in cases)
    oldbag = sum(bool(x["original9b"]["bag_ex"]) for x in cases)
    newset = sum(x["coder7b"]["set_ex"] for x in cases)
    newbag = sum(x["coder7b"]["bag_ex"] for x in cases)
    if (oldset, oldbag) != (3, 3):
        raise ValueError("original frozen Qwen9B strict EX baseline drift")
    rescues = [x for x in cases if not x["original9b"]["set_ex"] and x["coder7b"]["set_ex"]]
    regressions = [x for x in cases if x["original9b"]["set_ex"] and not x["coder7b"]["set_ex"]]
    return {
        "protocol": PROTOCOL, "model": MODEL, "model_digest": shards[0]["digest"],
        "cohort_sha256": COHORT_SHA, "n": 12,
        "original_set_ex": oldset, "coder7b_set_ex": newset,
        "original_bag_ex": oldbag, "coder7b_bag_ex": newbag,
        "rescues": [x["case_id"] for x in rescues],
        "regressions": [x["case_id"] for x in regressions],
        "distinct_rescue_databases": len({x["database_id"] for x in rescues}),
        "positive_screen_only": bool(
            newset - oldset >= 3 and not regressions and newbag >= oldbag
            and len({x["database_id"] for x in rescues}) >= 3
        ),
        "promoted": False,
        "no_sql_training": True,
        "baseline_bird100_unchanged": 39,
        "caveat": "Public BIRD DEV12 previously seen; only a Qwen-family generation-headroom screen, NOT independent hidden transfer.",
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="*", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path("inputs"))
    parser.add_argument("--shard", type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("do not overwrite previous experiment")
    if args.inputs:
        if len(args.inputs) != SHARDS or args.shard is not None:
            raise ValueError("need exactly three frozen inputs and no shard")
        result = aggregate([json.loads(p.read_text()) for p in args.inputs])
    else:
        if args.shard is None:
            raise ValueError("missing required shard")
        result = run_shard(args.source_root, args.shard)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("CODER7B_BIRD12_PHYSICAL_STRICT_EX",
          json.dumps({k: v for k, v in result.items() if k != "cases"}, sort_keys=True),
          flush=True)


if __name__ == "__main__":
    main()
