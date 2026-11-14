"""Matched generator swap on BIRD train databases held out from project DEV."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import partial
from hashlib import sha1, sha256
import json
from pathlib import Path
from time import perf_counter
from urllib.parse import quote
from urllib.request import urlopen

from evaluation.schema_context import select_schema_context
from evaluation.run_bird import CountingProvider, _ollama_model_digest
from evaluation.xiyan_qwen_factorial_fresh40 import (
    MODEL_IDS, QWEN_DIGEST, SOURCE_SHA256, _ex, generate, prompt_for,
)
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.contracts import SQLCandidate
from semantic_sql.execution import execute_readonly
from semantic_sql.inference import run_guarded

MIRROR = "https://huggingface.co"
DATASET = "Emulated-Inc/text-to-sql-with-evidence-training-pool"
REVISION = "cd2c4ea321075aa53ce7dfe52deaaffa505d046a"
TRAIN_SHA = "cd5884489dffd7c76d73509b396bd919fbe8b486efbc4ba2f48f89460de7d69f"
XIYAN_DIGEST = "d3ed251f77a4f187cdef40777dcd3f5935bd0f80a268ce6e56393c8954fde8f8"
SEED = "bird-generator-transfer-20261114"
N_SHARDS = 10


def select_cases(
    train: list[dict], dev: list[dict], mirror: dict, *, guarded: bool = False,
) -> list[dict]:
    def normalized(question):
        return " ".join(question.casefold().split())

    used_db = {row["db_id"] for row in dev}
    used_q = {normalized(row["question"]) for row in dev}
    if guarded:
        previous = select_cases(train, dev, mirror)
        used_db.update(r["database_id"] for r in previous)
        used_q.update(normalized(r["question"]) for r in previous)
    eligible = defaultdict(list)
    for index, row in enumerate(train):
        db = row["db_id"]
        if db not in used_db and db in mirror and mirror[db]["size"] <= 128 * 1024**2:
            q = normalized(row["question"])
            if q and q not in used_q:
                eligible[db].append({
                    "case_id": index, "database_id": db, "question": row["question"],
                    "evidence": row.get("evidence") or "",
                })
                used_q.add(q)
    dbs = sorted((db for db, rows in eligible.items() if len(rows) >= 2),
                 key=lambda db: sha256(f"{SEED}:{db}".encode()).digest())[:20]
    if len(dbs) != 20:
        raise ValueError("need 20 eligible BIRD train databases outside BIRD DEV")
    return [row for db in dbs for row in sorted(
        eligible[db], key=lambda r: sha256(f"{SEED}:{r['case_id']}".encode()).digest()
    )[:2]]


def load_checked(path: Path, digest: str):
    data = path.read_bytes()
    if sha256(data).hexdigest() != digest:
        raise ValueError(f"{path.name} checksum mismatch")
    return json.loads(data)


def remote_manifest() -> dict:
    url = (f"{MIRROR}/api/datasets/{DATASET}/tree/{REVISION}/raw/bird/train_databases"
           "?recursive=true&limit=1000")
    with urlopen(url, timeout=60) as response:
        records = json.load(response)
    dbs = {Path(row["path"]).stem: row for row in records if row["path"].endswith(".sqlite")}
    if len(dbs) != 51:
        raise ValueError("incomplete checksum-pinned BIRD train database catalog")
    return dbs


def check_database_bytes(data: bytes, metadata: dict) -> bool:
    if len(data) != metadata["size"]:
        return False
    if "lfs" in metadata:
        return sha256(data).hexdigest() == metadata["lfs"]["oid"]
    header = f"blob {len(data)}\0".encode()
    return sha1(header + data).hexdigest() == metadata["oid"]


class _ModelProvider:
    def __init__(self, model: str, generator):
        self.complete_text = partial(generator, model=model)


def guarded_qwen(db, question, schema, evidence, generator=generate):
    provider = CountingProvider(_ModelProvider(MODEL_IDS["qwen"], generator))
    start = perf_counter()
    result = run_guarded(
        database=db, provider=provider, question=question, schema_context=schema,
        evidence=evidence, max_repairs=2, max_rows=50_000,
    )
    return {"sql": result.candidate.sql, "model_calls": provider.total_calls,
            "latency_s": round(perf_counter() - start, 3), "result": result}


def run_shard(
    train_path: Path, dev_path: Path, shard: int, work: Path, *, guarded: bool = False,
) -> dict:
    if shard not in range(N_SHARDS):
        raise ValueError("invalid BIRD transfer shard")
    train = load_checked(train_path, TRAIN_SHA)
    dev = load_checked(dev_path, SOURCE_SHA256)
    db_manifest = remote_manifest()
    all_cases = select_cases(train, dev, db_manifest, guarded=guarded)
    rows = all_cases[4 * shard:4 * (shard + 1)]
    models = {key: _ollama_model_digest("http://127.0.0.1:11434", model)
              for key, model in MODEL_IDS.items()}
    if models != {"qwen": QWEN_DIGEST, "xiyan": XIYAN_DIGEST}:
        raise ValueError(f"generator identity drift: {models}")
    databases = {}
    for db in {row["database_id"] for row in rows}:
        path = work / db / f"{db}.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{MIRROR}/datasets/{DATASET}/resolve/{REVISION}/raw/bird/train_databases"
        with urlopen(f"{url}/{quote(db)}/{quote(db)}.sqlite", timeout=120) as response:
            data = response.read(db_manifest[db]["size"] + 1)
        if not check_database_bytes(data, db_manifest[db]):
            raise ValueError(f"{db} SQLite bytes did not match the pinned manifest")
        path.write_bytes(data)
        databases[db] = (path, DatabaseCatalog.from_sqlite(path))
    for row in rows:
        _, catalog = databases[row["database_id"]]
        row["schema_context"], _ = select_schema_context(
            catalog, work, row["database_id"], row["question"]
        )
        row["arms"] = {}
    for key, model in MODEL_IDS.items():
        for row in rows:
            question = row["question"]
            schema = row["schema_context"]
            evidence = row["evidence"].strip() or None
            db, _ = databases[row["database_id"]]
            if guarded and key == "qwen":
                arm = guarded_qwen(db, question, schema, evidence)
            else:
                system, user = prompt_for(condition="xiyan", question=question,
                                          schema_context=schema, evidence=evidence)
                start = perf_counter()
                sql = generate(model=model, system=system, user=user)
                arm = {"sql": sql, "latency_s": round(perf_counter() - start, 3)}
                if guarded:
                    result = run_guarded(
                        database=db, provider=_ModelProvider(model, generate),
                        question=question, schema_context=schema, evidence=evidence,
                        initial_candidate=SQLCandidate(sql), max_repairs=0, max_rows=50_000,
                    )
                    arm.update(result=result, model_calls=1,
                               latency_s=round(perf_counter() - start, 3))
            row["arms"][key] = arm
            print(f"generated {key} shard={shard} case={row['case_id']}", flush=True)

    # Neither the cohort nor any model request consumes the reference SQL.
    for row in rows:
        del row["schema_context"]
        db, _ = databases[row["database_id"]]
        gold = execute_readonly(db, train[row["case_id"]]["SQL"], max_rows=50_000)
        if gold.truncated:
            raise ValueError("transfer reference truncated")
        gold_rows = set(gold.rows)
        for arm in row["arms"].values():
            if guarded:
                result = arm.pop("result")
                executed = result.status == "ok" and not result.truncated
                arm.update(ex=executed and set(result.rows) == gold_rows,
                           execution_success=executed, preflight=result.verification.ok,
                           reason="truncated" if result.truncated else result.status)
                matched_rows = result.rows if arm["ex"] else ()
            else:
                arm.update(_ex(db, arm["sql"], gold_rows))
                matched_rows = (
                    execute_readonly(db, arm["sql"], max_rows=50_000).rows if arm["ex"] else ()
                )
            arm["bag_equal"] = bool(arm["ex"] and Counter(matched_rows) == Counter(gold.rows))
            arm["both_empty"] = bool(arm["ex"] and not gold.rows)
    return {"shard": shard, "models": models, "guarded": guarded, "cohort": [
        [row["case_id"], row["database_id"]] for row in all_cases
    ], "cases": rows}


def aggregate(shards: list[dict]) -> dict:
    if sorted(s["shard"] for s in shards) != list(range(N_SHARDS)):
        raise ValueError("need all ten BIRD transfer shards")
    ordered = sorted(shards, key=lambda s: s["shard"])
    first = ordered[0]
    guarded = first.get("guarded", False)
    if any(s["models"] != first["models"] or s["cohort"] != first["cohort"]
           or s.get("guarded", False) != guarded for s in shards):
        raise ValueError("cohort, model, or protocol drift")
    rows = [r for shard in ordered for r in shard["cases"]]
    if ([[r["case_id"], r["database_id"]] for r in rows] != first["cohort"]
        or len(rows) != 40 or len({r["case_id"] for r in rows}) != 40
        or len(Counter(r["database_id"] for r in rows)) != 20):
        raise ValueError("incomplete or unpaired transfer results")
    stats = {}
    for key in MODEL_IDS:
        vals = [r["arms"][key] for r in rows]
        stats[key] = {
            **{k: sum(row[k] for row in vals) for k in ("ex", "bag_equal", "both_empty", "preflight", "execution_success")},
            "mean_latency_s": round(sum(row["latency_s"] for row in vals) / len(vals), 3),
        }
        if guarded:
            stats[key]["model_calls"] = sum(r["model_calls"] for r in vals)
    return {
        "count": len(rows), "database_count": 20, "models": first["models"],
        "prompt": "Qwen generic + up to 2 repairs vs XiYan native + preflight"
        if guarded else "xiyan for both generators",
        "arms": stats,
        "wrong_to_correct": sum(r["arms"]["xiyan"]["ex"] and not r["arms"]["qwen"]["ex"] for r in rows),
        "correct_to_wrong": sum(r["arms"]["qwen"]["ex"] and not r["arms"]["xiyan"]["ex"] for r in rows),
        "bag_wrong_to_correct": sum(
            r["arms"]["xiyan"]["bag_equal"] and not r["arms"]["qwen"]["bag_equal"] for r in rows
        ),
        "bag_correct_to_wrong": sum(
            r["arms"]["qwen"]["bag_equal"] and not r["arms"]["xiyan"]["bag_equal"] for r in rows
        ),
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--dev", type=Path)
    parser.add_argument("--work", type=Path, default=Path("transfer-db"))
    parser.add_argument("--shard", type=int)
    parser.add_argument("--guarded", action="store_true")
    parser.add_argument("--inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.inputs:
        result = aggregate([json.loads(p.read_text()) for p in args.inputs])
    else:
        if args.train is None or args.dev is None or args.shard is None:
            parser.error("--train, --dev and --shard required for inference")
        result = run_shard(args.train, args.dev, args.shard, args.work, guarded=args.guarded)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("cases", "cohort")}), flush=True)


if __name__ == "__main__":
    main()
