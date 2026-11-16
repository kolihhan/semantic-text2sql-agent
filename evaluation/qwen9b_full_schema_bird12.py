"""Paired unchanged Qwen3.5-9B Guarded: lexical schema vs physical full schema.

One variable: original lexical table selection vs physically verified complete
SQLite schema. No SQL training, oracle selector or model replacement. A source-
only prospective cohort excludes original frozen BIRD100 before generation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic

from evaluation.qwen_schema_grounding_pair import (
    MODEL, MODEL_DIGEST, SOURCE_SHA, PinnedQwen,
    checked_source, case_id, legacy_tokens,
)
from evaluation.run_bird import _ollama_model_digest, _resolve_database
from evaluation.schema_context import select_schema_context
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.execution import execute_readonly, SQLExecutionError
from semantic_sql.inference import run_guarded

PROTOCOL = "original-qwen9b-full-physical-schema-vs-lexical-bird12-v1"
SEED = "qwen9b-full-physical-schema-prelabel-source-20261116"
N, SHARDS, PER_SHARD = 12, 4, 3
MAX_SCHEMA_CHARS = 12000
MIN_DBS = 4


def schemas(catalog: DatabaseCatalog, root: Path, dbid: str, question: str):
    """Original tokenizer in both arms; only table selection changes."""
    small, s = select_schema_context(
        catalog, root, dbid, question, tokenizer=legacy_tokens)
    full, f = select_schema_context(
        catalog, root, dbid, "", tokenizer=legacy_tokens)
    if set(s["selected_tables"]) - set(f["selected_tables"]):
        raise AssertionError("physical full schema lost original table")
    return (small, s), (full, f)


def build_manifest(source: Path, root: Path, reference: Path):
    questions = checked_source(source)
    prior = json.loads(reference.read_text(encoding="utf-8"))
    if (prior.get("status") != "complete" or prior.get("sample_size") != 100
        or len(prior.get("cases", [])) != 100):
        raise ValueError("original frozen100 source exclusion absent")
    excluded = {str(x["case_id"]) for x in prior["cases"]}
    if len(excluded) != 100:
        raise ValueError("duplicate frozen original100 IDs")
    ordered = sorted(
        [(case_id(q, i), q) for i, q in enumerate(questions)
         if case_id(q, i) not in excluded],
        key=lambda it: (sha256((SEED + ":" + it[0]).encode()).hexdigest(), it[0]),
    )
    selected, counts, catalogs = [], Counter(), {}
    eligible = Counter()
    for cid, row in ordered:
        dbid, question = str(row["db_id"]), str(row["question"])
        if dbid not in catalogs:
            try:
                catalogs[dbid] = DatabaseCatalog.from_sqlite(_resolve_database(root, dbid))
            except (FileNotFoundError, OSError):
                continue
        (small, sm), (full, fm) = schemas(catalogs[dbid], root, dbid, question)
        if (small == full or len(full) > MAX_SCHEMA_CHARS
            or len(set(fm["selected_tables"]) - set(sm["selected_tables"])) < 2):
            continue
        eligible[dbid] += 1
        if counts[dbid] >= 3 or len(selected) == N:
            continue
        selected.append({
            "case_id": cid, "database_id": dbid, "question": question,
            "evidence": str(row.get("evidence") or ""),
            "lexical_schema_sha256": sm["schema_context_sha256"],
            "full_schema_sha256": fm["schema_context_sha256"],
            "original_tables": sm["selected_tables"],
            "added_physical_tables": sorted(set(fm["selected_tables"]) - set(sm["selected_tables"])),
            "full_schema_chars": len(full),
        })
        counts[dbid] += 1
    if len(selected) != N or len(counts) < MIN_DBS:
        raise ValueError(
            f"Insufficient original physical schema transfer: {len(selected)} "
            f"cases/{len(counts)} databases; frozen gate requires {N}/{MIN_DBS}. "
            "Do not silently retune to exposed BIRD references"
        )
    ident = [(x["case_id"], x["database_id"]) for x in selected]
    return {
        "protocol": PROTOCOL, "source_sha256": SOURCE_SHA,
        "model": MODEL, "model_digest": MODEL_DIGEST,
        "selection_seed": SEED, "original100_excluded": 100,
        "case_count": N, "database_count": len(counts),
        "max_context_chars": MAX_SCHEMA_CHARS,
        "eligible_source_by_database": dict(sorted(eligible.items())),
        "cohort_sha256": sha256(json.dumps(ident, separators=(",", ":")).encode()).hexdigest(),
        "gold_used_for_selection": False, "model_calls_before_freeze": 0,
        "same_original_qwen9b": True, "no_sql_training": True,
        "cases": selected,
    }


def evaluated_rows(database: Path, reference_sql: str, arms: dict):
    try:
        gold = execute_readonly(database, reference_sql, max_rows=50000)
        if gold.truncated:
            raise ValueError("gold result truncated")
    except (SQLExecutionError, ValueError) as exc:
        for arm in arms.values():
            arm.update(strict_ex=None, strict_bag=None, both_empty=None)
        return False, type(exc).__name__
    for arm in arms.values():
        got = arm.pop("_result")
        matched = got is not None and bool(arm["execution_success"])
        arm["strict_ex"] = bool(matched and set(map(tuple, got)) == set(map(tuple, gold.rows)))
        arm["strict_bag"] = bool(matched and Counter(map(tuple, got)) == Counter(map(tuple, gold.rows)))
        arm["both_empty"] = bool(matched and len(gold.rows) == 0 and len(got) == 0)
    return True, None


def run_shard(source: Path, root: Path, manifest: Path, shard: int):
    if shard not in range(SHARDS):
        raise ValueError("invalid shard")
    original = checked_source(source)
    by_id = {case_id(x, i): x for i, x in enumerate(original)}
    config = json.loads(manifest.read_text())
    if (config.get("protocol") != PROTOCOL or config.get("model_digest") != MODEL_DIGEST
        or config.get("source_sha256") != SOURCE_SHA or config.get("case_count") != N
        or config.get("gold_used_for_selection") is not False
        or len(config.get("cases", [])) != N):
        raise ValueError("unfrozen SQL experiment manifest")
    if _ollama_model_digest("http://127.0.0.1:11434", MODEL) != MODEL_DIGEST:
        raise ValueError("original Qwen9B digest changed")
    picked = config["cases"][shard * PER_SHARD:(shard + 1) * PER_SHARD]
    if len(picked) != PER_SHARD:
        raise RuntimeError("insufficient shard cases")
    provider = PinnedQwen()
    staged = []
    catalog_cache = {}
    for item in picked:
        cid, dbid = item["case_id"], item["database_id"]
        row = by_id[cid]
        if (str(row["db_id"]) != dbid or str(row["question"]) != item["question"]):
            raise ValueError("question identity drift")
        if dbid not in catalog_cache:
            db = _resolve_database(root, dbid)
            catalog_cache[dbid] = (db, DatabaseCatalog.from_sqlite(db))
        db, catalog = catalog_cache[dbid]
        (small, sm), (full, fm) = schemas(catalog, root, dbid, item["question"])
        if (sm["schema_context_sha256"] != item["lexical_schema_sha256"]
            or fm["schema_context_sha256"] != item["full_schema_sha256"]
            or len(full) != item["full_schema_chars"]):
            raise ValueError("physical schema source drift")
        rec = {
            "case_id": cid, "database_id": dbid,
            "question": item["question"], "arms": {},
            "expanded_table_count": len(item["added_physical_tables"]),
        }
        # Both actual Qwen9B arms are generated before gold SQL is used.
        for name, context in (("original_lexical", small), ("expanded_physical", full)):
            start = monotonic()
            got = run_guarded(
                database=db, provider=provider,
                question=item["question"], schema_context=context,
                evidence=item["evidence"].strip() or None,
                max_repairs=2, max_rows=50000,
            )
            ok = bool(got.status == "ok" and not got.truncated)
            rec["arms"][name] = {
                "sql": got.candidate.sql if got.candidate else "",
                "execution_success": ok, "status": got.status,
                "repair_count": got.candidate.attempt if got.candidate else None,
                "seconds": round(monotonic()-start, 2),
                "_result": list(got.rows) if ok else None,
            }
        staged.append(rec)
        print("QWEN9B_FULL_SCHEMA_GENERATED", shard, cid, flush=True)
    for rec in staged:
        db, _ = catalog_cache[rec["database_id"]]
        row = by_id[rec["case_id"]]
        scorable, why = evaluated_rows(db, str(row["SQL"]), rec["arms"])
        rec["reference_scorable"] = scorable
        rec["reference_error"] = why
        # Even invalid gold does not retain raw predicted row sets.
        for arm in rec["arms"].values():
            arm.pop("_result", None)
    return {
        "protocol": PROTOCOL, "shard": shard,
        "cohort_sha256": config["cohort_sha256"],
        "model_digest": MODEL_DIGEST, "cases": staged,
    }


def aggregate(shards: list[dict]):
    if len(shards) != SHARDS or sorted(x.get("shard") for x in shards) != list(range(SHARDS)):
        raise ValueError("exactly 4 distinct original-model shards required")
    cohort = shards[0]["cohort_sha256"]
    if any(x.get("protocol") != PROTOCOL or x.get("cohort_sha256") != cohort
           or x.get("model_digest") != MODEL_DIGEST or len(x.get("cases", [])) != PER_SHARD
           for x in shards):
        raise ValueError("source/model/samples differ between shards")
    cases = [r for shard in shards for r in shard["cases"]]
    if len({r["case_id"] for r in cases}) != N:
        raise ValueError("duplicate BIRD questions")
    valid = [r for r in cases if r["reference_scorable"]]
    def total(arm, field):
        return sum(bool(r["arms"][arm][field]) for r in valid)
    rescued = [r["case_id"] for r in valid
               if not r["arms"]["original_lexical"]["strict_ex"]
               and r["arms"]["expanded_physical"]["strict_ex"]]
    regressed = [r["case_id"] for r in valid
                 if r["arms"]["original_lexical"]["strict_ex"]
                 and not r["arms"]["expanded_physical"]["strict_ex"]]
    net = len(rescued) - len(regressed)
    affected_dbs = {r["database_id"] for r in valid if r["case_id"] in rescued}
    return {
        "protocol": PROTOCOL, "model": MODEL, "model_digest": MODEL_DIGEST,
        "cohort_sha256": cohort, "n": N, "scorable": len(valid),
        "original_lexical_set_ex": total("original_lexical", "strict_ex"),
        "expanded_physical_set_ex": total("expanded_physical", "strict_ex"),
        "original_lexical_bag_ex": total("original_lexical", "strict_bag"),
        "expanded_physical_bag_ex": total("expanded_physical", "strict_bag"),
        "rescued": rescued, "regressed": regressed,
        "net_rescues": net, "distinct_rescue_dbs": len(affected_dbs),
        "positive_screen_only": bool(
            len(valid) >= 11 and net >= 3 and len(affected_dbs) >= 3
            and not regressed and total("expanded_physical","strict_bag") >= total("original_lexical","strict_bag")
        ),
        "promoted": False,
        "no_sql_training": True, "other_model_used": False,
        "original_BIRD100_semantic_ex_unchanged": 39,
        "public_BIRD_DEV_not_hidden": True,
        "cases": cases,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path)
    p.add_argument("--root", type=Path)
    p.add_argument("--reference", type=Path, default=Path("runs/bird-paired-dev-v5/paired.json"))
    p.add_argument("--manifest", type=Path)
    p.add_argument("--shard", type=int)
    p.add_argument("--inputs", nargs="+", type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("cannot overwrite frozen SQL experiment")
    if a.inputs:
        obj = aggregate([json.loads(x.read_text()) for x in a.inputs])
        print("QWEN9B_FULL_SCHEMA_STRICT_EX_FINAL",
              json.dumps({k:v for k,v in obj.items() if k != "cases"},sort_keys=True),flush=True)
    elif a.shard is not None:
        if not all([a.source, a.root, a.manifest]):
            p.error("shard requires source,root,manifest")
        obj = run_shard(a.source, a.root, a.manifest, a.shard)
    else:
        if not a.source or not a.root:
            p.error("preregister requires source and physical DB root")
        obj = build_manifest(a.source, a.root, a.reference)
        print("QWEN9B_FULL_SCHEMA_SOURCE_PREREGISTERED",
              json.dumps({k:v for k,v in obj.items() if k != "cases"},sort_keys=True),flush=True)
    a.output.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
