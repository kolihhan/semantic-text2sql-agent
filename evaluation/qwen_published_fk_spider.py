"""A/B: first-party published FK annotation augmentation on untouched Spider physical DBs.

Both arms use the same unchanged original Qwen3.5-9B Guarded model, original
read-only SQL checker, same prompt/task/DB; the one variable is adding Spider original FK source annotations validated against SQLite.

Original dev SQL reference is not read for task or route selection, and is
used only after both predictions have been generated for a given shard.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import file_digest, sha256
import json
from pathlib import Path
import re
from time import monotonic

from evaluation.run_bird import CountingProvider, _ollama_model_digest
from evaluation.spider_published_fk import enrichment
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.execution import SQLExecutionError, execute_readonly
from semantic_sql.inference import run_guarded
from semantic_sql.providers import OllamaProvider

PROTOCOL = "qwen9b-first-party-published-fks-spider-v1"
MODEL = "qwen3.5:9b-q4_K_M"
DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
ARCHIVE_SHA = "00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b"
EXCLUDED = frozenset({
    "pets_1", "dog_kennels", "orchestra", "voter_1", "course_teach", "poker_player",
    "car_1", "employee_hire_evaluation", "flight_2", "museum_visit", "singer", "tvshow",
    "concert_singer", "cre_Doc_Template_Mgt", "network_1",
    "student_transcripts_tracking", "world_1", "wta_1"
})
N = 12
DBS = 6
SHARDS = 3
ARMS = ("original", "fkpath")
CAP = 50000


def key(s: str) -> str:
    return sha256((PROTOCOL + ":" + s).encode()).hexdigest()


def physical(root: Path, dbid: str) -> Path:
    if not re.fullmatch(r"[a-zA-Z0-9_]+", dbid):
        raise ValueError("invalid physical database id")
    db = root / dbid / (dbid + ".sqlite")
    if not db.is_file():
        raise FileNotFoundError(db)
    return db


def freeze(train: list[dict], dev: list[dict], root: Path, metadata: dict) -> dict:
    train_dbs = {str(r["db_id"]) for r in train}
    forbidden = EXCLUDED | train_dbs
    grouped = defaultdict(dict)
    checked = Counter()
    for index, row in enumerate(dev):
        dbid = str(row["db_id"])
        if dbid in forbidden:
            continue
        question = " ".join(str(row.get("question") or "").split())
        if not 20 <= len(question) <= 250:
            continue
        grouped[dbid].setdefault(question.casefold(), {
            "id": f"{dbid}:{index}", "db_id": dbid,
            "index": index, "question": question
        })
    eligible_by_db = {}
    for dbid, unique in grouped.items():
        db = physical(root, dbid)
        catalog = DatabaseCatalog.from_sqlite(db)
        if len(catalog.tables) < 2 or dbid not in metadata:
            continue
        selected = []
        for row in unique.values():
            baseline, enriched, diag = enrichment(
                catalog, metadata[dbid], root, dbid, row["question"]
            )
            if baseline == enriched or not diag["visible_new_fks"]:
                continue
            if len(baseline) > 13500 or len(enriched) > 14500:
                continue
            checked[dbid] += 1
            selected.append({
                **row,
                "published_fk": diag["missing_physical_fk"],
                "visible_new_fks": diag["visible_new_fks"],
                "lexical_schema_sha256": sha256(baseline.encode()).hexdigest(),
                "route_schema_sha256": sha256(enriched.encode()).hexdigest(),
            })
        if len(selected) >= 2:
            eligible_by_db[dbid] = sorted(selected, key=lambda r: key(r["id"]))[:2]
    picked = []
    for dbid in sorted(eligible_by_db, key=lambda d: key("db:" + d)):
        picked.extend(eligible_by_db[dbid])
        if len(picked) == N:
            break
    if len(picked) != N or len({r["db_id"] for r in picked}) != DBS:
        raise ValueError(
            f"Source feasibility failure: {len(picked)}/{N} eligible questions, "
            f"{len(eligible_by_db)} unique databases with 2+ questions; "
            f"candidate distributions {dict(checked)}"
        )
    assert all("query" not in c and "SQL" not in c for c in picked)
    return {
        "protocol": PROTOCOL, "model_digest": DIGEST,
        "archive_sha256": ARCHIVE_SHA, "cases_sha256":
            sha256(json.dumps(picked, sort_keys=True).encode()).hexdigest(),
        "selection": "2 original Spider DEV questions per 6 untouched physical DBs with published-only FK facts; original TRAIN DBs excluded",
        "gold_blind": True, "count": N, "physical_dbs": DBS,
        "excluded_prior_physical": sorted(EXCLUDED), "cases": picked
    }


def generate(db: Path, question: str, schema: str):
    provider = CountingProvider(OllamaProvider(
        model=MODEL, timeout_s=600, temperature=0, seed=42, num_predict=700
    ))
    started = monotonic()
    result = run_guarded(
        database=db, provider=provider, question=question,
        schema_context=schema, max_repairs=2, max_rows=CAP
    )
    return {
        "sql": result.candidate.sql if result.candidate else "",
        "status": result.status, "truncated": result.truncated,
        "calls": provider.total_calls, "seconds": round(monotonic()-started, 3),
        "rows": result.rows,
    }


def one_shard(manifest: Path, trainfile: Path, devfile: Path,
              root: Path, shard: int, metadata: dict) -> dict:
    if shard not in range(SHARDS):
        raise ValueError("invalid shard")
    train, dev = json.loads(trainfile.read_text()), json.loads(devfile.read_text())
    frozen = json.loads(manifest.read_text())
    if frozen != freeze(train, dev, root, metadata):
        raise ValueError("original Spider physical corpus changed")
    if _ollama_model_digest("http://127.0.0.1:11434", MODEL) != DIGEST:
        raise ValueError("physical Qwen manifest digest changed")
    predictions = []
    for i, row in enumerate(frozen["cases"][shard * 4:(shard+1)*4]):
        db = physical(root, row["db_id"])
        cat = DatabaseCatalog.from_sqlite(db)
        baseline, extended, diagnostics = enrichment(
            cat, metadata[row["db_id"]], root, row["db_id"], row["question"]
        )
        assert sha256(baseline.encode()).hexdigest() == row["lexical_schema_sha256"]
        assert sha256(extended.encode()).hexdigest() == row["route_schema_sha256"]
        order = ARMS if (i+shard)%2 == 0 else ARMS[::-1]
        arm_results = {
            a: generate(db, row["question"], baseline if a == "original" else extended)
            for a in order
        }
        predictions.append((row, diagnostics, arm_results))
        print("PUBLISHED_FK_BOTH_QWEN_GENERATIONS_FROZEN", row["id"],
              {a: arm_results[a]["status"] for a in ARMS}, flush=True)
    # ONLY after all shard predictions have been frozen is the dev reference
    # SQL permitted to enter scoring, and never returned to the Qwen model.
    cases = []
    for row, diag, results in predictions:
        db = physical(root, row["db_id"])
        try:
            gold = execute_readonly(db, str(dev[row["index"]]["query"]), max_rows=CAP)
            scorable = not gold.truncated
        except (SQLExecutionError, ValueError):
            scorable, gold = False, None
        for a in ARMS:
            result = results[a]
            original_rows = result.pop("rows")
            valid = result["status"] == "ok" and not result["truncated"]
            result["set_ex"] = (bool(valid and set(map(tuple, original_rows))
                                     == set(map(tuple, gold.rows)))
                                if scorable else None)
            result["bag_ex"] = (bool(valid and Counter(map(tuple, original_rows))
                                     == Counter(map(tuple, gold.rows)))
                                if scorable else None)
        with db.open("rb") as f:
            db_sha = file_digest(f, "sha256").hexdigest()
        cases.append({
            "id": row["id"], "db_id": row["db_id"],
            "question": row["question"], "db_sha256": db_sha,
            "scorable": scorable, "source_route_info": diag,
            **results
        })
    return {
        "protocol": PROTOCOL, "digest": DIGEST,
        "archive_sha256": ARCHIVE_SHA,
        "cases_sha256": frozen["cases_sha256"],
        "shard": shard, "cases": cases
    }


def aggregate(shards: list[dict]) -> dict:
    if (len(shards) != SHARDS or sorted(r["shard"] for r in shards) != list(range(SHARDS))
        or any(r["protocol"] != PROTOCOL or r["digest"] != DIGEST
               or r["archive_sha256"] != ARCHIVE_SHA
               or len(r["cases"]) != N//SHARDS for r in shards)
        or len({r["cases_sha256"] for r in shards}) != 1):
        raise ValueError("missing or drifting paired shards")
    rows = [r for shard in sorted(shards, key=lambda s: s["shard"])
            for r in shard["cases"]]
    dbs = {r["db_id"] for r in rows}
    if len(rows) != N or len({r["id"] for r in rows}) != N or len(dbs) != DBS:
        raise ValueError("physical source coverage/IDs invalid")
    if any(len({r["db_sha256"] for r in rows if r["db_id"] == db}) != 1 for db in dbs):
        raise ValueError("physical SQLite files differ across paired cases")
    scorable = [r for r in rows if r["scorable"]]
    rescues = [r for r in scorable if r["fkpath"]["set_ex"] and not r["original"]["set_ex"]]
    losses = [r for r in scorable if r["original"]["set_ex"] and not r["fkpath"]["set_ex"]]
    metrics = {a: {
        "set_ex": sum(bool(r[a]["set_ex"]) for r in scorable),
        "bag_ex": sum(bool(r[a]["bag_ex"]) for r in scorable),
        "calls": sum(r[a]["calls"] for r in rows),
        "seconds": round(sum(r[a]["seconds"] for r in rows), 2)
    } for a in ARMS}
    screen = (
        len(scorable) == N
        and len(rescues) - len(losses) >= 2
        and len(losses) <= 1
        and len({r["db_id"] for r in rescues}) >= 2
        and metrics["fkpath"]["bag_ex"] >= metrics["original"]["bag_ex"]
    )
    return {
        "protocol": PROTOCOL, "total": N, "scorable": len(scorable),
        "physical_dbs": len(dbs), "digest": DIGEST,
        "arms": metrics, "rescues": [r["id"] for r in rescues],
        "regressions": [r["id"] for r in losses],
        "cases_with_source_only_fk":
            sum(r["source_route_info"]["visible_new_fks"]>0 for r in rows),
        "positive_screen_only": screen, "promotion_allowed": False,
        "not_BIRD": True, "training": False, "cases": rows
    }


def main():
    p = argparse.ArgumentParser()
    for key in ("train", "dev", "db-root", "manifest", "tables", "output"):
        p.add_argument("--" + key, type=Path)
    p.add_argument("--shard", type=int)
    p.add_argument("--inputs", type=Path, nargs="+")
    a = p.parse_args()
    if a.output is None or a.output.exists():
        raise FileExistsError("output must be new")
    metadata = ({str(e['db_id']):e for e in json.loads(a.tables.read_text())}
                if a.tables is not None else None)
    if a.inputs:
        result = aggregate([json.loads(x.read_text()) for x in a.inputs])
        print("PUBLISHED_FK_QWEN_FINAL", json.dumps(
            {k: v for k, v in result.items() if k!="cases"}, sort_keys=True), flush=True)
    elif a.manifest:
        result = one_shard(a.manifest, a.train, a.dev, a.db_root, a.shard, metadata)
    else:
        if any(x is None for x in (a.train, a.dev, a.db_root, a.tables)):
            p.error("source preflight needs train/dev/db-root/tables")
        result = freeze(json.loads(a.train.read_text()),
                        json.loads(a.dev.read_text()), a.db_root, metadata)
        print("PUBLISHED_FK_SOURCE_FREEZE", result["cases_sha256"],
              sorted({c["db_id"] for c in result["cases"]}), flush=True)
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")


if __name__ == "__main__":
    main()
