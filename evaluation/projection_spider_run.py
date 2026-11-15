"""Source-frozen paired strict-EX test of projection-only repair on new Spider DBs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256, file_digest
import json
from pathlib import Path
import re
from time import monotonic

from evaluation.run_bird import CountingProvider, _ollama_model_digest
from evaluation.schema_context import select_schema_context
from evaluation.projection_repair import eligible, repair
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.execution import SQLExecutionError, execute_readonly
from semantic_sql.inference import run_guarded
from semantic_sql.providers import OllamaProvider
from semantic_sql.verifier import verify_sql_preflight

P = "qwen9b-projection-spider-dev-fresh-v1"
M = "qwen3.5:9b-q4_K_M"
D = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
SOURCE = "00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b"
EXCLUDED = frozenset({
    "car_1", "employee_hire_evaluation", "flight_2", "museum_visit",
    "singer", "tvshow", "pets_1", "dog_kennels", "orchestra",
    "voter_1", "course_teach", "poker_player",
})
CASES = 18
NDB = 6
N_SHARDS = 3


def physical(root: Path, dbid: str) -> Path:
    if not re.fullmatch(r"[a-zA-Z0-9_]+", dbid):
        raise ValueError("illegal database ID")
    file = root / dbid / (dbid + ".sqlite")
    if not file.is_file():
        raise FileNotFoundError(file)
    return file


def freeze(train: list[dict], dev: list[dict], root: Path) -> dict:
    old = {str(x["db_id"]) for x in train} | EXCLUDED
    groups = defaultdict(dict)
    for i, row in enumerate(dev):
        dbid = str(row["db_id"])
        question = " ".join(str(row.get("question") or "").split())
        if dbid in old or not eligible(question):
            continue
        groups[dbid].setdefault(question.casefold(), {
            "id": f"{dbid}:{i}", "index": i, "db_id": dbid, "question": question
        })
    def key(s: str) -> str:
        return sha256((P + ":" + s).encode()).hexdigest()
    candidates = sorted(
        (db for db, items in groups.items() if len(items) >= 3), key=lambda d: key("db:" + d)
    )
    rows = []
    for db in candidates:
        if len(DatabaseCatalog.from_sqlite(physical(root, db)).tables) < 2:
            continue
        rows.extend(sorted(groups[db].values(), key=lambda r: key(r["id"]))[:3])
        if len(rows) == CASES:
            break
    if len(rows) != CASES or len({r["db_id"] for r in rows}) != NDB:
        raise ValueError(f"too few new gold-blind DBs: {len(rows)} rows from "
                         f"{len({r['db_id'] for r in rows})} DBs")
    if any("query" in r for r in rows):
        raise AssertionError("gold leaked into frozen task selection")
    return {
        "protocol": P, "digest": D, "archive_sha256": SOURCE,
        "cohort_sha256": sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        "size": CASES, "dbs": NDB, "gold_blind": True,
        "excluded_prior_physical_dbs": sorted(EXCLUDED),
        "cases": rows,
    }


def one_shard(manifest: Path, trainfile: Path, devfile: Path, root: Path,
              shard: int) -> dict:
    if shard not in range(N_SHARDS):
        raise ValueError("invalid shard")
    train, dev = json.loads(trainfile.read_text()), json.loads(devfile.read_text())
    frozen = json.loads(manifest.read_text())
    if frozen != freeze(train, dev, root):
        raise ValueError("cohort or physical source drift")
    if _ollama_model_digest("http://127.0.0.1:11434", M) != D:
        raise ValueError("Qwen digest mismatch")
    records = []
    for row in frozen["cases"][shard * 6:(shard + 1) * 6]:
        db = physical(root, row["db_id"])
        schema, _ = select_schema_context(
            DatabaseCatalog.from_sqlite(db), root, row["db_id"], row["question"]
        )
        provider = CountingProvider(OllamaProvider(
            model=M, timeout_s=600, temperature=0, seed=42, num_predict=700
        ))
        started = monotonic()
        baseline = run_guarded(
            database=db, provider=provider, question=row["question"],
            schema_context=schema, max_repairs=2, max_rows=50_000
        )
        sql = baseline.candidate.sql if baseline.candidate else ""
        projected, trigger = repair(row["question"], sql)
        triggered = (baseline.status == "ok" and not baseline.truncated
                     and projected != sql)
        observation = None
        if triggered and not verify_sql_preflight(db, projected).ok:
            triggered, trigger = False, "rejected_by_preflight"
        if triggered:
            try:
                observation = execute_readonly(db, projected, max_rows=50_000)
                if observation.truncated:
                    triggered, trigger = False, "result_truncated"
            except SQLExecutionError:
                triggered, trigger = False, "rejected_by_execution"
        record = {
            "id": row["id"], "db_id": row["db_id"],
            "question": row["question"],
            "baseline": {
                "sql": sql, "status": baseline.status, "truncated": baseline.truncated,
                "rows": baseline.rows, "calls": provider.total_calls,
                "seconds": round(monotonic() - started, 2)
            },
            "repair": {
                "sql": projected if triggered else sql,
                "triggered": triggered, "reason": trigger,
                "status": "ok" if triggered else baseline.status,
                "truncated": False if triggered else baseline.truncated,
                "rows": observation.rows if triggered else baseline.rows,
                "calls": provider.total_calls,
            },
        }
        records.append(record)
        print("PAIRED_SQL_FROZEN", row["id"], trigger, triggered, flush=True)
    # Source references are read only after both arm outputs are fixed.
    for row in records:
        original = next(x for x in frozen["cases"] if x["id"] == row["id"])
        db = physical(root, row["db_id"])
        try:
            gold = execute_readonly(db, str(dev[original["index"]]["query"]),
                                    max_rows=50_000)
            row["scorable"] = not gold.truncated
        except SQLExecutionError:
            gold, row["scorable"] = None, False
        for arm in ("baseline", "repair"):
            item = row[arm]
            if row["scorable"]:
                valid = item["status"] == "ok" and not item["truncated"]
                item["set_ex"] = bool(valid and set(map(tuple, item["rows"])) == set(gold.rows))
                item["bag_ex"] = bool(valid and Counter(map(tuple, item["rows"])) == Counter(gold.rows))
            else:
                item["set_ex"], item["bag_ex"] = None, None
            del item["rows"]
        with db.open("rb") as f:
            row["db_sha256"] = file_digest(f, "sha256").hexdigest()
    return {"protocol": P, "digest": D, "cohort_sha256": frozen["cohort_sha256"],
            "shard": shard, "cases": records}


def aggregate(shards: list[dict]) -> dict:
    if (len(shards) != N_SHARDS
        or sorted(s["shard"] for s in shards) != list(range(N_SHARDS))
        or any(s["protocol"] != P or s["digest"] != D
               or len(s["cases"]) != CASES // N_SHARDS for s in shards)
        or len({s["cohort_sha256"] for s in shards}) != 1):
        raise ValueError("incomplete or mismatched paired result matrix")
    rows = [r for shard in sorted(shards, key=lambda s: s["shard"])
            for r in shard["cases"]]
    dbids = {r["db_id"] for r in rows}
    if len(rows) != CASES or len({r["id"] for r in rows}) != CASES or len(dbids) != NDB:
        raise ValueError("missing or duplicate frozen source")
    if any(len({r["db_sha256"] for r in rows if r["db_id"] == db}) != 1
           for db in dbids):
        raise ValueError("physical database changed across shards")
    scorable = [r for r in rows if r["scorable"]]
    rescues = [r for r in scorable
               if r["repair"]["set_ex"] and not r["baseline"]["set_ex"]]
    regression = [r for r in scorable
                  if r["baseline"]["set_ex"] and not r["repair"]["set_ex"]]
    metrics = {arm: {
        "set_ex": sum(bool(r[arm]["set_ex"]) for r in scorable),
        "bag_ex": sum(bool(r[arm]["bag_ex"]) for r in scorable),
        "calls": sum(r[arm]["calls"] for r in rows),
    } for arm in ("baseline", "repair")}
    positive = (len(scorable) == CASES and len(rescues) - len(regression) >= 2
                and len(regression) == 0
                and len({r["db_id"] for r in rescues}) >= 2
                and metrics["repair"]["bag_ex"] >= metrics["baseline"]["bag_ex"])
    return {
        "protocol": P, "total": CASES, "scorable": len(scorable),
        "physical_dbs": len(dbids), "digest": D,
        "arms": metrics,
        "triggers": sum(r["repair"]["triggered"] for r in rows),
        "rescues": [r["id"] for r in rescues],
        "regressions": [r["id"] for r in regression],
        "positive_screen_only": positive,
        "promotion_allowed": False, "training": False,
        "not_BIRD_or_official_Spider": True, "cases": rows
    }


def main() -> None:
    p = argparse.ArgumentParser()
    for key in ("train", "dev", "db-root", "manifest", "output"):
        p.add_argument("--" + key, type=Path)
    p.add_argument("--shard", type=int)
    p.add_argument("--inputs", nargs="+", type=Path)
    a = p.parse_args()
    if a.output is None or a.output.exists():
        raise FileExistsError("new output path required")
    if a.inputs:
        result = aggregate([json.loads(f.read_text()) for f in a.inputs])
        print("PROJECTION_SQL_FINAL", json.dumps(
            {k: v for k, v in result.items() if k != "cases"}, sort_keys=True), flush=True)
    elif a.manifest:
        result = one_shard(a.manifest, a.train, a.dev, a.db_root, a.shard)
    else:
        result = freeze(json.loads(a.train.read_text()),
                        json.loads(a.dev.read_text()), a.db_root)
        print("PROJECTION_SQL_FREEZE", result["cohort_sha256"], flush=True)
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
