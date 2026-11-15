"""Post-hoc, gold-sealed BIRD100 replay of the EXACT frozen Spider projection rule.

Historical original Qwen3.5-9B Guarded predictions are retained unchanged.
No Qwen calls, no SQL training, no new model. Physical SQLite source is pinned.
This BIRD cohort has previously been inspected, so even positive results are
NOT an independent OOD proof; compare exact paired errors/regressions honestly.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256, file_digest
import json
from pathlib import Path

from evaluation.bird_loader import load_bird_json
from evaluation.projection_repair import repair
from evaluation.run_bird import _resolve_database
from semantic_sql.execution import SQLExecutionError, execute_readonly
from semantic_sql.verifier import verify_sql_preflight

PROTOCOL = "frozen-spider-projection-original-bird100-posthoc-v1"
LF_SHA = "ffd8018378ddb1a8794753e0a31cfc81862ff7318a5184c22f3dc4ce03a03feb"
CRLF_SHA = "0310b7ca2dfb6247234e2073f93e7a75106ee41ba55231f75d06d2ef68efe2ec"
DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
TREATMENT_SOURCE_SHA = "fa6ad187c2c0c331753f1cb1ddbd0da7840f23b1"
CAP = 50000
N_SHARDS = 4


def execute(db: Path, sql: str | None):
    if not sql:
        return None
    try:
        return execute_readonly(db, sql, max_rows=CAP)
    except (SQLExecutionError, ValueError):
        return None


def row_replay(row: dict, gold_sql: str, db: Path) -> dict:
    baseline = row["guarded"]
    original_sql = baseline["final_sql"]
    question = row["question"]
    ok = baseline["status"] == "ok"
    original = execute(db, original_sql) if ok else None
    reference = execute(db, gold_sql)
    scorable = reference is not None and not reference.truncated
    valid_baseline = original is not None and not original.truncated
    old_correct = bool(scorable and valid_baseline and
                       set(map(tuple, original.rows)) == set(map(tuple, reference.rows)))
    if old_correct != bool(baseline["official_ex"]):
        raise ValueError(f"historical Guarded EX mismatch {row['case_id']}")
    proposed_sql, trigger_reason = repair(question, original_sql or "")
    triggered = (ok and valid_baseline and proposed_sql != original_sql)
    changed = None
    if triggered and not verify_sql_preflight(db, proposed_sql).ok:
        triggered, trigger_reason = False, "preflight_rejected"
    if triggered:
        changed = execute(db, proposed_sql)
        if changed is None or changed.truncated:
            triggered, trigger_reason = False, "execution_rejected"
    current = changed if triggered else original
    current_sql = proposed_sql if triggered else original_sql
    usable = current is not None and not current.truncated
    new_correct = bool(scorable and usable and
                       set(map(tuple, current.rows)) == set(map(tuple, reference.rows)))
    old_bag = bool(scorable and valid_baseline and
                   Counter(map(tuple, original.rows)) == Counter(map(tuple, reference.rows)))
    new_bag = bool(scorable and usable and
                   Counter(map(tuple, current.rows)) == Counter(map(tuple, reference.rows)))
    return {
        "case_id": str(row["case_id"]), "db_id": str(row["database_id"]),
        "question": question, "scorable": scorable,
        "status": baseline["status"],
        "original_sql": original_sql,
        "repaired_sql": current_sql,
        "triggered": triggered,
        "trigger_reason": trigger_reason,
        "before": {"set_ex": old_correct, "bag_ex": old_bag},
        "after": {"set_ex": new_correct, "bag_ex": new_bag},
        "physical_db_sha256": db_file_sha(db)
    }


def db_file_sha(db: Path) -> str:
    with db.open("rb") as fh:
        return file_digest(fh, "sha256").hexdigest()


def run_shard(source: Path, artifact: Path, root: Path, reference: Path, shard: int) -> dict:
    if shard not in range(N_SHARDS):
        raise ValueError("invalid historical shard")
    source_bytes = source.read_bytes()
    if sha256(source_bytes).hexdigest() != LF_SHA:
        raise ValueError("original BIRD LF source changed")
    if sha256(source_bytes.replace(b"\n", b"\r\n")).hexdigest() != CRLF_SHA:
        raise ValueError("historical CRLF source changed")
    prev = json.loads(artifact.read_text())
    frozen = json.loads(reference.read_text())
    expected = [str(c["case_id"]) for i, c in enumerate(frozen["cases"]) if i % 4 == shard]
    cases = prev.get("cases") or []
    if (
        frozen.get("status") != "complete" or frozen.get("sample_size") != 100
        or len(frozen["cases"]) != 100 or len(cases) != 25
        or prev.get("status") not in ("complete", "invalid")
        or prev.get("model_digest") != DIGEST
        or prev.get("source_sha256") != CRLF_SHA
        or [str(c["case_id"]) for c in cases] != expected
        or not all(c.get("initial_sql_identical") is True for c in cases)
    ):
        raise ValueError("BIRD historical case identity/model/protocol mismatch")
    questions, labels = load_bird_json(source)
    case_map = {c.case_id: c for c in questions}
    label_map = {r.case_id: r.sql for r in labels}
    db_cache = {}
    results = []
    for case in cases:
        cid, dbid = str(case["case_id"]), str(case["database_id"])
        if (cid not in case_map or cid not in label_map or
            case_map[cid].database_id != dbid or case_map[cid].question != case["question"]):
            raise ValueError("source question/DB changed: " + cid)
        if dbid not in db_cache:
            db_cache[dbid] = _resolve_database(root, dbid)
        record = row_replay(case, label_map[cid], db_cache[dbid])
        results.append(record)
        print("BIRD_FROZEN_PROJECTION_CASE", shard, cid,
              record["before"]["set_ex"], record["after"]["set_ex"],
              record["triggered"], flush=True)
    return {
        "protocol": PROTOCOL, "digest": DIGEST, "source_sha256": CRLF_SHA,
        "frozen_treatment_source_sha": TREATMENT_SOURCE_SHA,
        "shard": shard, "historical_status": prev["status"],
        "cases": results,
    }


def aggregate(parts: list[dict]) -> dict:
    if (len(parts) != N_SHARDS or sorted(p["shard"] for p in parts) != list(range(N_SHARDS))
        or any(p["protocol"] != PROTOCOL or p["digest"] != DIGEST
               or p["source_sha256"] != CRLF_SHA
               or p["frozen_treatment_source_sha"] != TREATMENT_SOURCE_SHA
               or len(p["cases"]) != 25 for p in parts)):
        raise ValueError("incomplete 4-shard exact replay")
    rows = [r for p in parts for r in p["cases"]]
    if len(rows) != 100 or len({r["case_id"] for r in rows}) != 100:
        raise ValueError("not exact frozen BIRD100 cohort")
    if any(len({r["physical_db_sha256"] for r in rows if r["db_id"] == db}) != 1
           for db in {r["db_id"] for r in rows}):
        raise ValueError("same DB has inconsistent physical bytes")
    scorable = [r for r in rows if r["scorable"]]
    rescues = [r for r in scorable
               if r["after"]["set_ex"] and not r["before"]["set_ex"]]
    regressions = [r for r in scorable
                   if r["before"]["set_ex"] and not r["after"]["set_ex"]]
    summary = {arm: {
        "set_ex": sum(r[arm]["set_ex"] for r in scorable),
        "bag_ex": sum(r[arm]["bag_ex"] for r in scorable)
    } for arm in ("before", "after")}
    invalid = [p["shard"] for p in parts if p["historical_status"] != "complete"]
    # Pre-registration: preserve previously known invalid marker. Never
    # claim a clean official BIRD benchmark just because SQL replay ran.
    return {
        "protocol": PROTOCOL, "total": len(rows), "scorable": len(scorable),
        "physical_dbs": len({r["db_id"] for r in rows}),
        "model_digest": DIGEST,
        "source_sha256": CRLF_SHA,
        "frozen_treatment_source_sha": TREATMENT_SOURCE_SHA,
        "arms": summary, "triggered": sum(r["triggered"] for r in rows),
        "rescues": [r["case_id"] for r in rescues],
        "regressions": [r["case_id"] for r in regressions],
        "rescued_databases": sorted({r["db_id"] for r in rescues}),
        "original_invalid_shards": invalid,
        "replay_matches_historical_guarded": summary["before"]["set_ex"] == 39,
        "positive_screen_only": len(scorable) == 100
           and len(rescues) - len(regressions) >= 2
           and not regressions
           and len({r["db_id"] for r in rescues}) >= 2
           and summary["after"]["bag_ex"] >= summary["before"]["bag_ex"],
        "promotion_allowed": False,
        "qwen_calls": 0, "model_training": False,
        "posthoc_exposed_source": True, "not_new_independent_inference": True,
        "cases": rows,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path)
    p.add_argument("--artifact", type=Path)
    p.add_argument("--db-root", type=Path)
    p.add_argument("--reference", type=Path, default=Path("runs/bird-paired-dev-v5/paired.json"))
    p.add_argument("--shard", type=int)
    p.add_argument("--inputs", type=Path, nargs="+")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("refuse result overwrite")
    if args.inputs:
        output = aggregate([json.loads(x.read_text()) for x in args.inputs])
        print("BIRD100_FROZEN_PROJECTION_FINAL", json.dumps(
            {k: v for k, v in output.items() if k != "cases"}, sort_keys=True), flush=True)
    else:
        if any(x is None for x in (args.source, args.artifact, args.db_root, args.shard)):
            p.error("physical shard needs --source, --artifact, --db-root, --shard")
        output = run_shard(args.source, args.artifact, args.db_root, args.reference, args.shard)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
