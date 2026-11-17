"""Offline replay of the ORIGINAL four Qwen9B BIRD100 shards, no model calls.

Diagnoses why the 2026-10-05 historical Action failed after all 100
predictions: gold execution errors, row caps, scorer drift, and repair outcomes.
Never call this a clean benchmark rerun: candidates were generated previously.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from evaluation.bird_loader import load_bird_json
from evaluation.run_bird import _resolve_database
from semantic_sql.execution import SQLExecutionError, execute_readonly

PROTOCOL = "qwen9b-offline-replay-shard-v1"
SOURCE_LF_SHA256 = "ffd8018378ddb1a8794753e0a31cfc81862ff7318a5184c22f3dc4ce03a03feb"
SOURCE_CRLF_SHA256 = "0310b7ca2dfb6247234e2073f93e7a75106ee41ba55231f75d06d2ef68efe2ec"
MODEL_DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
CAP = 50_000


def _execute(db: Path, sql: str | None, *, max_rows: int):
    if not sql:
        return None, None
    try:
        return execute_readonly(db, sql, max_rows=max_rows), None
    except (SQLExecutionError, ValueError) as exc:
        return None, type(exc).__name__


def replay_case(case: dict, database: Path, gold_sql: str, *, max_rows: int = CAP) -> dict:
    """Recompute strict BIRD set-of-tuples equality; preserve original labels."""
    gold, gold_error = _execute(database, gold_sql, max_rows=max_rows)
    gold_truncated = bool(gold is not None and gold.truncated)
    scorable = bool(gold is not None and not gold_truncated)
    output = {
        "case_id": str(case["case_id"]), "db_id": str(case["database_id"]),
        "reference_error": gold_error, "reference_truncated": gold_truncated,
        "reference_scorable": scorable,
        "reference_rows": len(gold.rows) if gold is not None else None,
        "original": {
            "direct": bool(case["direct"]["official_ex"]),
            "guarded": bool(case["guarded"]["official_ex"]),
        },
        "original_guarded_status": case["guarded"]["status"],
        "original_guarded_repair_attempts": int(case["guarded"].get("repair_attempts",0)),
    }
    for arm in ("direct", "guarded"):
        old = case[arm]
        # Original Guarded refused runs must remain refused. Replaying their
        # rejected SQL directly would bypass the frozen verifier.
        attempted = old["status"] == "ok"
        predicted, error = _execute(
            database, old["final_sql"] if attempted else None,
            max_rows=max_rows,
        )
        truncated = bool(predicted is not None and predicted.truncated)
        ex = bool(scorable and predicted is not None and not truncated and
                  set(map(tuple,predicted.rows)) == set(map(tuple,gold.rows)))
        output[arm] = {
            "replay_ex": ex, "error": error,
            "truncated": truncated,
            "rows": len(predicted.rows) if predicted is not None else None,
            "both_empty": bool(scorable and predicted is not None and
                               not truncated and len(predicted.rows) == 0 and len(gold.rows) == 0),
        }
    output["original_ex_verified"] = all(
        output["original"][arm] == output[arm]["replay_ex"] for arm in ("direct", "guarded")
    )
    return output


def audit_shard(source: Path, artifact: Path, root: Path, reference: Path, shard: int) -> dict:
    if shard not in range(4):
        raise ValueError("shard must be in [0,3]")
    raw = source.read_bytes()
    if sha256(raw).hexdigest() != SOURCE_LF_SHA256:
        raise ValueError("BIRD original source LF SHA mismatch")
    historical = raw.replace(b"\n", b"\r\n")
    if sha256(historical).hexdigest() != SOURCE_CRLF_SHA256:
        raise ValueError("BIRD original source historical CRLF SHA mismatch")
    candidates = json.loads(artifact.read_text(encoding="utf-8"))
    old = json.loads(reference.read_text(encoding="utf-8"))
    if (old.get("status") != "complete" or old.get("sample_size") != 100
        or len(old.get("cases", [])) != 100):
        raise ValueError("historical frozen 100-case reference missing")
    expected = [str(c["case_id"]) for i,c in enumerate(old["cases"]) if i % 4 == shard]
    records = candidates.get("cases")
    if (candidates.get("status") not in {"complete","invalid"}
        or len(records or []) != 25
        or candidates.get("model_digest") != MODEL_DIGEST
        or candidates.get("source_sha256") != SOURCE_CRLF_SHA256
        or candidates.get("sample_size") != 25
        or [str(c["case_id"]) for c in records] != expected
        or not all(c.get("initial_sql_identical") is True for c in records)):
        raise ValueError("original BIRD shard contents/protocol mismatch")
    cases, labels = load_bird_json(source)
    questions = {c.case_id:c for c in cases}
    sql_by_id = {c.case_id:c.sql for c in labels}
    db_cache = {}
    replayed = []
    for row in records:
        cid, dbid = str(row["case_id"]), str(row["database_id"])
        question = questions.get(cid)
        if question is None or dbid != question.database_id or row["question"] != question.question:
            raise ValueError(f"frozen question/database changed: {cid}")
        if cid not in sql_by_id:
            raise ValueError(f"reference label absent: {cid}")
        if dbid not in db_cache:
            db_cache[dbid] = _resolve_database(root, dbid)
        result = replay_case(row, db_cache[dbid], sql_by_id[cid])
        replayed.append(result)
        print("QWEN9B_ORIGINAL_BIRD_REPLAY",shard,cid,
              "match",result["original_ex_verified"],
              "gold_truncated",result["reference_truncated"],
              "gold_error",result["reference_error"],flush=True)
    return {
        "protocol": PROTOCOL, "shard": shard,
        "source_sha256": SOURCE_CRLF_SHA256, "model_digest": MODEL_DIGEST,
        "original_status": candidates["status"], "cases": replayed,
    }


def aggregate(shards: list[dict], *, expected_digest: str = MODEL_DIGEST,
              expected_source: str = SOURCE_CRLF_SHA256) -> dict:
    if len(shards) != 4 or sorted(s["shard"] for s in shards) != list(range(4)):
        raise ValueError("four distinct complete original artifact shards required")
    if any(s.get("protocol") != PROTOCOL
           or s.get("model_digest") != expected_digest
           or s.get("source_sha256") != expected_source
           or len(s.get("cases", [])) != 25 for s in shards):
        raise ValueError("BIRD original model/source/protocol drift")
    rows = [c for s in shards for c in s["cases"]]
    if len(rows) != 100 or len({c["case_id"] for c in rows}) != 100:
        raise ValueError("duplicate or missing original BIRD cases")
    original_invalid = [s["shard"] for s in shards if s["original_status"] != "complete"]
    status = Counter(c["original_guarded_status"] for c in rows)
    raw_direct = sum(c["original"]["direct"] for c in rows)
    raw_guard = sum(c["original"]["guarded"] for c in rows)
    verified = [c for c in rows if c["original_ex_verified"]]
    capped = [c["case_id"] for c in rows if c["reference_truncated"] or
              c["direct"]["truncated"] or c["guarded"]["truncated"]]
    gold_errors = [c["case_id"] for c in rows if c["reference_error"] is not None]
    mismatches = [c["case_id"] for c in rows if not c["original_ex_verified"]]
    return {
        "protocol": PROTOCOL,
        "source_sha256": expected_source, "model_digest": expected_digest,
        "requested": 100, "unique_cases": len(rows),
        "raw_direct_ex": raw_direct, "raw_guarded_ex": raw_guard,
        "replay_direct_ex": sum(c["direct"]["replay_ex"] for c in rows),
        "replay_guarded_ex": sum(c["guarded"]["replay_ex"] for c in rows),
        "strict_matched_replay_labels": len(verified),
        "original_guarded_status_counts": dict(sorted(status.items())),
        "executable_but_wrong": sum(
            c["original_guarded_status"] == "ok" and not c["original"]["guarded"]
            for c in rows),
        "repair_rescues": sum(
            not c["original"]["direct"] and c["original"]["guarded"] for c in rows),
        "repair_regressions": sum(
            c["original"]["direct"] and not c["original"]["guarded"] for c in rows),
        "invalid_original_shards": original_invalid,
        "gold_error_case_ids": gold_errors,
        "truncated_case_ids": capped,
        "replay_disagreement_case_ids": mismatches,
        "both_empty_correct_cases": sum(c["guarded"]["both_empty"] and c["guarded"]["replay_ex"] for c in rows),
        "clean_replay": not (original_invalid or gold_errors or capped or mismatches),
        "model_calls": 0, "training": False,
        "not_a_new_benchmark_run": True,
        "caveat": "Post-hoc read-only replay of original model-generated SQL, not independent inference, not new score improvement. Failed original shard invalidity remains visible.",
        "cases": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",type=Path)
    ap.add_argument("--artifact",type=Path)
    ap.add_argument("--reference",type=Path,default=Path("runs/bird-paired-dev-v5/paired.json"))
    ap.add_argument("--db-root",type=Path)
    ap.add_argument("--shard",type=int)
    ap.add_argument("--inputs",type=Path,nargs="+")
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():
        raise FileExistsError("refuse overwrite")
    if args.inputs:
        report=aggregate([json.loads(p.read_text(encoding="utf-8")) for p in args.inputs])
        print("QWEN9B_FROZEN100_ORIGINAL_OFFLINE_REPLAY",
              json.dumps({k:v for k,v in report.items() if k!="cases"},sort_keys=True),flush=True)
    else:
        if any(x is None for x in (args.source,args.artifact,args.db_root,args.shard)):
            ap.error("--source --artifact --db-root --shard required")
        report=audit_shard(args.source,args.artifact,args.db_root,args.reference,args.shard)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
