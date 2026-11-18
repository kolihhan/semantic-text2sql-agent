"""Bounded, source-locked recovery probe for original BIRD DEV500 K8 shard 7.

Never sample a new model candidate. The existing unbounded scoring protocol is
not modified. If ANY original candidate/reference query times out, mark this
shard CENSORED and forbid merging it into the main unbounded DEV500 metric.
No performance result can be promoted from this diagnostic.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from hashlib import sha256
from pathlib import Path
from typing import Any

from evaluation.kcache_protocol import (
    MODEL_DIGEST, PROTOCOL, SOURCE_SHA256, MANIFEST, candidate_outcome,
    majority_selector, read_frozen_manifest, sample_correctness, source_cases,
)
from evaluation.qwen9b_kcache import validate_cache, prepare_prompt
from evaluation.run_bird import _resolve_database


def _worker(send: Any, db: str, sql: str) -> None:
    try:
        send.send(candidate_outcome(Path(db), sql))
    except BaseException as exc:
        # No input SQL or DB contents exported to the parent error record.
        send.send({"valid": False, "reason": "worker_" + type(exc).__name__,
                   "signature": None, "set_hash": None, "bag_hash": None})
    finally:
        send.close()


def bounded_outcome(db: Path, sql: str, seconds: float) -> dict:
    if not 0 < seconds <= 120:
        raise ValueError("bounded execution seconds must be in (0,120]")
    parent, child = mp.get_context("spawn").Pipe(duplex=False)
    proc = mp.get_context("spawn").Process(target=_worker, args=(child, str(db), sql))
    try:
        proc.start()
        child.close()
        if not parent.poll(seconds):
            proc.terminate()
            proc.join(timeout=3)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=3)
            return {"valid": False, "reason": "query_timeout",
                    "signature": None, "set_hash": None, "bag_hash": None}
        try:
            result = parent.recv()
        except EOFError:
            raise RuntimeError("bounded scoring worker exited without output")
        proc.join(timeout=3)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=3)
        return result
    finally:
        parent.close()
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=3)


def score_one(db: Path, cache: dict, gold_sql: str, seconds: float) -> dict:
    if cache["k"] != 8 or len(cache["samples"]) != 8:
        raise ValueError("not a complete original K8 case")
    outcomes = [bounded_outcome(db, sample["sql"], seconds) for sample in cache["samples"]]
    # Lock the selected candidate without access to reference SQL.
    chosen = majority_selector(outcomes)["index"]
    gold = bounded_outcome(db, gold_sql, seconds)
    set_ok = [sample_correctness(x, gold) for x in outcomes]
    bag_ok = [sample_correctness(x, gold, bag=True) for x in outcomes]
    return {
        "case_id": cache["case_id"], "db_id": cache["db_id"], "k": 8,
        "first_set_ex": bool(set_ok[0]), "first_bag_ex": bool(bag_ok[0]),
        "pass1_empirical_set": sum(set_ok) / 8,
        "passK_oracle_set": bool(any(set_ok)),
        "passK_oracle_bag": bool(any(bag_ok)),
        "selected_set_ex": bool(set_ok[chosen]),
        "selected_bag_ex": bool(bag_ok[chosen]),
        "selected_index": chosen,
        "vote_agreement": majority_selector(outcomes)["agreement"],
        "unique_valid_result_sets": majority_selector(outcomes)["unique_valid_result_sets"],
        "candidate_execution_success": sum(x["valid"] for x in outcomes),
        "gold_scorable": bool(gold["valid"]),
        "zero_gold_in_selector": True,
        "candidate_timeout_count": sum(x.get("reason") == "query_timeout" for x in outcomes),
        "gold_timeout": gold.get("reason") == "query_timeout",
    }


def bounded_shard(cache_file: Path, source: Path, dbroot: Path,
                  manifest_file: Path, seconds: float) -> dict:
    manifest = read_frozen_manifest(manifest_file)
    rows = source_cases(source, manifest, split="dev")
    cache = json.loads(cache_file.read_text("utf-8"))
    validate_cache(cache, rows)
    if cache["shard"] != 7 or cache["shards"] != 32 or cache["k"] != 8:
        raise ValueError("only the original 16-case frozen shard 7 is allowed")
    by_id = {str(x["question_id"]): x for x in rows}
    records = []
    for case in cache["cases"]:
        source_row = by_id[case["case_id"]]
        db = _resolve_database(dbroot, case["db_id"])
        _, expected_prompt_sha = prepare_prompt(source_row, dbroot, {})
        if (case["db_id"] != str(source_row["db_id"]) or
            case["question_sha256"] != sha256(str(source_row["question"]).encode()).hexdigest() or
            case["prompt_sha256"] != expected_prompt_sha):
            raise ValueError("frozen case/prompt drift")
        result = score_one(db, case, str(source_row["SQL"]), seconds)
        records.append(result)
        print("BOUNDED_ORIGINAL_K8_CASE", len(records), "/16",
              "candidate_timeouts", result["candidate_timeout_count"],
              "gold_timeout", result["gold_timeout"], flush=True)
    timeouts = sum(x["candidate_timeout_count"] for x in records)
    gold_timeouts = sum(x["gold_timeout"] for x in records)
    censored = bool(timeouts or gold_timeouts or any(not x["gold_scorable"] for x in records))
    return {
        "status": "SCORING_TIMEOUT_CENSORED_SOURCE_ONLY" if censored else "SCORED_FROM_IMMUTABLE_CACHE_ONLY",
        "protocol": PROTOCOL, "source_sha256": SOURCE_SHA256,
        "model_digest": MODEL_DIGEST, "shard": 7, "shards": 32, "k": 8,
        "model_calls": 0, "selector_never_sees_gold": True,
        "query_timeout_seconds": seconds, "bounded_recovery_only": True,
        "candidate_timeouts": timeouts, "reference_timeouts": gold_timeouts,
        "claim_boundary": (
            "If censored, no main DEV500 EX aggregation or promotion: "
            "bounded worker execution differs from originally unbounded source scoring."
        ),
        "cases": records,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--db-root", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=Path(MANIFEST))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-query-seconds", type=float, default=12.)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError("immutable diagnostic must never be overwritten")
    result = bounded_shard(a.cache, a.source, a.db_root, a.manifest, a.per_query_seconds)
    a.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("BOUNDED_SHARD7_RESULT",
          json.dumps({k: v for k, v in result.items() if k != "cases"}, sort_keys=True),
          flush=True)


if __name__ == "__main__":
    main()
