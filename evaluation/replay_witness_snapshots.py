"""Verify generated probe and answer SQL on one physical snapshot per corpus."""

import argparse
from collections import Counter
from hashlib import file_digest
import json
from pathlib import Path

from evaluation.qwen_sql_witness import ARMS, MAX_ROWS, aggregate, sample
from evaluation.run_bird import _write
from semantic_sql.execution import SQLExecutionError, execute_readonly
from semantic_sql.verifier import verify_sql_preflight


def replay(shards: list[dict], databases: dict[str, Path]) -> dict:
    original_digests = {}
    verified_probes = 0
    verified_outcomes = 0
    for shard in shards:
        db_id = shard["dataset"]
        db = databases[db_id]
        original_digests[f"{db_id}-{shard['shard']}"] = shard["db_sha"]
        with db.open("rb") as file:
            shard["db_sha"] = file_digest(file, "sha256").hexdigest()
        for row in shard["cases"]:
            seen = sample(db, row["probe"])
            if seen != row["observation"]:
                raise ValueError(f"actual probe input drifted on physical replay: {db_id}:{row['id']}")
            verified_probes += 1
            references = row["reference_sql"]
            references = [references] if isinstance(references, str) else references
            tuples = []
            for reference in references:
                try:
                    expected = execute_readonly(db, reference, max_rows=MAX_ROWS)
                except SQLExecutionError:
                    continue
                if not expected.truncated:
                    tuples.append(expected.rows)
            if bool(tuples) != row["scorable"] or len(tuples) != row["scorable_references"]:
                raise ValueError(f"source reference changed: {db_id}:{row['id']}")
            for arm in ARMS:
                result = row[arm]
                if verify_sql_preflight(db, result["sql"]).ok:
                    try:
                        actual = execute_readonly(db, result["sql"], max_rows=MAX_ROWS)
                        status, capped = "ok", actual.truncated
                    except SQLExecutionError:
                        actual, status, capped = None, "execution_failed", False
                else:
                    actual, status, capped = None, "verification_failed", False
                if status != result["status"] or capped != result["capped"]:
                    raise ValueError(f"prediction execution changed: {db_id}:{row['id']}:{arm}")
                for metric, signature in (("set_ex", set), ("bag_ex", Counter)):
                    score = bool(actual and not capped and any(
                        signature(actual.rows) == signature(ref) for ref in tuples
                    )) if tuples else None
                    if score != result[metric]:
                        raise ValueError(f"{metric} drift: {db_id}:{row['id']}:{arm}")
                verified_outcomes += 1
    summary = aggregate(shards)
    summary["physical_replay"] = {
        "original_shard_db_sha256": original_digests, "matched_probe_payloads": verified_probes,
        "matched_execution_and_strict_outcomes": verified_outcomes,
        "new_model_calls": 0,
    }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--chinook", type=Path, required=True)
    parser.add_argument("--sakila", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    out = replay([json.loads(p.read_text()) for p in args.inputs],
                 {"chinook": args.chinook, "sakila": args.sakila})
    _write(args.output, out)
    print("WITNESS_REPLAY_FINAL", json.dumps({k:v for k,v in out.items() if k != "cases"}))
