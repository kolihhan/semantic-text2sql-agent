"""Frozen DEV-derived zero-row repair on historically exposed, DB-disjoint TRAIN64.
Candidate generation is physically separate from publisher SQL/outcomes.
This is a transfer DIAGNOSTIC ONLY, never a held-out performance claim.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re

from semantic_sql.execution import execute_readonly, SQLExecutionError
from semantic_sql.verifier import verify_sql_preflight
from scripts.qwen9b_empty_result_targeted_repair import (
    MODEL_DIGEST, MAX_ROWS, check_model, evidence, one_qwen_call,
)
from scripts.transfer_normalization_birdtrain64 import (
    N, PROTECTED, VAL64_DBS, read, write, canonical_sha,
    prepare, validated_databases,
)

PROTOCOL = "DEV-FROZEN-ZERO-ROW-QWEN9B-TRANSFER-TRAIN64-V1"
MAX_CALLS = 30

def validate_unlabeled(data):
    if (data.get("status") != "FROZEN_TRAIN64_QWEN_SQL_NO_GOLD"
            or data.get("n") != N or data.get("model_digest") != MODEL_DIGEST
            or set(data.get("dbs", [])) != VAL64_DBS
            or len(data.get("cases", [])) != N):
        raise ValueError("wrong Qwen9B original TRAIN64 source")
    if set(data["dbs"]) & PROTECTED:
        raise ValueError("protected DB exposure prohibited")
    allowed = {"source_index", "db_id", "question", "sql", "prompt_sha256"}
    if any(set(c) != allowed for c in data["cases"]):
        raise ValueError("gold or outcome contamination in inference input")
    if len({int(x["source_index"]) for x in data["cases"]}) != N:
        raise ValueError("source ID drift")

def freeze(unlabeled: Path, root: Path, output: Path):
    source = read(unlabeled)
    validate_unlabeled(source)
    paths = validated_databases(root)
    check_model()
    output_rows, calls, seconds = [], 0, 0.0
    for case in sorted(source["cases"], key=lambda x: int(x["source_index"])):
        dbid, sql = case["db_id"], case["sql"]
        path = paths[dbid]
        row = {
            "source_index": case["source_index"], "db_id": dbid,
            "original_sha": sha256(sql.encode()).hexdigest(),
            "status": "ABSTAIN", "model_called": False, "candidate_sql": None,
        }
        if not sql or not re.search(r"\bWHERE\b|\bHAVING\b", sql, re.I):
            row["status"] = "NO_FILTER"
        elif not verify_sql_preflight(path, sql).ok:
            row["status"] = "ORIGINAL_INVALID"
        else:
            try:
                old = execute_readonly(path, sql, max_rows=MAX_ROWS)
            except SQLExecutionError:
                row["status"] = "ORIGINAL_EXEC_ERROR"
            else:
                if old.truncated or old.rows:
                    row["status"] = "ORIGINAL_NONEMPTY"
                elif calls >= MAX_CALLS:
                    row["status"] = "MODEL_BUDGET_LIMIT"
                else:
                    schema, observations = evidence(path, sql)
                    alternative, sec, outcome = one_qwen_call(
                        case["question"], sql, schema, observations
                    )
                    calls += 1
                    seconds += sec
                    row["model_called"] = True
                    if outcome != "OK":
                        row["status"] = outcome
                    elif not alternative or alternative.strip().rstrip(";") == sql.strip().rstrip(";"):
                        row["status"] = "UNCHANGED_OR_BAD_OUTPUT"
                    elif not verify_sql_preflight(path, alternative).ok:
                        row["status"] = "CANDIDATE_PREFLIGHT_FAIL"
                    else:
                        try:
                            changed = execute_readonly(path, alternative, max_rows=MAX_ROWS)
                            if changed.rows and not changed.truncated:
                                row["status"] = "FROZEN_CANDIDATE"
                                row["candidate_sql"] = alternative
                            else:
                                row["status"] = "CANDIDATE_EMPTY_OR_TRUNCATED"
                        except SQLExecutionError:
                            row["status"] = "CANDIDATE_EXEC_ERROR"
        output_rows.append(row)
        print("TRAIN64_ZERO_ROW_CASE", row["source_index"], row["status"], flush=True)
    packet = {
        "status": "FROZEN_NO_GOLD", "protocol": PROTOCOL,
        "n": N, "dbs": sorted(VAL64_DBS), "model_digest": MODEL_DIGEST,
        "source_digest": canonical_sha(source), "no_gold": True,
        "max_calls": MAX_CALLS, "calls": calls,
        "model_seconds": round(seconds, 2), "cases": output_rows,
    }
    write(output, packet)
    print("TRAIN64_ZERO_ROW_FROZEN", json.dumps({
        "n": N, "calls": calls,
        "proposals": sum(x["status"] == "FROZEN_CANDIDATE" for x in output_rows),
    }), flush=True)

def paired_score(frozen: Path, unlabeled: Path, labels: Path, root: Path, output: Path):
    packet, source, sealed = read(frozen), read(unlabeled), read(labels)
    validate_unlabeled(source)
    if (packet.get("status") != "FROZEN_NO_GOLD"
            or packet.get("protocol") != PROTOCOL or packet.get("no_gold") is not True
            or packet.get("source_digest") != canonical_sha(source)
            or packet.get("n") != N or packet.get("model_digest") != MODEL_DIGEST
            or packet.get("dbs") != sorted(VAL64_DBS)):
        raise ValueError("inference not frozen or source modified")
    if (sealed.get("status") != "SEALED_TRAIN64_PUBLISHER_GOLD_AND_EX"
            or sealed.get("n") != N or len(sealed.get("cases", [])) != N):
        raise ValueError("sealed publisher labels drift")
    by_id = {x["source_index"]: x for x in source["cases"]}
    gold_by_id = {x["source_index"]: x for x in sealed["cases"]}
    rows = packet["cases"]
    if len(rows) != N or len(by_id) != N or len(gold_by_id) != N:
        raise ValueError("unmatched cohort")
    if {x["source_index"] for x in rows} != set(by_id):
        raise ValueError("missing or duplicated frozen case")
    paths = validated_databases(root)
    totals, dbstats, drift = Counter(), defaultdict(lambda: Counter()), []
    changed = []
    for row in rows:
        idx = row["source_index"]
        item = by_id[idx]
        reference = gold_by_id[idx]
        dbid = item["db_id"]
        if (row["db_id"] != dbid or dbid not in VAL64_DBS
                or reference["db_id"] != dbid
                or sha256(item["sql"].encode()).hexdigest() != row["original_sha"]):
            raise ValueError("row identity mismatch")
        path = paths[dbid]
        try:
            gold = execute_readonly(path, reference["gold_sql"], max_rows=MAX_ROWS)
            if gold.truncated:
                raise ValueError("gold result truncated " + str(idx))
        except SQLExecutionError as exc:
            raise ValueError("publisher SQL failed on physical DB " + str(idx)) from exc
        def evaluate(sql):
            try:
                result = execute_readonly(path, sql, max_rows=MAX_ROWS)
                if result.truncated:
                    return False, False
                return set(result.rows) == set(gold.rows), Counter(result.rows) == Counter(gold.rows)
            except SQLExecutionError:
                return False, False
        before_set, before_bag = evaluate(item["sql"])
        if (before_set != bool(reference["original_ex"])
                or before_bag != bool(reference["original_bag"])):
            drift.append(idx)
        if row["status"] == "FROZEN_CANDIDATE":
            if not row["candidate_sql"]:
                raise ValueError("missing candidate SQL")
            after_set, after_bag = evaluate(row["candidate_sql"])
            totals["proposals"] += 1
            changed.append({"source_index": idx, "db_id": dbid,
                            "before_set": before_set, "after_set": after_set,
                            "before_bag": before_bag, "after_bag": after_bag})
        else:
            if row["candidate_sql"] is not None:
                raise ValueError("noncandidate has SQL")
            after_set, after_bag = before_set, before_bag
        totals["base_set"] += int(before_set)
        totals["base_bag"] += int(before_bag)
        totals["new_set"] += int(after_set)
        totals["new_bag"] += int(after_bag)
        totals["rescues"] += int(not before_set and after_set)
        totals["regressions"] += int(before_set and not after_set)
        dbstats[dbid]["base"] += int(before_set)
        dbstats[dbid]["new"] += int(after_set)
        dbstats[dbid]["n"] += 1
    if drift:
        raise ValueError("original historical EX/BAG baseline drift for source indices: "
                         + ",".join(map(str, drift[:12])))
    improved = sorted(db for db, counts in dbstats.items() if counts["new"] > counts["base"])
    data = {
        "status": "EXPOSED_DB_DISJOINT_TRAIN64_TRANSFER_DIAGNOSTIC_ONLY",
        "protocol": PROTOCOL, "n": N, "model_calls": packet["calls"],
        "model_seconds": packet["model_seconds"], "scores": dict(totals),
        "physical_dbs": sorted(VAL64_DBS), "improved_dbs": improved,
        "per_db": {db: dict(v) for db, v in sorted(dbstats.items())},
        "changed_case_details": changed,
        "research_gate": "REJECT_OR_HOLD_NO_PRISTINE_VALIDATION",
        "promotion_allowed": False, "default_pipeline_unchanged": True,
        "warning": "TRAIN64 DBs disjoint from DEV, but cases historically exposed; not pristine.",
    }
    write(output, data)
    print("TRAIN64_ZERO_ROW_PAIRED", json.dumps({
        "baseline_set": totals["base_set"], "new_set": totals["new_set"],
        "baseline_bag": totals["base_bag"], "new_bag": totals["new_bag"],
        "rescues": totals["rescues"], "regressions": totals["regressions"],
        "proposals": totals["proposals"], "calls": packet["calls"],
        "db_improved": improved, "verdict": data["research_gate"],
    }), flush=True)

def main():
    parser = argparse.ArgumentParser()
    cmds = parser.add_subparsers(dest="cmd", required=True)
    a = cmds.add_parser("prepare")
    for field in ("source", "baseline", "unlabeled", "labels"):
        a.add_argument("--" + field, required=True, type=Path)
    b = cmds.add_parser("infer")
    for field in ("unlabeled", "database-root", "output"):
        b.add_argument("--" + field, required=True, type=Path)
    c = cmds.add_parser("score")
    for field in ("frozen", "unlabeled", "labels", "database-root", "output"):
        c.add_argument("--" + field, required=True, type=Path)
    args = parser.parse_args()
    if args.cmd == "prepare":
        prepare(args.source, args.baseline, args.unlabeled, args.labels)
    elif args.cmd == "infer":
        freeze(args.unlabeled, args.database_root, args.output)
    else:
        paired_score(args.frozen, args.unlabeled, args.labels, args.database_root, args.output)

if __name__ == "__main__":
    main()
