"""Fail-closed CPU gate for historical Qwen9B frozen BIRD100 artifacts.

Usage: python research/frozen100_evidence_gate.py PATH_TO_JSON
A consistent case replay is not an upstream scorer replay, independent
model run, or database-independent generalization.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

MODEL_DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
UPSTREAM_REVISION = "483554eae102996f5ec1f4feab4e78ef29c2a394"

def audit(payload: dict) -> dict:
    blockers = []
    if payload.get("model_digest") != MODEL_DIGEST:
        blockers.append("model digest missing or mismatched")
    upstream = payload.get("official_bird_evaluator") or {}
    if upstream.get("revision") != UPSTREAM_REVISION:
        blockers.append("upstream evaluator revision missing or mismatched")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        blockers.append("100 raw paired case records unavailable: aggregate-only artifact")
        return {"status":"BLOCKED", "blockers":blockers, "replay_labels_consistent":False,
                "upstream_execution_verified":False,"new_model_inference":False}
    if len(cases) != 100:
        blockers.append(f"expected 100 cases, got {len(cases)}")
    seen, databases, counts = set(), set(), Counter()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            blockers.append(f"case {index} is not an object")
            continue
        case_id = str(case.get("case_id") or "")
        if not case_id or case_id in seen:
            blockers.append(f"empty/duplicate ID at position {index}")
        seen.add(case_id)
        db = case.get("database_id")
        if not isinstance(db, str) or not db:
            blockers.append(f"missing database_id at {case_id}")
        else:
            databases.add(db)
        if case.get("initial_sql_identical") is not True:
            blockers.append(f"shared initial SQL not proven at {case_id}")
        a, b = case.get("direct"), case.get("guarded")
        if not isinstance(a, dict) or not isinstance(b, dict):
            blockers.append(f"missing arm at {case_id}")
            continue
        left, right = a.get("official_ex"), b.get("official_ex")
        if not isinstance(left, bool) or not isinstance(right, bool):
            blockers.append(f"invalid score labels at {case_id}")
            continue
        counts["direct_correct"] += left
        counts["guarded_correct"] += right
        counts["rescues"] += not left and right
        counts["regressions"] += left and not right
    for key, expected in [("direct_correct",33),("guarded_correct",39),("rescues",6),("regressions",0)]:
        if counts[key] != expected:
            blockers.append(f"{key}={counts[key]}, expected={expected}")
    if len(databases) != 11:
        blockers.append(f"database families={len(databases)}, expected=11")
    return {"status":"BLOCKED" if blockers else "LABEL_REPLAY_CONSISTENT",
            "blockers":blockers,"replay_labels_consistent":not blockers,
            "upstream_execution_verified":False,"new_model_inference":False,
            "cases":len(cases),"databases":len(databases),"transitions":dict(counts)}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact",type=Path)
    args = parser.parse_args()
    result = audit(json.loads(args.artifact.read_text(encoding="utf-8")))
    print(json.dumps(result,indent=2))
    return 0 if result["status"] == "LABEL_REPLAY_CONSISTENT" else 2

if __name__ == "__main__":
    raise SystemExit(main())
