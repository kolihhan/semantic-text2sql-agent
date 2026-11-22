"""Recompute historical frozen Qwen9B DEV100 reported scores without inference.
This verifies the archived score ledger, NOT physical database EX replay.
"""
from collections import Counter
import json
from pathlib import Path

RUN_ID = 37307108955
MODEL_DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
SOURCE_SHA256 = "0310b7ca2dfb6247234e2073f93e7a75106ee41ba55231f75d06d2ef68efe2ec"

def audit(root: Path):
    ids = set()
    statuses = []
    shards = []
    rows = []
    for i in range(4):
        p = root/f"sql-shard-{i}"/f"qwen9b-shard-{i}.json"
        j = json.loads(p.read_text("utf-8"))
        if j["model"] != "qwen3.5:9b-q4_K_M" or j["model_digest"] != MODEL_DIGEST:
            raise ValueError("model drift or corrupted historical artifact")
        if j["source_sha256"] != SOURCE_SHA256 or j["sample_size"] != 25 or len(j["cases"]) != 25:
            raise ValueError("publisher source or sample changed")
        if j["status"] not in {"complete","invalid"}:
            raise ValueError("unknown or unfinished shard")
        statuses.append(j["status"])
        for c in j["cases"]:
            case_id = str(c["case_id"])
            if case_id in ids or c["initial_sql_identical"] is not True:
                raise ValueError("duplicate ID or original Base/Guarded initial SQL changed")
            ids.add(case_id)
        rows.extend(j["cases"])
        shards.append({
            "shard":i,
            "status":j["status"],
            "direct_correct":sum(x["direct"]["official_ex"] is True for x in j["cases"]),
            "guarded_correct":sum(x["guarded"]["official_ex"] is True for x in j["cases"]),
        })
    if len(rows) != 100 or len(ids) != 100:
        raise ValueError("not complete unique 100 original Qwen cases")
    counts = {
        "direct":sum(x["direct"]["official_ex"] is True for x in rows),
        "guarded":sum(x["guarded"]["official_ex"] is True for x in rows),
        "rescues":sum(x["direct"]["official_ex"] is False and x["guarded"]["official_ex"] is True for x in rows),
        "regressions":sum(x["direct"]["official_ex"] is True and x["guarded"]["official_ex"] is False for x in rows),
    }
    if counts != {"direct":33,"guarded":39,"rescues":6,"regressions":0}:
        raise ValueError("original 33->39 ledger cannot be independently recomputed from archived flags")
    if statuses != ["complete","complete","complete","invalid"]:
        raise ValueError("unexpected canonical workflow shard validity")
    capped = sorted(c["case_id"] for c in rows if any(
        "rows=50000" in str(stage.get("summary",""))
        for stage in c["guarded"].get("stages",[])))
    if capped != ["305"]:
        raise ValueError("frozen capped result evidence unexpectedly changed")
    c305=next(c for c in rows if c["case_id"] == "305")
    if c305["guarded"]["official_ex"] is not False:
        raise ValueError("known capped wrong-result score changed")
    return {
        "status":"HISTORICAL_SCORE_RECOMPUTED_CANONICAL_RUN_INVALID",
        "run_id":RUN_ID,"source_sha256":SOURCE_SHA256,
        "model_digest":MODEL_DIGEST,
        "shard_scores":shards,"counts":counts,
        "original_workflow_aggregate":"SKIPPED",
        "capped_observed_case_ids":capped,
        "no_qwen_inference":True,
        "physical_sql_reexecution":False,
        "promotion_allowed":False,
        "warning":"These are the four original stored official_ex flags, including an invalid shard. They do not prove a clean rerun or fresh DEV100 gain.",
    }

def main():
    a = audit(Path("evidence"))
    Path("audit").mkdir(exist_ok=True)
    Path("audit/qwen9b-dev100-original-score-integrity.json").write_text(json.dumps(a,sort_keys=True,indent=2)+"\n")
    print("QWEN9B_HISTORICAL_SCORE_AUDITED",json.dumps({
        "counts":a["counts"],"shard_statuses":[x["status"] for x in a["shard_scores"]],
        "capped":a["capped_observed_case_ids"],
        "promote":a["promotion_allowed"],
    }), flush=True)

if __name__ == "__main__":
    main()
