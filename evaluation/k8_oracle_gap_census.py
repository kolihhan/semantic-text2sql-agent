"""Source-locked, retrospective 52-case oracle-gap mechanism census.

The target cohort is derived ONLY from the fixed original physical 500-case
K8 worker outcomes: exactly the tasks where pass8 oracle is True and the
label-blind plurality selection is wrong. This is diagnostic error analysis,
not an evaluation set, training set, selection model, or new score.
NO LLM calls, no new candidates, and no HOLDOUT300 inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from evaluation.bounded_shard7_probe import bounded_outcome
from evaluation.kcache_protocol import (
    MODEL_DIGEST, SOURCE_SHA256, MANIFEST, majority_selector, sample_correctness,
    read_frozen_manifest, source_cases,
)
from evaluation.qwen9b_kcache import validate_cache, prepare_prompt
from evaluation.run_bird import _resolve_database
from evaluation.k8_prefix_frontier import PROTOCOL as PHYSICAL_PROTOCOL, SQL_TIMEOUT

CENSUS_PROTOCOL="BIRD-DEV500-PHYSICAL-K8-ORACLE-GAP-52-SOURCE-ONLY-DIAGNOSTIC-V1"
NUM_PARTS=4
EXPECTED_GAP_COUNT=52


def describe_support(outcomes: list[dict], gold: dict, selected_index: int) -> dict:
    """Count correct candidates AFTER locked prediction; never choose from gold."""
    if len(outcomes)!=8 or selected_index not in range(8):
        raise ValueError("expected exact eight source candidates")
    chosen=majority_selector(outcomes)["index"]
    correct=[i for i,x in enumerate(outcomes) if sample_correctness(x,gold)]
    freq=Counter(x["signature"] for x in outcomes if x.get("valid") and x.get("signature"))
    correct_vote_support=max((freq[outcomes[i]["signature"]] for i in correct),default=0)
    selected_support=freq[outcomes[chosen]["signature"]] if outcomes[chosen].get("valid") else 0
    return {
        "correct_candidate_count":len(correct),
        "correct_result_support":correct_vote_support,
        "selected_result_support":selected_support,
        "valid_candidate_count":sum(bool(x.get("valid")) for x in outcomes),
        "selected_index_matches_physical_record":chosen==selected_index,
        "selected_is_correct":bool(sample_correctness(outcomes[selected_index],gold)),
        "gold_scorable":bool(gold.get("valid")),
        "candidate_timeouts":sum(x.get("reason")=="query_timeout" for x in outcomes),
        "gold_timeout":gold.get("reason")=="query_timeout",
    }


def read_locked_evidence(worker_files: list[Path], cache_files: list[Path], source_rows: list[dict],
                         frozen_ids: list[str]) -> tuple[list[dict],dict[str,dict],dict[str,dict]]:
    if len(worker_files)!=8 or len(cache_files)!=32:
        raise ValueError("need exact 8 original physical workers and all 32 cache files")
    workers=[json.loads(p.read_text("utf-8")) for p in worker_files]
    if sorted(p.get("worker") for p in workers)!=list(range(8)):
        raise ValueError("missing or duplicated physical 8-worker score evidence")
    for w in workers:
        if (w.get("protocol")!=PHYSICAL_PROTOCOL or w.get("source_sha256")!=SOURCE_SHA256
            or w.get("model_digest")!=MODEL_DIGEST or w.get("query_timeout_s")!=SQL_TIMEOUT
            or w.get("model_calls")!=0 or w.get("holdout300_used") is not False):
            raise ValueError("original source/model/budget drift")
    scored=[c for w in workers for c in w["cases"]]
    unique={str(x["case_id"]):x for x in scored}
    if len(scored)!=500 or len(unique)!=500 or set(unique)!=set(frozen_ids):
        raise ValueError("original 500 unique source IDs missing or duplicated")
    targets=[c for c in scored if c["pass8_oracle_set"] and not c["k8_set_ex"]]
    if len(targets)!=EXPECTED_GAP_COUNT:
        raise ValueError("oracle-gap cohort drift: cannot select post-hoc different target set")
    caches={}
    for p in cache_files:
        sh=json.loads(p.read_text("utf-8"))
        validate_cache(sh,source_rows)
        if sh["shards"]!=32 or sh["k"]!=8:
            raise ValueError("original cache-shard contract drift")
        for c in sh["cases"]:
            if c["case_id"] in caches:
                raise ValueError("duplicate candidate ID")
            caches[c["case_id"]]=c
    if len(caches)!=500 or set(caches)!=set(frozen_ids):
        raise ValueError("all source candidate caches not present")
    return sorted(targets,key=lambda x:int(x["case_id"])),unique,caches


def run_part(source:Path, db_root:Path, manifest_path:Path, worker_dir:Path,
             cache_dir:Path, part:int) -> dict:
    if part not in range(NUM_PARTS):
        raise ValueError("only four frozen diagnostic parts")
    m=read_frozen_manifest(manifest_path)
    source_rows=source_cases(source,m,split="dev")
    source_map={str(x["question_id"]):x for x in source_rows}
    targets,physical,caches=read_locked_evidence(
        sorted(worker_dir.glob("prefix-frontier-worker-*.json")),
        sorted(cache_dir.glob("k8-dev500-candidate-cache-*.json")),
        source_rows,m["dev_case_ids"])
    selected=[x for i,x in enumerate(targets) if i%NUM_PARTS==part]
    rows=[]
    catalog={}
    for target in selected:
        cid=target["case_id"]
        c=caches[cid]; original=source_map[cid]
        db=_resolve_database(db_root,c["db_id"])
        _,expected_prompt_sha=prepare_prompt(original,db_root,catalog)
        if (c["db_id"]!=str(original["db_id"]) or
            c["question_sha256"]!=sha256(str(original["question"]).encode()).hexdigest() or
            c["prompt_sha256"]!=expected_prompt_sha):
            raise ValueError("question-to-source/prompt alignment drift")
        # Exact original eight source SQLs; original model is never invoked.
        outcomes=[bounded_outcome(db,s["sql"],SQL_TIMEOUT) for s in c["samples"]]
        # The chosen index was frozen in the original physical worker before
        # any reference/gold execution. Gold is used only for diagnosis.
        gold=bounded_outcome(db,str(original["SQL"]),SQL_TIMEOUT)
        row=describe_support(outcomes,gold,int(target["k_indices"]["8"]))
        row.update(case_id=cid,db_id=c["db_id"])
        rows.append(row)
        print("ORACLE_GAP_REPLAY",part,len(rows),len(selected),
              "stable",int(row["selected_index_matches_physical_record"]),
              "correct_sample_count",row["correct_candidate_count"],
              "gold_ok",int(row["gold_scorable"]),flush=True)
    return {
        "status":"SOURCE_ONLY_ORACLE_GAP_CENSUS_PART_COMPLETE",
        "protocol":CENSUS_PROTOCOL,"physical_protocol":PHYSICAL_PROTOCOL,
        "source_sha256":SOURCE_SHA256,"model_digest":MODEL_DIGEST,
        "original_fixed_DEV500":500,"original_oracle_gap_cohort":EXPECTED_GAP_COUNT,
        "part":part,"parts":NUM_PARTS,"n":len(rows),
        "query_timeout_s":SQL_TIMEOUT,"model_calls":0,"holdout300_used":False,
        "not_an_independent_accuracy_evaluation":True,"cases":rows,
    }


def aggregate(parts:list[dict]) -> dict:
    if len(parts)!=NUM_PARTS or sorted(x.get("part") for x in parts)!=list(range(NUM_PARTS)):
        raise ValueError("complete four-part original source diagnostic required")
    for p in parts:
        if (p.get("status")!="SOURCE_ONLY_ORACLE_GAP_CENSUS_PART_COMPLETE" or
            p.get("protocol")!=CENSUS_PROTOCOL or
            p.get("physical_protocol")!=PHYSICAL_PROTOCOL or
            p.get("source_sha256")!=SOURCE_SHA256 or p.get("model_digest")!=MODEL_DIGEST or
            p.get("parts")!=NUM_PARTS or p.get("query_timeout_s")!=SQL_TIMEOUT or
            p.get("model_calls")!=0 or p.get("holdout300_used") is not False):
            raise ValueError("original source/diagnostic procedure mismatch")
    records=[x for p in parts for x in p["cases"]]
    ids=[x["case_id"] for x in records]
    if len(records)!=EXPECTED_GAP_COUNT or len(set(ids))!=EXPECTED_GAP_COUNT:
        raise ValueError("not exactly original 52 source-gap cases")
    mismatch=[x for x in records if not x["selected_index_matches_physical_record"]]
    invalid=[x for x in records if not x["gold_scorable"] or
             x["selected_is_correct"] or x["correct_candidate_count"]<1]
    # Do NOT silently hide unstable timeout replays or create fabricated
    # candidate-label support for cases that cannot reproduce.
    stable=[x for x in records if x not in mismatch and x not in invalid]
    count_correct=Counter(str(x["correct_candidate_count"]) for x in stable)
    count_support=Counter(str(x["correct_result_support"]) for x in stable)
    by_db=Counter(x["db_id"] for x in stable)
    return {
        "status":"COMPLETE_REPRODUCED_52_CASE_ORACLE_GAP_CENSUS"
                 if not mismatch and not invalid else "DIAGNOSTIC_REPLAY_INSTABILITY_DO_NOT_INFER",
        "protocol":CENSUS_PROTOCOL,
        "original_fixed_DEV500":500,"intended_oracle_gap_cases":EXPECTED_GAP_COUNT,
        "replayed_cases":len(records),"stable_oracle_gap_cases":len(stable),
        "selected_candidate_index_drift":len(mismatch),
        "invalid_gold_or_census_cases":len(invalid),
        "correct_candidate_count_histogram":dict(sorted(count_correct.items())),
        "correct_result_vote_support_histogram":dict(sorted(count_support.items())),
        "stable_gap_case_counts_by_database":dict(sorted(by_db.items())),
        "wrong_selected_result_support_histogram":dict(sorted(Counter(
            str(x["selected_result_support"]) for x in stable).items())),
        "source_sha256":SOURCE_SHA256,"model_digest":MODEL_DIGEST,
        "new_original_Qwen_inference_calls":0,
        "holdout300_unused":True,
        "disclaimer":"Retrospectively conditioned on original oracle-positive/selector-wrong DEV outcomes. This is NOT a prospective accuracy result; labels are used ONLY to diagnose failure mechanics, not to tune or deploy a selector. Timing-dependent SQL replay drift invalidates any affected per-case conclusion. No hidden holdout or source-independent transfer.",
    }


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("score","aggregate"))
    p.add_argument("--source",type=Path)
    p.add_argument("--db-root",type=Path)
    p.add_argument("--manifest",type=Path,default=Path(MANIFEST))
    p.add_argument("--worker-dir",type=Path)
    p.add_argument("--cache-dir",type=Path)
    p.add_argument("--part",type=int)
    p.add_argument("--inputs",type=Path,nargs="+")
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():
        raise FileExistsError("never overwrite fixed-source diagnostic evidence")
    if a.mode=="score":
        if (not a.source or not a.db_root or not a.cache_dir or
            not a.worker_dir or a.part is None):
            p.error("score requires pinned source, 8 worker outcomes, 32 caches and part")
        result=run_part(a.source,a.db_root,a.manifest,a.worker_dir,a.cache_dir,a.part)
    else:
        if not a.inputs or len(a.inputs)!=NUM_PARTS:
            p.error("aggregate requires four source-verified parts")
        result=aggregate([json.loads(path.read_text("utf-8")) for path in a.inputs])
        print("DEV500_K8_ORACLE_GAP_CENSUS",json.dumps(result,sort_keys=True),flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
