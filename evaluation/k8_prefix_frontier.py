"""One-cache source-locked K1/K2/K4/K8 prefix frontier, NO Qwen calls.

All 500 immutable original BIRD DEV candidates: 8 seeds/case. Use the SAME
original frozen SQL text, original physical SQLite DBs, original source and
gold-blind physical-result plurality. All K arms share candidate prefix and
a uniform 8-second per SQL worker budget. This is a NEW explicitly bounded
scoring protocol; never silently mix with original unbounded DEV500 results.
HOLDOUT300 is never accessed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from evaluation.bounded_shard7_probe import bounded_outcome
from evaluation.kcache_protocol import (
    MODEL_DIGEST, SOURCE_SHA256, MANIFEST, read_frozen_manifest, source_cases,
    majority_selector, sample_correctness, paired_bootstrap
)
from evaluation.qwen9b_kcache import validate_cache, prepare_prompt, source_shard
from evaluation.run_bird import _resolve_database


K_VALUES = (1, 2, 4, 8)
NUM_WORKERS = 8
PROTOCOL = "BIRD-DEV500-SAME-ORIGINAL-QWEN9B-K8-CACHE-UNIFORM-8S-SQLITE-PREFIX-FRONTIER-V1"
SQL_TIMEOUT = 8.


def case_from_outcomes(cache: dict, outcomes: list[dict], gold: dict) -> dict:
    """Only outcomes before reference are used to select every K index."""
    if cache["k"] != 8 or len(outcomes) != 8:
        raise ValueError("wrong K8 cache")
    choices = {k: majority_selector(outcomes[:k])["index"] for k in K_VALUES}
    # No reference/gold inside the voting rule. Evaluate only after freezing.
    scores = {f"k{k}_set_ex": bool(sample_correctness(outcomes[i], gold))
              for k,i in choices.items()}
    scores.update({
        "case_id": cache["case_id"], "db_id": cache["db_id"],
        "gold_scorable": bool(gold["valid"]),
        "gold_timeout": gold.get("reason") == "query_timeout",
        "candidate_timeouts": sum(x.get("reason") == "query_timeout" for x in outcomes),
        "candidate_executed": sum(x["valid"] for x in outcomes),
        "pass8_oracle_set": any(sample_correctness(x,gold) for x in outcomes),
        "goldblind_selection": True,
        "k_indices": {str(k): i for k,i in choices.items()},
    })
    return scores


def score_worker(paths: list[Path], source: Path, db_root: Path,
                 manifest_file: Path, worker: int) -> dict:
    if worker not in range(NUM_WORKERS):
        raise ValueError("invalid worker")
    manifest = read_frozen_manifest(manifest_file)
    source_rows = source_cases(source, manifest, split="dev")
    index = {str(r["question_id"]):r for r in source_rows}
    selected_paths = []
    for p in paths:
        x = json.loads(p.read_text("utf-8"))
        if x["shard"] % NUM_WORKERS == worker:
            selected_paths.append((x["shard"],p,x))
    if len(selected_paths) != 4 or sorted(n for n,_,_ in selected_paths) != [
            worker,worker+8,worker+16,worker+24]:
        raise ValueError("must cover exactly the four original shards for worker")
    catalog={}
    results=[]
    for shard,path,cache in sorted(selected_paths):
        validate_cache(cache,source_rows)
        if cache["shards"]!=32 or cache["k"]!=8 or cache["model_digest"]!=MODEL_DIGEST:
            raise ValueError("original source/model/candidate budget drift")
        expected=source_shard(source_rows,shard,32)
        if [c["case_id"] for c in cache["cases"]] != [str(r["question_id"]) for r in expected]:
            raise ValueError("manifest source membership differs")
        for case in cache["cases"]:
            original=index[case["case_id"]]
            db=_resolve_database(db_root,case["db_id"])
            _,prompt_sha=prepare_prompt(original,db_root,catalog)
            if (case["db_id"]!=str(original["db_id"]) or
                case["question_sha256"]!=sha256(str(original["question"]).encode()).hexdigest() or
                case["prompt_sha256"]!=prompt_sha):
                raise ValueError("candidate original question/prompt modified")
            # The 8 physical results are cached ONCE in memory, then sliced to
            # emulate deployment with K1,K2,K4,K8 original generated prefixes.
            outcomes=[bounded_outcome(db,sample["sql"],SQL_TIMEOUT)
                      for sample in case["samples"]]
            # Prediction choices depend only on SQL and physical SQLite data,
            # NEVER publisher reference SQL or its result.
            # Gold is executed only after all choices are fixed.
            selected={k:majority_selector(outcomes[:k])["index"] for k in K_VALUES}
            gold=bounded_outcome(db,str(original["SQL"]),SQL_TIMEOUT)
            record=case_from_outcomes(case,outcomes,gold)
            assert record["k_indices"]=={str(k):i for k,i in selected.items()}
            results.append(record)
            print("FROZEN_K_PREFIX_CASE",worker,shard,len(results),
                  "query_timeouts",record["candidate_timeouts"],
                  "reference_timeout",record["gold_timeout"],flush=True)
    if len(results) != (63 if worker < 4 else 62):
        raise ValueError("worker task count differs from original frozen manifest")
    return {
        "status":"SOURCE_LOCKED_K_PREFIX_WORKER_COMPLETE",
        "protocol":PROTOCOL,"worker":worker,"workers":NUM_WORKERS,
        "source_sha256":SOURCE_SHA256,"model_digest":MODEL_DIGEST,
        "query_timeout_s":SQL_TIMEOUT,"model_calls":0,"no_training":True,
        "holdout300_used":False,"cases":results,
    }


def aggregate(parts: list[dict], manifest: dict) -> dict:
    if len(parts)!=NUM_WORKERS or sorted(x.get("worker") for x in parts)!=list(range(NUM_WORKERS)):
        raise ValueError("must have all eight unique source worker parts")
    for p in parts:
        if (p["status"]!="SOURCE_LOCKED_K_PREFIX_WORKER_COMPLETE" or
            p["protocol"]!=PROTOCOL or p["workers"]!=NUM_WORKERS or
            p["source_sha256"]!=SOURCE_SHA256 or p["model_digest"]!=MODEL_DIGEST or
            p["query_timeout_s"]!=SQL_TIMEOUT or p["model_calls"]!=0 or
            p["holdout300_used"] is not False):
            raise ValueError("source/model/timeout/horizon drift")
    rows=[x for p in parts for x in p["cases"]]
    ids=[x["case_id"] for x in rows]
    if len(rows)!=500 or len(set(ids))!=500 or set(ids)!=set(manifest["dev_case_ids"]):
        raise ValueError("missing or repeated source DEV500 questions")
    if any(x["goldblind_selection"] is not True for x in rows):
        raise ValueError("oracle leaked into selector")
    nonscorable=[x["case_id"] for x in rows if not x["gold_scorable"]]
    result={
        "status": "COMPLETE_K_PREFIX_BOUNDED_DEV500" if not nonscorable else "BLOCKED_REFERENCE_SQL_TIMEOUT",
        "protocol":PROTOCOL,"n":500,"n_physical_databases":len(set(x["db_id"] for x in rows)),
        "k_candidates_original":8,"original_model_inference_calls_reused":4000,
        "new_model_inference_calls":0,"source_sha256":SOURCE_SHA256,
        "model_digest":MODEL_DIGEST,"query_timeout_s":SQL_TIMEOUT,
        "candidate_timeouts":sum(x["candidate_timeouts"] for x in rows),
        "reference_timeouts":sum(x["gold_timeout"] for x in rows),
        "unscorable_count":len(nonscorable),
        "holdout300_unused":True,"no_new_QLoRA":True,
        "scores":{},"disclaimer":(
            "Uniform 8s bounded original DEV500 physical SQLite metric, "
            "exploratory source-local NOT the original unbounded main EX; "
            "source DBs historically visited, case-level and DB-cluster CI descriptive. "
            "K8 generates 8x as many SQLs per question as K1. "
            "No untouched final holdout or external generalization claimed."
        )
    }
    if nonscorable:
        return result
    def rate(key):return sum(int(x[key]) for x in rows)/len(rows)
    for k in K_VALUES:
        key=f"k{k}_set_ex"
        base="k1_set_ex"
        rescues=sum(not x[base] and x[key] for x in rows)
        regressions=sum(x[base] and not x[key] for x in rows)
        result["scores"][str(k)]={
            "correct":sum(x[key] for x in rows),
            "n":500,"set_ex":rate(key),
            "original_Qwen_generations_per_task":k,
            "rescues_vs_first":rescues,"regressions_vs_first":regressions,
            "net_gain_vs_first":rescues-regressions,
            "paired_question_95ci":paired_bootstrap(rows,base,key),
            "paired_DB_cluster_95ci":paired_bootstrap(rows,base,key,cluster=True),
        }
    result["pass8_oracle_upper_bound"]=sum(x["pass8_oracle_set"] for x in rows)
    return result


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("score","aggregate"))
    p.add_argument("--source",type=Path)
    p.add_argument("--db-root",type=Path)
    p.add_argument("--cache-dir",type=Path)
    p.add_argument("--worker",type=int)
    p.add_argument("--inputs",type=Path,nargs="+")
    p.add_argument("--manifest",type=Path,default=Path(MANIFEST))
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():
        raise FileExistsError("never overwrite immutable evaluation evidence")
    m=read_frozen_manifest(args.manifest)
    if args.mode=="score":
        if not args.source or not args.db_root or not args.cache_dir or args.worker is None:
            p.error("score requires frozen source, physical DB root, 32 caches, worker")
        caches=sorted(args.cache_dir.glob("k8-dev500-candidate-cache-*.json"))
        if len(caches)!=32:
            raise ValueError("all 32 immutable caches required")
        value=score_worker(caches,args.source,args.db_root,args.manifest,args.worker)
    else:
        if not args.inputs or len(args.inputs)!=NUM_WORKERS:
            p.error("aggregate needs 8 immutable CPU worker scores")
        parts=[json.loads(path.read_text("utf-8")) for path in args.inputs]
        value=aggregate(parts,m)
        print("FROZEN_DEV500_UNIFORM_SQLITE_PREFIX_FRONTIER",
              json.dumps({k:v for k,v in value.items() if k!="cases"},sort_keys=True),
              flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(value,sort_keys=True,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
