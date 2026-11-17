"""Predeclared K8 bag-result plurality vs original set-result plurality.

NO new Qwen calls. All predictions fixed before gold SQL execution.
Same immutable BIRD DEV500 original cache. This is a sensitivity test,
not license to select a post-hoc winner and advertise generalization.
"""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
from evaluation.kcache_protocol import (
    PROTOCOL as CACHE_PROTOCOL, SOURCE_SHA256, MODEL_DIGEST,
    candidate_outcome, majority_selector, paired_bootstrap,
    sample_correctness, read_frozen_manifest, MANIFEST)
from evaluation.qwen9b_kcache import validate_cache
from evaluation.run_bird import _resolve_database

PROTOCOL="BIRD-DEV500-K8-OFFLINE-BAG-SET-V1"
MIN_EFFECT=.05

def bag_majority_selector(outcomes:list[dict])->dict:
    if not outcomes:raise ValueError("no candidates")
    valid=[(i,x["bag_hash"]) for i,x in enumerate(outcomes)
           if x.get("valid") and x.get("bag_hash")]
    if not valid:return {"index":0,"valid":0,"agreement":0.}
    counts=Counter(s for _,s in valid)
    best=max(counts.values())
    idx=next(i for i,s in valid if counts[s]==best)
    return {"index":idx,"valid":len(valid),"agreement":best/len(outcomes)}

def evaluate_one(cached:dict,db:Path,reference_sql:str)->dict:
    if cached.get("k")!=8 or len(cached.get("samples",[]))!=8:
        raise ValueError("not complete original K8")
    responses=[candidate_outcome(db,s["sql"]) for s in cached["samples"]]
    # Both prediction indices computed BEFORE touching publisher reference.
    set_index=majority_selector(responses)["index"]
    bag_index=bag_majority_selector(responses)["index"]
    gold=candidate_outcome(db,reference_sql)
    set_ok=[sample_correctness(x,gold) for x in responses]
    bag_ok=[sample_correctness(x,gold,bag=True) for x in responses]
    return {"case_id":cached["case_id"],"db_id":cached["db_id"],
        "k":8,"gold_scorable":bool(gold["valid"]),
        "original_set_vote_ex":bool(set_ok[set_index]),
        "bag_vote_set_ex":bool(set_ok[bag_index]),
        "original_set_vote_bag_ex":bool(bag_ok[set_index]),
        "bag_vote_bag_ex":bool(bag_ok[bag_index]),
        "same_index":set_index==bag_index,
        "selector_no_gold":True}

def score_cache(cache:dict,source_rows:dict[str,dict],dbroot:Path)->dict:
    if cache.get("status")!="CANDIDATE_CACHE_COMPLETE" or cache.get("k")!=8:
        raise ValueError("not immutable K8 candidate cache")
    rows=[source_rows[x["case_id"]] for x in cache["cases"]]
    validate_cache(cache,rows,require_shard=False)
    records=[]
    for row in cache["cases"]:
        original=source_rows[row["case_id"]]
        if row["db_id"]!=str(original["db_id"]):
            raise ValueError("cached case database drift")
        records.append(evaluate_one(
            row,_resolve_database(dbroot,row["db_id"]),str(original["SQL"])))
    return {"status":"OFFLINE_BAG_SET_ONE_SHARD","protocol":PROTOCOL,
        "source_sha256":SOURCE_SHA256,"model_digest":MODEL_DIGEST,
        "shard":cache["shard"],"shards":cache["shards"],
        "k":8,"model_calls":0,"n":len(records),"cases":records}

def aggregate(shards:list[dict],manifest:dict)->dict:
    if len(shards)!=32 or sorted(x.get("shard") for x in shards)!=list(range(32)):
        raise ValueError("require all 32 original frozen shards")
    if any(x.get("status")!="OFFLINE_BAG_SET_ONE_SHARD" or
           x.get("protocol")!=PROTOCOL or x.get("shards")!=32 or
           x.get("model_digest")!=MODEL_DIGEST or
           x.get("source_sha256")!=SOURCE_SHA256 or
           x.get("k")!=8 or x.get("model_calls")!=0 or
           x.get("n")!=len(x.get("cases",[])) for x in shards):
        raise ValueError("original cached protocol/model changed")
    rows=[r for s in shards for r in s["cases"]]
    ids=[r["case_id"] for r in rows]
    if len(rows)!=500 or len(set(ids))!=500 or set(ids)!=set(manifest["dev_case_ids"]):
        raise ValueError("incomplete frozen DEV500")
    if any(r.get("gold_scorable") is not True or
           r.get("selector_no_gold") is not True for r in rows):
        raise ValueError("incomplete gold or source contamination")
    rate=lambda k:sum(float(x[k]) for x in rows)/500
    qci=paired_bootstrap(rows,"original_set_vote_ex","bag_vote_set_ex")
    dbci=paired_bootstrap(rows,"original_set_vote_ex","bag_vote_set_ex",cluster=True)
    delta=qci["estimate"]
    return {"status":"COMPLETE_DEV500_K8_OFFLINE_BAG_SET_SENSITIVITY",
        "protocol":PROTOCOL,"n":500,"k":8,"model_calls":0,
        "set_plurality_ex":rate("original_set_vote_ex"),
        "bag_plurality_ex":rate("bag_vote_set_ex"),
        "set_plurality_bag_ex":rate("original_set_vote_bag_ex"),
        "bag_plurality_bag_ex":rate("bag_vote_bag_ex"),
        "changed_prediction_count":sum(not x["same_index"] for x in rows),
        "paired_ex_gain":delta,
        "task_paired_ci95":qci,"db_cluster_paired_ci95":dbci,
        "predeclared_gate":delta>=MIN_EFFECT and qci["ci95_lower"]>0 and dbci["ci95_lower"]>0,
        "promotion_allowed":False,"holdout300_unused":True,
        "interpretation":"Same frozen DEV500 candidates; bag plurality is one label-blind sensitivity test. DEV is not independent proof and human semantic score remains undefined."}
