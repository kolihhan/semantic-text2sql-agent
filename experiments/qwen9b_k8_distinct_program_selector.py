"""Run full frozen 500-case original Qwen9B K8 deduplicated program vote.

The gold-blind choice is SEALED to disk before the publisher source/gold SQL
is read in a separate step. Bounded SQLite scoring on original 11 physical
BIRD DEV databases, never new model inference or a model swap. This cannot
fix any of the candidate-absent original K8 tasks.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from evaluation.k8_program_diversity_vote import (
    PROTOCOL,FIXED_MODEL_DIGEST,DEV_SOURCE_SHA,SHARDS,
    MIN_PROGRAM_COLLAPSE_AMBIGUITY,MIN_AMBIGUITY_DB_SPREAD,
    MIN_PROMOTION_EX_NET,MIN_PROMOTION_DB_CI_LOWER,
    read_frozen_manifest,read_all_caches,choose_index,score_original_candidates,
    _bounded_fingerprint,_safe_db,dev_rows,bootstrap,aggregate_census
)

WORKERS=8


def fresh(path:Path,obj:dict)->None:
    if path.exists():
        raise FileExistsError("do not overwrite original model evidence")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n",encoding="utf-8")


def metadata(directory:Path)->dict[str,dict]:
    """Only read old execution-success / distinct-result diagnostics, NOT EX."""
    out={}
    for i in range(SHARDS):
        p=directory/f"k8-dev500-offline-scored-{i}.json"
        if not p.exists():
            continue
        x=json.loads(p.read_text("utf-8"))
        if (x.get("model_digest")!=FIXED_MODEL_DIGEST
            or x.get("shard")!=i or x.get("shards")!=SHARDS
            or x.get("source_sha256")!=DEV_SOURCE_SHA):
            raise ValueError("old K8 non-label metadata source invalid")
        for r in x["cases"]:
            cid=str(r["case_id"])
            if cid in out:
                raise ValueError("duplicate old original model source")
            out[cid]={
                "unique_valid_result_sets":int(r["unique_valid_result_sets"]),
                "candidate_execution_success":int(r["candidate_execution_success"])
            }
    return out


def census(directory:Path,manifest:Path,destination:Path)->dict:
    rows=read_all_caches(directory,read_frozen_manifest(manifest))
    r=aggregate_census(rows,metadata(directory))
    if r["n"]!=500 or r["score_metadata_available"] not in (484,500):
        raise ValueError("missing old frozen K8 source cohort")
    r.update({
        "status":"COMPLETE_FROZEN_K8_PROGRAM_MODE_COLLAPSE_SOURCE_ONLY",
        "protocol":PROTOCOL,"model_digest":FIXED_MODEL_DIGEST,
        "source_sha256":DEV_SOURCE_SHA,
        "min_ambiguous_duplicate_cases":MIN_PROGRAM_COLLAPSE_AMBIGUITY,
        "min_ambiguous_databases":MIN_AMBIGUITY_DB_SPREAD,
        "original_model_calls_reused":4000,
        "new_model_calls":0,"publisher_SQL_labels_read":False,
        "HOLDOUT300_used":False,
        "not_a_semantic_EX_improvement":True
    })
    fresh(destination,r)
    print("ORIGINAL_QWEN_K8_CENSUS",json.dumps(r,sort_keys=True),flush=True)
    return r


def choose(cache_dir:Path,manifest:Path,dbroot:Path,worker:int,destination:Path)->dict:
    if worker not in range(WORKERS):
        raise ValueError("worker out of range")
    data=read_all_caches(cache_dir,read_frozen_manifest(manifest))
    rows=[r for r in data if r["original_shard"]%WORKERS==worker]
    if not 60<=len(rows)<=64:
        raise ValueError("frozen Qwen worker missing source cases")
    picked=[]
    for i,row in enumerate(rows):
        sql=[s["sql"] for s in row["samples"]]
        outcomes=score_original_candidates(_safe_db(dbroot,row["db_id"]),sql)
        a=choose_index(sql,outcomes,dedup_program=False)
        b=choose_index(sql,outcomes,dedup_program=True)
        picked.append({
            "case_id":row["case_id"],"db_id":row["db_id"],
            "question_sha256":row["question_sha256"],
            "prompt_sha256":row["prompt_sha256"],
            "original_shard":row["original_shard"],
            "baseline_index":a["index"],"dedup_index":b["index"],
            "baseline_valid":a["valid"],"dedup_valid":b["valid"],
            "outcomes":[{
                "valid":x["valid"],"reason":x["reason"],
                "signature":x["signature"],"set_hash":x["set_hash"],
                "bag_hash":x["bag_hash"]} for x in outcomes]
        })
        print("GOLD_BLIND_K8_PROGRAM_VOTE",worker,i+1,"/",len(rows),
              "changed",a["index"]!=b["index"],flush=True)
    result={
        "status":"ORIGINAL_K8_GOLD_BLIND_CHOICES_SEALED",
        "protocol":PROTOCOL,"model_digest":FIXED_MODEL_DIGEST,
        "source_sha256":DEV_SOURCE_SHA,
        "worker":worker,"n":len(picked),"model_calls":0,
        "publisher_SQL_reference_read":False,"HOLDOUT300_used":False,
        "case_id_sha256":sha256(
            "\n".join(sorted(x["case_id"] for x in picked)).encode()).hexdigest(),
        "cases":picked
    }
    fresh(destination,result)
    print("ORIGINAL_K8_CHOICES_SEALED_BEFORE_GOLD",worker,len(picked),flush=True)
    return result


def score(choices:Path,source:Path,dbroot:Path,destination:Path)->dict:
    selected=json.loads(choices.read_text("utf-8"))
    if (selected.get("status")!="ORIGINAL_K8_GOLD_BLIND_CHOICES_SEALED"
        or selected.get("protocol")!=PROTOCOL
        or selected.get("model_digest")!=FIXED_MODEL_DIGEST
        or selected.get("source_sha256")!=DEV_SOURCE_SHA
        or selected.get("publisher_SQL_reference_read") is not False
        or selected.get("HOLDOUT300_used") is not False
        or selected.get("model_calls")!=0):
        raise ValueError("original source-only selector not sealed")
    source_by_id=dev_rows(source)
    records=[]
    for r in selected["cases"]:
        cid=str(r["case_id"])
        original=source_by_id.get(cid)
        if (original is None or original["db_id"]!=r["db_id"]
            or sha256(str(original["question"]).encode()).hexdigest()!=r["question_sha256"]
            or len(r["outcomes"])!=8 or not 0<=r["baseline_index"]<8
            or not 0<=r["dedup_index"]<8):
            raise ValueError("original source/model/proposal pairing drift")
        # This is the FIRST place reference SQL is read, AFTER choice sealed.
        gold=_bounded_fingerprint(_safe_db(dbroot,r["db_id"]),str(original["SQL"]))
        base=r["outcomes"][r["baseline_index"]]
        alt=r["outcomes"][r["dedup_index"]]
        valid=bool(gold["valid"])
        records.append({
            "case_id":cid,"db_id":r["db_id"],
            "reference_scorable":valid,
            "reference_reason":None if valid else gold["reason"],
            "baseline_index":r["baseline_index"],
            "dedup_index":r["dedup_index"],
            "selection_changed":r["baseline_index"]!=r["dedup_index"],
            "baseline_set_EX":bool(base["valid"] and base["set_hash"]==gold["set_hash"]) if valid else None,
            "dedup_set_EX":bool(alt["valid"] and alt["set_hash"]==gold["set_hash"]) if valid else None,
            "baseline_bag_EX":bool(base["valid"] and base["bag_hash"]==gold["bag_hash"]) if valid else None,
            "dedup_bag_EX":bool(alt["valid"] and alt["bag_hash"]==gold["bag_hash"]) if valid else None
        })
    report={
        "status":"ORIGINAL_K8_DEV_REFERENCE_SCORED_AFTER_CHOICE_SEAL",
        "protocol":PROTOCOL,"model_digest":FIXED_MODEL_DIGEST,
        "source_sha256":DEV_SOURCE_SHA,
        "worker":selected["worker"],"n":len(records),
        "model_calls":0,"HOLDOUT300_used":False,
        "choice_sealed_before_reference":True,
        "cases":records
    }
    fresh(destination,report)
    print("SEALED_K8_MODE_VOTE_SCORING",selected["worker"],
          "scorable",sum(r["reference_scorable"] for r in records),flush=True)
    return report


def aggregate(inputs:list[Path],manifest:Path,destination:Path)->dict:
    expected=set(read_frozen_manifest(manifest))
    if len(inputs)!=WORKERS:
        raise ValueError("need 8 complete original scoring workers")
    workers=set();cases=[]
    for p in inputs:
        d=json.loads(p.read_text("utf-8"))
        if (d.get("status")!="ORIGINAL_K8_DEV_REFERENCE_SCORED_AFTER_CHOICE_SEAL"
            or d.get("protocol")!=PROTOCOL or d.get("model_digest")!=FIXED_MODEL_DIGEST
            or d.get("source_sha256")!=DEV_SOURCE_SHA or d.get("n") not in (60,61,62,63,64)
            or d.get("model_calls")!=0 or d.get("HOLDOUT300_used") is not False
            or d.get("choice_sealed_before_reference") is not True
            or d["worker"] in workers):
            raise ValueError("incomplete original Qwen gold-blind worker")
        workers.add(d["worker"])
        cases+=d["cases"]
    if workers!=set(range(WORKERS)) or len(cases)!=500 or {r["case_id"] for r in cases}!=expected:
        raise ValueError("must aggregate all exact original DEV500 IDs")
    usable=[r for r in cases if r["reference_scorable"]]
    rescues=[r for r in usable if not r["baseline_set_EX"] and r["dedup_set_EX"]]
    losses=[r for r in usable if r["baseline_set_EX"] and not r["dedup_set_EX"]]
    net=len(rescues)-len(losses)
    dbci=bootstrap(usable,"baseline_set_EX","dedup_set_EX",cluster=True)
    qci=bootstrap(usable,"baseline_set_EX","dedup_set_EX",cluster=False)
    passed=(len(usable)==500 and net>=MIN_PROMOTION_EX_NET
            and dbci["lower95"]>MIN_PROMOTION_DB_CI_LOWER
            and len({r["db_id"] for r in rescues})>=3)
    summary={
        "status":"ORIGINAL_DEV500_K8_PROGRAM_VOTE_COMPLETE_WITH_CENSORS"
            if len(usable)<500 else "ORIGINAL_DEV500_K8_PROGRAM_VOTE_COMPLETE",
        "protocol":PROTOCOL,"model_digest":FIXED_MODEL_DIGEST,
        "source_sha256":DEV_SOURCE_SHA,"n":500,"scorable":len(usable),
        "censored_ids":[r["case_id"] for r in cases if not r["reference_scorable"]],
        "db_clusters":len({r["db_id"] for r in cases}),
        "selected_program_changes":sum(r["selection_changed"] for r in cases),
        "baseline_set_EX_correct":sum(r["baseline_set_EX"] for r in usable),
        "distinct_program_set_EX_correct":sum(r["dedup_set_EX"] for r in usable),
        "baseline_bag_EX_correct":sum(r["baseline_bag_EX"] for r in usable),
        "distinct_program_bag_EX_correct":sum(r["dedup_bag_EX"] for r in usable),
        "set_rescues":len(rescues),"set_regressions":len(losses),
        "set_net":net,
        "rescue_db_clusters":len({r["db_id"] for r in rescues}),
        "question_paired_CI95":qci,"physical_db_cluster_CI95":dbci,
        "acceptance_gate":{
            "minimum_net_correct":MIN_PROMOTION_EX_NET,
            "DB_cluster_95CI_lower_gt":MIN_PROMOTION_DB_CI_LOWER,
            "rescue_DBs_min":3,
            "require_all_500_scorable":True
        },
        "accepted_for_promotion":passed,
        "original_model_calls_reused":4000,
        "new_Qwen_calls":0,"new_training_steps":0,
        "HOLDOUT300_used":False,
        "original_DEV500_already_exposed":True,
        "independent_OOD_generalization":False,
        "boundary":"Exact same 4000 immutable Qwen candidates; bounded read-only SQLite execution (8 seconds, row cap 50000). This is existing development data and does not generate new SQL. Censored reference cases block unconditional official EX claims."
    }
    fresh(destination,summary)
    print("COMPLETE_ORIGINAL_K8_DISTINCT_PROGRAM_VOTE",json.dumps(summary,sort_keys=True),flush=True)
    return summary


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("census","choose","score","aggregate"))
    p.add_argument("--cache-dir",type=Path)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--db-root",type=Path)
    p.add_argument("--source",type=Path)
    p.add_argument("--choices",type=Path)
    p.add_argument("--worker",type=int,default=0)
    p.add_argument("--inputs",type=Path,nargs="+")
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.mode=="census":
        if a.cache_dir is None:p.error("census requires cache directory")
        census(a.cache_dir,a.manifest,a.out)
    elif a.mode=="choose":
        if a.cache_dir is None or a.db_root is None:p.error("choose needs K8 caches and original DBs")
        choose(a.cache_dir,a.manifest,a.db_root,a.worker,a.out)
    elif a.mode=="score":
        if a.choices is None or a.db_root is None or a.source is None:
            p.error("score requires sealed choices, physical database, original source")
        score(a.choices,a.source,a.db_root,a.out)
    else:
        if not a.inputs:p.error("aggregate requires all eight original workers")
        aggregate(a.inputs,a.manifest,a.out)


if __name__=="__main__":
    main()
