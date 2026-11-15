"""Read-only BIRD100 replay: original frozen Qwen9B vs one physical-FK JOIN fix.

Uses the original Qwen October candidate SQL and original BIRD databases;
no new model calls, no SQL training, no reference data in repair selector.
The reference SQL is only read after both original and proposed SQL freeze.
Original shard #3 invalid status is retained in the final report.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from evaluation.qwen9b_frozen_replay import (
    SOURCE_CRLF_SHA256, MODEL_DIGEST, CAP, audit_shard
)
from evaluation.bird_loader import load_bird_json
from evaluation.run_bird import _resolve_database
from evaluation.unique_fk_join_repair import repair_one_unique_fk_join
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.execution import SQLExecutionError, execute_readonly
from semantic_sql.verifier import verify_sql_preflight

PROTOCOL="qwen9b-bird100-unique-fk-join-one-correct-endpoint-v1"


def execute(db:Path,sql:str|None):
    if not sql:
        return None
    try:
        response=execute_readonly(db,sql,max_rows=CAP)
        return None if response.truncated else response
    except (SQLExecutionError,ValueError):
        return None


def strict_match(pred,reference,*,bag:bool=False)->bool:
    if pred is None or reference is None:
        return False
    a=map(tuple,pred.rows)
    b=map(tuple,reference.rows)
    return Counter(a)==Counter(b) if bag else set(a)==set(b)


def score_case(original:dict, database:Path, reference_sql:str)->dict:
    """The modifier receives ONLY physical database and original model SQL."""
    if str(original["guarded"]["status"])!="ok":
        return {
            "case_id":str(original["case_id"]),
            "database_id":str(original["database_id"]),
            "original_status":original["guarded"]["status"],
            "triggered":False,"reason":"original_not_executable",
            "original_set":False,"treatment_set":False,
            "original_bag":False,"treatment_bag":False,
            "original_sql":None,"treatment_sql":None,
            "scorable":bool(execute(database,reference_sql))
        }
    base_sql=str(original["guarded"]["final_sql"] or "")
    cat=DatabaseCatalog.from_sqlite(database)
    candidate=repair_one_unique_fk_join(base_sql,cat)
    # The candidate has NO gold; only now execute original/treatment/reference.
    before=execute(database,base_sql)
    modified=candidate.sql
    verified=bool(candidate.changed and verify_sql_preflight(database,modified).ok)
    after=execute(database,modified) if verified else before
    reference=execute(database,reference_sql)
    scorable=reference is not None
    return {
        "case_id":str(original["case_id"]),
        "database_id":str(original["database_id"]),
        "original_status":original["guarded"]["status"],
        "triggered":candidate.changed,"verified":verified,
        "reason":candidate.reason,
        "scorable":scorable,
        "original_set":strict_match(before,reference),
        "treatment_set":strict_match(after,reference),
        "original_bag":strict_match(before,reference,bag=True),
        "treatment_bag":strict_match(after,reference,bag=True),
        "original_sql":base_sql if candidate.changed else None,
        "treatment_sql":modified if candidate.changed else None,
        "fk_old":candidate.original_join,
        "fk_new":candidate.physical_fk,
    }


def shard_run(source:Path,artifact:Path,root:Path,reference:Path,shard:int)->dict:
    # Source hashes, original BIRD ids, old model digest and per-shard
    # original labels are checked by existing external auditor first.
    prior=audit_shard(source,artifact,root,reference,shard)
    raw=json.loads(artifact.read_text())
    cases,labels=load_bird_json(source)
    by_id={c.case_id:c for c in cases}
    ref={x.case_id:x.sql for x in labels}
    records=[]
    for original,audited in zip(raw["cases"],prior["cases"],strict=True):
        cid=str(original["case_id"])
        if cid!=audited["case_id"] or cid not in by_id:
            raise ValueError("original Qwen BIRD source identity drift")
        db=_resolve_database(root,by_id[cid].database_id)
        output=score_case(original,db,ref[cid])
        if output["scorable"] and output["original_set"] != audited["guarded"]["replay_ex"]:
            raise ValueError("historical original strict BIRD EX label drift")
        output["original_label_agreed"]=audited["original_ex_verified"]
        records.append(output)
        print("QWEN9B_FK_JOIN_REPLAY",shard,cid,"trigger",output["triggered"],
              "before",output["original_set"],"after",output["treatment_set"],flush=True)
    if len(records)!=25:
        raise ValueError("original physical 25-row shard incomplete")
    return {
        "protocol":PROTOCOL,"source_sha256":SOURCE_CRLF_SHA256,
        "model_digest":MODEL_DIGEST,"original_status":prior["original_status"],
        "shard":shard,"cases":records,
        "new_model_calls":0,"sql_training":False,
    }


def aggregate(shards:list[dict])->dict:
    if (len(shards)!=4
        or sorted(s["shard"] for s in shards)!=list(range(4))
        or any(s["protocol"]!=PROTOCOL or s["model_digest"]!=MODEL_DIGEST
               or s["source_sha256"]!=SOURCE_CRLF_SHA256
               or len(s["cases"])!=25 or s["new_model_calls"]!=0
               for s in shards)):
        raise ValueError("incomplete or source-drifting physical BIRD replay")
    cases=[c for s in sorted(shards,key=lambda s:s["shard"]) for c in s["cases"]]
    if len({c["case_id"] for c in cases})!=100:
        raise ValueError("not 100 original distinct BIRD identities")
    rescues=[c for c in cases if not c["original_set"] and c["treatment_set"]]
    regressions=[c for c in cases if c["original_set"] and not c["treatment_set"]]
    triggered=[c for c in cases if c["triggered"]]
    stats={
        "original_set_ex":sum(c["original_set"] for c in cases),
        "treatment_set_ex":sum(c["treatment_set"] for c in cases),
        "original_bag_ex":sum(c["original_bag"] for c in cases),
        "treatment_bag_ex":sum(c["treatment_bag"] for c in cases),
        "source_candidate_triggers":len(triggered),
        "source_verified_modifications":sum(c["verified"] for c in triggered),
        "scorable":sum(c["scorable"] for c in cases),
    }
    screen=(
        stats["original_set_ex"]==39
        and len(rescues)-len(regressions)>=3
        and not regressions
        and len({c["database_id"] for c in rescues})>=2
        and stats["treatment_bag_ex"]>=stats["original_bag_ex"]
        and all(c["original_label_agreed"] for c in cases)
    )
    return {
        "protocol":PROTOCOL,"requested":100,"stats":stats,
        "rescue_ids":[c["case_id"] for c in rescues],
        "regression_ids":[c["case_id"] for c in regressions],
        "trigger_ids":[c["case_id"] for c in triggered],
        "trigger_dbs":dict(Counter(c["database_id"] for c in triggered)),
        "original_invalid_shards":[s["shard"] for s in shards
                                   if s["original_status"]!="complete"],
        "source_label_agreement":sum(c["original_label_agreed"] for c in cases),
        "positive_screen_only":screen,
        "promotion_allowed":False,
        "model_calls":0,"training":False,
        "original_Qwen_model_unchanged":True,
        "not_independent_model_generation":True,
        "cases":cases,
    }


def main():
    p=argparse.ArgumentParser()
    for key in ("source","artifact","reference","db-root","output"):
        p.add_argument("--"+key,type=Path)
    p.add_argument("--shard",type=int)
    p.add_argument("--inputs",type=Path,nargs="+")
    args=p.parse_args()
    if args.output is None or args.output.exists():
        raise FileExistsError("output must be a new report")
    if args.inputs:
        result=aggregate([json.loads(f.read_text()) for f in args.inputs])
        print("QWEN9B_FK_JOIN_FROZEN_BIRD100",json.dumps({
            k:v for k,v in result.items() if k!="cases"
        },sort_keys=True),flush=True)
    else:
        if any(x is None for x in (args.source,args.artifact,args.db_root,args.shard)):
            p.error("source/artifact/db-root/shard required")
        result=shard_run(args.source,args.artifact,args.db_root,
                         args.reference or Path("runs/bird-paired-dev-v5/paired.json"),
                         args.shard)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")


if __name__=="__main__":
    main()
