"""Freeze novel BIRD TRAIN test questions on DBs unseen in prior SQL-QLoRA prep.

Predefined intervention: accurate SQLite implicit FK target columns only.
Cohort selection uses physical schema, reference-SQL compile validity,
question identity and deterministic hash; NO inference scores or outputs.
Heldout BIRD DEV questions/answers not used for model calls.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

from evaluation.implicit_fk_grounding import resolve_implicit_fk_targets
from evaluation.schema_context import select_schema_context
from scripts.prepare_birdtrain_qwen9b_sft_source import (
    TRAIN_SHA,DEV_SHA,load_checked,fetch_manifest,fetch_database,
    normalized,validate_gold_sql,
)
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.direct import _SYSTEM

SEED="fk-target-20261117-novel-source32-v1"
N_DBS=4
N_PER_DB=8
N=N_DBS*N_PER_DB
MAX_CHECKED_DBS=31
MAX_DB_BYTES=96*1024*1024


def hashkey(kind:str,value:object)->str:
    return sha256(f"{SEED}:{kind}:{value}".encode()).hexdigest()


def freeze(train_path:Path,dev_path:Path,prior_path:Path,work:Path,out:Path)->dict:
    train=load_checked(train_path,TRAIN_SHA)
    dev=load_checked(dev_path,DEV_SHA)
    prior=json.loads(prior_path.read_text())
    if prior.get("status")!="SOURCE_VERIFIED_NO_MODEL_TRAINING":
        raise ValueError("Original source audit not verified")
    prior_train_sha=prior.get("train_sha256")
    prior_val_sha=prior.get("validation_sha256")
    if len(prior_train_sha)!=64 or len(prior_val_sha)!=64:
        raise ValueError("Previous source population missing SHA")
    # Freeze full prior TRAIN+validation database exclusions from source lists.
    used_train={r["source_db_id"] for r in [
        json.loads(x) for x in (out.parent/"prior_train.jsonl").read_text().splitlines()
    ]}
    used_valid={r["source_db_id"] for r in [
        json.loads(x) for x in (out.parent/"prior_validation.jsonl").read_text().splitlines()
    ]}
    if len(used_train)!=16 or len(used_valid)!=4 or (used_train&used_valid):
        raise ValueError("Prior 20 distinct database exclusions not intact")
    if sha256((out.parent/"prior_train.jsonl").read_bytes()).hexdigest()!=prior_train_sha:
        raise ValueError("Training cohort changed")
    if sha256((out.parent/"prior_validation.jsonl").read_bytes()).hexdigest()!=prior_val_sha:
        raise ValueError("Validation cohort changed")
    excluded=used_train|used_valid|{r["db_id"] for r in dev}
    dev_questions={normalized(r.get("question","")) for r in dev}
    previous_questions={normalized(r["messages"][1]["content"].split("\nSchema context:\n")[0].removeprefix("Question: ")) for r in (
        json.loads(x) for file in ("prior_train.jsonl","prior_validation.jsonl")
        for x in (out.parent/file).read_text().splitlines()
    )}
    by_db=defaultdict(list)
    seen=set(dev_questions)|previous_questions
    manifest=fetch_manifest()
    for idx,r in enumerate(train):
        dbid=r.get("db_id")
        question=normalized(r.get("question",""))
        if (dbid in excluded or dbid not in manifest or not question
            or question in seen or len(str(r.get("question","")))>=450
            or int(manifest[dbid]["size"])>MAX_DB_BYTES):
            continue
        seen.add(question)
        by_db[dbid].append((idx,r))
    eligible=[dbid for dbid,rows in by_db.items() if len(rows)>=N_PER_DB]
    eligible.sort(key=lambda x:hashkey("db",x))
    frozen=[]
    db_attest=[]
    checked=0
    for dbid in eligible:
        if len(db_attest)==N_DBS:break
        if checked>=MAX_CHECKED_DBS:break
        checked+=1
        path=fetch_database(dbid,manifest[dbid],work)
        cat=DatabaseCatalog.from_sqlite(path)
        corrected,report=resolve_implicit_fk_targets(cat)
        if report.inferred==0 or report.unresolved:
            continue
        possibilities=sorted(by_db[dbid],key=lambda item:hashkey("row",item[0]))
        valid=[]
        for idx,r in possibilities:
            if validate_gold_sql(path,r.get("SQL","")) is not None:
                continue
            question=r["question"]
            old,_=select_schema_context(cat,work,dbid,question)
            new,_=select_schema_context(corrected,work,dbid,question)
            # Treatment must be fully schema-only and actually exposed in every selected case.
            if old==new or "-> " not in old or ".None" not in old:
                continue
            old_plain=[l for l in old.splitlines() if not l.startswith("FK ")]
            new_plain=[l for l in new.splitlines() if not l.startswith("FK ")]
            if old_plain!=new_plain:
                raise ValueError("Semantic intervention changed non-FK prompt content")
            evidence=str(r.get("evidence") or "").strip()
            def make_user(schema):
                user=f"Question: {question}\nSchema context:\n{schema}"
                if evidence:user+=f"\nExternal evidence:\n{evidence}"
                return user
            u0,u1=make_user(old),make_user(new)
            if len(u0)>16_000 or len(u1)>16_000:
                continue
            valid.append({
                "source_index":idx,"db_id":dbid,"question":question,
                "original_user":u0,"fk_corrected_user":u1,
                "gold_sql":r["SQL"],
                "original_user_sha256":sha256(u0.encode()).hexdigest(),
                "corrected_user_sha256":sha256(u1.encode()).hexdigest(),
            })
            if len(valid)==N_PER_DB:break
        if len(valid)<N_PER_DB:
            continue
        frozen.extend(valid)
        db_attest.append({"db_id":dbid,"sha":manifest[dbid].get("lfs",{}).get("oid") or manifest[dbid]["oid"],
                          "implicit_fk_inferred":report.inferred,"new_question_count":len(valid)})
    if len(frozen)!=N or len(db_attest)!=N_DBS:
        raise RuntimeError(f"Only {len(frozen)}/{N} eligible novel-FK cases on {len(db_attest)}/{N_DBS} DBs")
    if {r["db_id"] for r in frozen}&excluded:
        raise ValueError("Database contamination")
    if len({r["source_index"] for r in frozen})!=N:
        raise ValueError("Duplicate selected source IDs")
    out.mkdir(parents=True,exist_ok=True)
    file=out/"prospective32.jsonl"
    file.write_text("".join(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n" for r in frozen))
    report={
        "protocol":"Qwen9B-BIRD-TRAIN-NOVEL-FK32-v1",
        "status":"FROZEN_BEFORE_INFERENCE",
        "selection_seed":SEED,
        "source_train_sha256":TRAIN_SHA,"source_dev_exclusion_sha256":DEV_SHA,
        "prior_train_sha256":prior_train_sha,
        "prior_validation_sha256":prior_val_sha,
        "cohort_sha256":sha256(file.read_bytes()).hexdigest(),
        "n":N,"physical_databases":N_DBS,
        "excluded_previous_databases":len(used_train|used_valid),
        "excluded_dev_databases":len({r["db_id"] for r in dev}),
        "questions_sha256":sha256("\n".join(str(r["source_index"]) for r in frozen).encode()).hexdigest(),
        "model_inference_occurred":False,
        "only_intervention":"SQLite implicit target FK resolution, same original Qwen3.5-9B system prompt/model/scaffold",
        "not_hidden":True,
        "dbs":db_attest,
    }
    (out/"freeze.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--dev",type=Path,required=True)
    p.add_argument("--prior",type=Path,required=True)
    p.add_argument("--work",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    print("BIRD_FK32_FROZEN",json.dumps(freeze(a.train,a.dev,a.prior,a.work,a.out),sort_keys=True),flush=True)


if __name__=="__main__":
    main()
