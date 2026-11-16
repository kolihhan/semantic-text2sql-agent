"""Freeze 16 BIRD TRAIN-only questions from FOUR NEW physical source DBs.

No model outputs or scores accessed during selection. Excludes:
- earlier 16 SFT TRAIN DBs and 4 source validation DBs;
- the 4 FK-only experiment source DBs;
- every BIRD DEV database and all previously used question texts.
Select only original publisher SQL compiled on source SQLite and questions
with independently located exact value-to-column physical evidence.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

from evaluation.physical_value_grounding import candidate_literals,physical_value_columns,evidence_string
from evaluation.schema_context import select_schema_context
from scripts.prepare_birdtrain_qwen9b_sft_source import (
    TRAIN_SHA,DEV_SHA,load_checked,fetch_manifest,fetch_database,
    normalized,validate_gold_sql,
)
from scripts.audit_birdtrain242_label_contracts import question_sql_flags
from semantic_sql.catalog import DatabaseCatalog

SEED="qwen9b-second-novel-db-physical-value16-seeded-20261117-v1"
N_DB=4
N_PER_DB=4
MAX_DB_BYTES=96*1024*1024
MAX_EXAMINED_DBS=25


def keyed(kind:str,value)->str:
    return sha256(f"{SEED}:{kind}:{value}".encode()).hexdigest()


def freeze(train_file:Path,dev_file:Path,prior_train:Path,
           prior_validation:Path,prior_fk32:Path,prior_value16:Path,root:Path,out:Path):
    train=load_checked(train_file,TRAIN_SHA)
    dev=load_checked(dev_file,DEV_SHA)
    excluded_db={r["db_id"] for r in dev}
    seen={normalized(r.get("question","")) for r in dev}
    previous={}
    for label,path in (("prior_train",prior_train),("prior_validation",prior_validation)):
        original=path.read_bytes()
        prior=[json.loads(line) for line in original.decode().splitlines() if line.strip()]
        previous[label]=sha256(original).hexdigest()
        if len(prior)!=(242 if label=="prior_train" else 64):
            raise RuntimeError("Old source cohort not verified")
        for r in prior:
            excluded_db.add(r["source_db_id"])
            seen.add(normalized(r["messages"][1]["content"].split("\nSchema context:\n",1)[0][10:]))
    fk_data=prior_fk32.read_bytes()
    previous["prior_fk32"]=sha256(fk_data).hexdigest()
    fk=[json.loads(line) for line in fk_data.decode().splitlines() if line.strip()]
    if len(fk)!=32:raise RuntimeError("Prior FK32 cohort changed")
    for r in fk:
        excluded_db.add(r["db_id"])
        seen.add(normalized(r["question"]))
    value_data=prior_value16.read_bytes()
    previous["prior_value16"]=sha256(value_data).hexdigest()
    if previous["prior_value16"]!="4de8471341b38d0fb00ded1f17f265d33b528ec78b9779eb94937914df196955":
        raise ValueError("First source-value16 cohort byte SHA mismatch")
    prior_value_rows=[json.loads(line) for line in value_data.decode().splitlines() if line.strip()]
    if len(prior_value_rows)!=16 or len({r["source_index"] for r in prior_value_rows})!=16:
        raise RuntimeError("Original source-value16 frozen set incomplete")
    for row in prior_value_rows:
        excluded_db.add(row["db_id"])
        seen.add(normalized(row["question"]))
    if len(excluded_db - {r["db_id"] for r in dev})!=28:
        raise ValueError("Expected 28 distinct previously observed BIRD TRAIN source DBs")
    manifest=fetch_manifest()
    by_db=defaultdict(list)
    for idx,r in enumerate(train):
        db=r.get("db_id")
        q=normalized(r.get("question",""))
        if db not in manifest or db in excluded_db or not q or q in seen:
            continue
        if manifest[db]["size"]>MAX_DB_BYTES:continue
        seen.add(q)
        by_db[db].append((idx,r))
    eligible=sorted((db for db,rows in by_db.items() if len(rows)>=N_PER_DB),
                    key=lambda db:keyed("db",db))
    frozen=[]; selected=[]; checks=0
    for dbid in eligible:
        if len(selected)==N_DB or checks>=MAX_EXAMINED_DBS:break
        checks+=1
        physical=fetch_database(dbid,manifest[dbid],root)
        catalog=DatabaseCatalog.from_sqlite(physical)
        accepted=[]
        for idx,row in sorted(by_db[dbid],key=lambda x:keyed("row",x[0]))[:35]:
            question=str(row["question"])
            if not candidate_literals(question):continue
            source_sql=str(row.get("SQL") or "")
            if validate_gold_sql(physical,source_sql) is not None:
                continue
            if question_sql_flags(question,source_sql):
                continue
            matches=physical_value_columns(physical,question)
            if not matches:continue
            schema,_=select_schema_context(catalog,root,dbid,question)
            publisher_evidence=str(row.get("evidence") or "").strip()
            original=f"Question: {question}\nSchema context:\n{schema}"
            if publisher_evidence:
                original+=f"\nExternal evidence:\n{publisher_evidence}"
            physical_evidence=evidence_string(matches)
            if not physical_evidence:continue
            if len(original)+len(physical_evidence)>16000:continue
            augmented=original+"\n"+physical_evidence
            accepted.append({
                "source_index":idx,"db_id":dbid,
                "question":question,
                "gold_sql":source_sql,
                "original_user":original,
                "grounded_user":augmented,
                "exact_value_evidence":matches,
                "original_prompt_sha256":sha256(original.encode()).hexdigest(),
                "grounded_prompt_sha256":sha256(augmented.encode()).hexdigest(),
            })
            if len(accepted)==N_PER_DB:break
        if len(accepted)<N_PER_DB:continue
        frozen.extend(accepted);selected.append(dbid)
    if len(selected)!=N_DB or len(frozen)!=N_DB*N_PER_DB:
        raise RuntimeError(f"Only {len(selected)} eligible source DBs, {len(frozen)} case matches")
    if set(selected)&excluded_db or len({r["source_index"] for r in frozen})!=len(frozen):
        raise RuntimeError("Source DB or row reuse")
    out.mkdir(parents=True,exist_ok=True)
    data=out/"physical-value16.jsonl"
    data.write_text("".join(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n" for r in frozen))
    report={
        "status":"FROZEN_BEFORE_ANY_MODEL_INFERENCE",
        "protocol":"BIRD-TRAIN-SECOND-UNSEEN-DB4X4-VALUE-EVIDENCE-V1",
        "selection_seed":SEED,
        "train_source_sha256":TRAIN_SHA,
        "dev_exclusion_sha256":DEV_SHA,
        "excluded_prior_train_source_databases":28,
        "excluded_dev_databases":len({r["db_id"] for r in dev}),
        "source_previous_checksums":previous,
        "cohort_sha256":sha256(data.read_bytes()).hexdigest(),
        "questions":len(frozen),"physical_databases":len(selected),
        "source_databases":selected,
        "physical_value_match_count":sum(len(r["exact_value_evidence"]) for r in frozen),
        "selection_no_reference_execution_match_outcomes":True,
        "same_qwen_base_only":True,
        "new_model_inference":False,
        "publisher_references_compiled_not_human_semantic_verified":True,
        "conditioned_on_physical_entity_value_match":True,
    }
    (out/"freeze.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report


def main():
    p=argparse.ArgumentParser()
    for k in ("train","dev","prior-train","prior-validation","prior-fk32","prior-value16","db-work","out"):
        p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args()
    print("FROZEN_NEW_DB_VALUE16_SOURCE",json.dumps(
        freeze(a.train,a.dev,a.prior_train,a.prior_validation,a.prior_fk32,a.prior_value16,a.db_work,a.out),
        sort_keys=True),flush=True)


if __name__=="__main__":
    main()
