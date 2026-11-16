"""Prospective paired Qwen3.5-9B BIRD TRAIN physical-value16 test on 4 NEW DBs.

Only new variable: gold-blind exact physical database literal->column hints.
Model/seed/temperature/repair budget/query/physical DB identical in each pair.
Source cohort was frozen on CPU before any predictions. No BIRD DEV, no teacher.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from evaluation.run_bird import _ollama_model_digest
from scripts.prepare_birdtrain_qwen9b_sft_source import fetch_manifest,fetch_database
from semantic_sql.contracts import SQLCandidate
from semantic_sql.direct import _SYSTEM,_strip_fence
from semantic_sql.execution import execute_readonly,SQLExecutionError
from semantic_sql.inference import run_guarded
from semantic_sql.providers import OllamaProvider

MODEL="qwen3.5:9b-q4_K_M"
MODEL_DIGEST="56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
COHORT_SHA="4de8471341b38d0fb00ded1f17f265d33b528ec78b9779eb94937914df196955"
PROTOCOL="BIRD-TRAIN-UNSEEN-DB4X4-VALUE-EVIDENCE-V1"
SHARDS=8
PER_SHARD=2


def strict_score(db:Path,sql:str,gold_set:set,gold_bag:Counter)->dict:
    if not sql:
        return {"ex":False,"bag":False,"execution_ok":False}
    try:
        rows=execute_readonly(db,sql,max_rows=50_000)
        if rows.truncated:
            return {"ex":False,"bag":False,"execution_ok":False}
        return {
            "ex":set(rows.rows)==gold_set,
            "bag":Counter(rows.rows)==gold_bag,
            "execution_ok":True,
        }
    except (SQLExecutionError,ValueError):
        return {"ex":False,"bag":False,"execution_ok":False}


def split_prompt(full:str)->tuple[str,str,str|None]:
    if not full.startswith("Question: ") or "\nSchema context:\n" not in full:
        raise ValueError("Source prompt has an unknown format")
    question,remainder=full[10:].split("\nSchema context:\n",1)
    if "\nExternal evidence:\n" in remainder:
        schema,extra=remainder.split("\nExternal evidence:\n",1)
    else:
        schema,extra=remainder,None
    if not question or not schema:
        raise ValueError("Question/schema missing")
    return question,schema,extra


def load_frozen(path:Path,meta_path:Path)->list[dict]:
    raw=path.read_bytes()
    if sha256(raw).hexdigest()!=COHORT_SHA:
        raise RuntimeError("Pre-inference frozen NEW DB source SHA drift")
    meta=json.loads(meta_path.read_text())
    if (meta.get("status")!="FROZEN_BEFORE_ANY_MODEL_INFERENCE"
        or meta.get("protocol")!=PROTOCOL
        or meta.get("cohort_sha256")!=COHORT_SHA
        or meta.get("questions")!=16
        or meta.get("physical_databases")!=4
        or meta.get("new_model_inference") is not False
        or meta.get("conditioned_on_physical_entity_value_match") is not True):
        raise ValueError("Pinned source preregistration not found")
    rows=[json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if len(rows)!=16 or len({r["source_index"] for r in rows})!=16:
        raise ValueError("Duplicate or missing frozen source IDs")
    if len({r["db_id"] for r in rows})!=4 or any(len({z["db_id"] for z in rows[i:i+4]})!=1 for i in range(0,16,4)):
        raise ValueError("Not exactly four 4-case physical source DBs")
    if set(r["db_id"] for r in rows)!=set(meta.get("source_databases",[])):
        raise ValueError("Frozen source DB names changed")
    for r in rows:
        a,b=r["original_user"],r["grounded_user"]
        if (not b.startswith(a+"\n")
            or not r["exact_value_evidence"]
            or sha256(a.encode()).hexdigest()!=r["original_prompt_sha256"]
            or sha256(b.encode()).hexdigest()!=r["grounded_prompt_sha256"]
            or not b[len(a):].lstrip().startswith("Verified database value-column candidates")):
            raise ValueError("Evidence-only source prompt intervention drift")
    return rows


def run_shard(path:Path,meta:Path,shard:int,work:Path)->dict:
    if shard not in range(SHARDS):raise ValueError("Shard not in [0,7]")
    rows=load_frozen(path,meta)[shard*PER_SHARD:(shard+1)*PER_SHARD]
    if len(rows)!=2 or len({r["db_id"] for r in rows})!=1:
        raise ValueError("A shard must have exactly two same-DB cases")
    if _ollama_model_digest("http://127.0.0.1:11434",MODEL)!=MODEL_DIGEST:
        raise RuntimeError("Qwen3.5-9B model digest changed")
    manifest=fetch_manifest()
    dbid=rows[0]["db_id"]
    db=fetch_database(dbid,manifest[dbid],work)
    provider=OllamaProvider(model=MODEL,temperature=0.0,seed=42,num_predict=768,timeout_s=900.)
    results=[]
    for idx,row in enumerate(rows):
        gold=execute_readonly(db,row["gold_sql"],max_rows=50_000)
        if gold.truncated:raise RuntimeError("Publisher gold truncated")
        gold_set,gold_bag=set(gold.rows),Counter(gold.rows)
        arms={}
        # ABBA over physically repeated 2-case blocks, not always baseline first.
        order=("original","grounded") if (shard+idx)%2==0 else ("grounded","original")
        for arm in order:
            prompt=row["original_user"] if arm=="original" else row["grounded_user"]
            question,schema,evidence=split_prompt(prompt)
            started=perf_counter()
            sql=_strip_fence(provider.complete_text(system=_SYSTEM,user=prompt))
            direct=strict_score(db,sql,gold_set,gold_bag)
            guarded=run_guarded(
                database=db,provider=provider,question=question,
                schema_context=schema,evidence=evidence,
                initial_candidate=SQLCandidate(sql=sql,attempt=0),
                max_repairs=2,max_rows=50_000,
            )
            final=guarded.candidate.sql if guarded.candidate else ""
            strict=strict_score(db,final,gold_set,gold_bag)
            arms[arm]={
                "direct":direct,"guarded":strict,
                "initial_sql":sql,"final_sql":final,
                "guarded_status":guarded.status,
                "repair_attempts":guarded.candidate.attempt if guarded.candidate else None,
                "elapsed_s":round(perf_counter()-started,2),
            }
            print("VALUE16_PAIR",shard,row["source_index"],arm,"strict_set_ex",strict["ex"],flush=True)
        results.append({
            "source_index":row["source_index"],"db_id":dbid,
            "order":list(order),"arms":arms,
        })
    return {
        "status":"complete","protocol":PROTOCOL,
        "cohort_sha256":COHORT_SHA,"model_digest":MODEL_DIGEST,
        "shard":shard,"db_id":dbid,"n":PER_SHARD,
        "model":MODEL,"seed":42,"temperature":0.0,
        "num_predict":768,"max_repairs":2,
        "source_only_value_lookup":True,"no_bird_dev":True,
        "cases":results,
    }


def aggregate(paths:list[Path])->dict:
    if len(paths)!=SHARDS:raise ValueError("Exactly eight physical paired shards required")
    shards=[json.loads(p.read_text()) for p in paths]
    if sorted(r.get("shard") for r in shards)!=list(range(SHARDS)):
        raise ValueError("Shard IDs missing or repeated")
    if any(r.get("status")!="complete" or r.get("protocol")!=PROTOCOL
           or r.get("cohort_sha256")!=COHORT_SHA or r.get("model_digest")!=MODEL_DIGEST
           or len(r.get("cases",[]))!=PER_SHARD
           or r.get("source_only_value_lookup") is not True
           for r in shards):
        raise ValueError("Physical model, source or result protocol changed")
    rows=[r for shard in sorted(shards,key=lambda s:s["shard"]) for r in shard["cases"]]
    if len(rows)!=16 or len({r["source_index"] for r in rows})!=16 or len({r["db_id"] for r in rows})!=4:
        raise ValueError("Incomplete or duplicate original source case")
    summary={}
    for arm in ("original","grounded"):
        v=[r["arms"][arm] for r in rows]
        summary[arm]={
            "strict_set_ex":sum(s["guarded"]["ex"] for s in v),
            "strict_bag_ex":sum(s["guarded"]["bag"] for s in v),
            "direct_set_ex":sum(s["direct"]["ex"] for s in v),
            "guarded_execution_ok":sum(s["guarded"]["execution_ok"] for s in v),
            "total_elapsed_s":round(sum(s["elapsed_s"] for s in v),2),
        }
    rescued=[r["source_index"] for r in rows if not r["arms"]["original"]["guarded"]["ex"] and r["arms"]["grounded"]["guarded"]["ex"]]
    regressed=[r["source_index"] for r in rows if r["arms"]["original"]["guarded"]["ex"] and not r["arms"]["grounded"]["guarded"]["ex"]]
    by_db={db:{
        "n":sum(r["db_id"]==db for r in rows),
        "original":sum(r["db_id"]==db and r["arms"]["original"]["guarded"]["ex"] for r in rows),
        "grounded":sum(r["db_id"]==db and r["arms"]["grounded"]["guarded"]["ex"] for r in rows),
    } for db in sorted({r["db_id"] for r in rows})}
    return {
        "status":"COMPLETE_PREDECLARED_PAIRED_SCREEN_NOT_PROMOTED",
        "protocol":PROTOCOL,"cohort_sha256":COHORT_SHA,
        "n":16,"distinct_physical_dbs":4,
        "model":MODEL,"model_digest":MODEL_DIGEST,
        "arms":summary,"by_db":by_db,
        "rescues":len(rescued),"regressions":len(regressed),
        "rescued_ids":rescued,"regressed_ids":regressed,
        "net_strict_set_ex":len(rescued)-len(regressed),
        "decision_rule":"net <=0 reject; if >=3 rescues net and 0 regressions spread >=2 DBs, candidate replication ONLY, never automatic production promotion",
        "source_conditioned_on_value_match":True,
        "no_heldout_dev":True,
        "claim_boundary":"Publisher BIRD TRAIN source only, 16 frozen question/DB-distinct cases selected for exact-match value grounding; label semantic quality not independently human adjudicated. Small screen, not hidden generalization.",
        "cases":rows,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cohort",type=Path)
    parser.add_argument("--meta",type=Path)
    parser.add_argument("--shard",type=int)
    parser.add_argument("--db-work",type=Path,default=Path("original-source"))
    parser.add_argument("--inputs",type=Path,nargs="+")
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.inputs:
        result=aggregate(args.inputs)
        print("VALUE16_QWEN9B_PAIRED_FINAL",json.dumps({k:v for k,v in result.items() if k!="cases"},sort_keys=True),flush=True)
    else:
        if args.cohort is None or args.meta is None or args.shard is None:
            parser.error("cohort, meta and shard required")
        result=run_shard(args.cohort,args.meta,args.shard,args.db_work)
    args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")

if __name__=="__main__":
    main()
