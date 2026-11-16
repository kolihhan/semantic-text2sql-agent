"""Preregistered matched Qwen3.5-9B test: only fix bad SQLite FK 'to=None'.

32 original BIRD TRAIN questions on four physical databases excluded from all
prior BIRD TRAIN train242/val64 and from BIRD DEV. Freeze SHA and DBs before model.
Identical model digest, system, seed, temp, decoding budget and Guarded repairs.
No training, no stronger teacher, no test-time gold in the LLM prompt.
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
from semantic_sql.execution import execute_readonly, SQLExecutionError
from semantic_sql.inference import run_guarded
from semantic_sql.providers import OllamaProvider

FROZEN_SHA="ee1b5035302d36a1241952bd2e5acd3dfc2518d1fba2581f557e04e179960535"
MODEL="qwen3.5:9b-q4_K_M"
MODEL_DIGEST="56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
def gold_score(database:Path,sql:str,gold_rows:set,gold_bag:Counter)->dict:
    """Same strict execution-match scorer used in the frozen original baseline."""
    if not sql:
        return {"ex":False,"bag":False,"execution_ok":False}
    try:
        got=execute_readonly(database,sql,max_rows=50_000)
        if got.truncated:
            return {"ex":False,"bag":False,"execution_ok":False}
        return {
            "ex":set(got.rows)==gold_rows,
            "bag":Counter(got.rows)==gold_bag,
            "execution_ok":True,
        }
    except (SQLExecutionError,ValueError):
        return {"ex":False,"bag":False,"execution_ok":False}


N_SHARDS=8
N_PER_SHARD=4
PROTOCOL="Qwen9B-BIRD-TRAIN-NOVEL-FK32-v1"

def load_cohort(path:Path,metadata:Path):
    blob=path.read_bytes()
    if sha256(blob).hexdigest()!=FROZEN_SHA:
        raise RuntimeError("Prospectively frozen full cohort byte hash changed")
    rows=[json.loads(x) for x in blob.decode().splitlines() if x.strip()]
    meta=json.loads(metadata.read_text())
    if (meta.get("cohort_sha256")!=FROZEN_SHA
        or meta.get("status")!="FROZEN_BEFORE_INFERENCE"
        or meta.get("protocol")!=PROTOCOL
        or meta.get("n")!=32 or meta.get("physical_databases")!=4
        or meta.get("model_inference_occurred") is not False):
        raise ValueError("Frozen protocol or candidate count drift")
    if len(rows)!=32 or len({r["source_index"] for r in rows})!=32:
        raise ValueError("Frozen 32 source IDs are incomplete/duplicated")
    dbs=[r["db_id"] for r in rows]
    if len(set(dbs))!=4 or any(len(set(dbs[i:i+8]))!=1 for i in range(0,32,8)):
        raise ValueError("Four distinct 8-question source DB blocks required")
    if set(dbs)!={d["db_id"] for d in meta["dbs"]}:
        raise ValueError("Physical source DB provenance drift")
    for r in rows:
        if (r["original_user"]==r["fk_corrected_user"]
            or ".None" not in r["original_user"]
            or "-> " not in r["fk_corrected_user"]
            or r["original_user_sha256"]!=sha256(r["original_user"].encode()).hexdigest()
            or r["corrected_user_sha256"]!=sha256(r["fk_corrected_user"].encode()).hexdigest()):
            raise ValueError("Original vs FK-corrected intervention integrity changed")
        a=r["original_user"].split("\nSchema context:\n",1)
        b=r["fk_corrected_user"].split("\nSchema context:\n",1)
        if len(a)!=2 or len(b)!=2 or a[0]!=b[0]:
            raise ValueError("Question changed")
        if [l for l in a[1].splitlines() if not l.startswith("FK ")] != [l for l in b[1].splitlines() if not l.startswith("FK ")]:
            raise ValueError("The FK-only intervention changed another prompt component")
    return rows,meta


def run_shard(path:Path,metadata:Path,shard:int,work:Path):
    if not 0<=shard<N_SHARDS:
        raise ValueError("Invalid shard")
    rows,meta=load_cohort(path,metadata)
    chosen=rows[shard*N_PER_SHARD:(shard+1)*N_PER_SHARD]
    if len({r["db_id"] for r in chosen})!=1:
        raise ValueError("Same DB expected within each 4-case shard")
    if _ollama_model_digest("http://127.0.0.1:11434",MODEL)!=MODEL_DIGEST:
        raise RuntimeError("Exact fixed Qwen3.5-9B digest mismatch")
    dbid=chosen[0]["db_id"]
    manifest=fetch_manifest()
    physical=fetch_database(dbid,manifest[dbid],work)
    provider=OllamaProvider(model=MODEL,temperature=0.0,seed=42,num_predict=768,timeout_s=900.0)
    cases=[]
    for local_index,row in enumerate(chosen):
        gold=execute_readonly(physical,row["gold_sql"],max_rows=50_000)
        if gold.truncated:
            raise RuntimeError("Pinned publisher SQL produced truncated rows")
        gold_set=set(gold.rows);gold_bag=Counter(gold.rows)
        arms={}
        # Balanced within one source DB: ABBA prevents one treatment always first.
        order=["original","fk_corrected"] if (local_index%4 in (0,3)) else ["fk_corrected","original"]
        for arm in order:
            prompt=row["original_user"] if arm=="original" else row["fk_corrected_user"]
            split=prompt.split("\nSchema context:\n",1)
            question=split[0][10:]
            schema,*evidence_parts=split[1].split("\nExternal evidence:\n",1)
            evidence=evidence_parts[0] if evidence_parts else None
            start=perf_counter()
            initial_sql=_strip_fence(provider.complete_text(system=_SYSTEM,user=prompt))
            direct=gold_score(physical,initial_sql,gold_set,gold_bag)
            result=run_guarded(
                database=physical,provider=provider,question=question,
                schema_context=schema,evidence=evidence,
                initial_candidate=SQLCandidate(sql=initial_sql,attempt=0),
                max_repairs=2,max_rows=50_000,
            )
            guarded_sql=result.candidate.sql if result.candidate else ""
            guarded=gold_score(physical,guarded_sql,gold_set,gold_bag)
            arms[arm]={
                "direct":direct,"guarded":guarded,
                "direct_sql":initial_sql,"guarded_sql":guarded_sql,
                "status":result.status,
                "model_calls":1+(result.candidate.attempt if result.candidate else 0),
                "seconds":round(perf_counter()-start,2),
                "input_sha256":row["original_user_sha256"] if arm=="original" else row["corrected_user_sha256"],
            }
            print("FK32_REAL_PAIR",shard,row["source_index"],arm,
                  "ex",guarded["ex"],"direct",direct["ex"],
                  "exec",guarded["execution_ok"],flush=True)
        cases.append({
            "source_index":row["source_index"],"db_id":dbid,"order":order,
            "arms":arms,
        })
    return {
        "status":"complete","protocol":PROTOCOL,"cohort_sha256":FROZEN_SHA,
        "n":N_PER_SHARD,"shard":shard,"db_id":dbid,
        "model":MODEL,"model_digest":MODEL_DIGEST,
        "temp":0.0,"seed":42,"num_predict":768,"max_repairs":2,
        "same_model":True,"source_questions_train_only":True,"no_teacher":True,
        "cases":cases,
    }


def aggregate(paths:list[Path]):
    if len(paths)!=N_SHARDS:
        raise ValueError(f"Require all {N_SHARDS} shards, no partial success")
    shards=[json.loads(p.read_text()) for p in paths]
    if sorted(r["shard"] for r in shards)!=list(range(N_SHARDS)):
        raise ValueError("Missing/duplicate shard")
    if any(r.get("status")!="complete" or r.get("protocol")!=PROTOCOL
           or r.get("cohort_sha256")!=FROZEN_SHA or r.get("model_digest")!=MODEL_DIGEST
           or r.get("n")!=N_PER_SHARD or len(r.get("cases",[]))!=N_PER_SHARD
           or r.get("same_model") is not True
           for r in shards):
        raise ValueError("Run configuration/outcome mismatch or incomplete")
    rows=[row for s in shards for row in s["cases"]]
    if len(rows)!=32 or len({r["source_index"] for r in rows})!=32 or len({r["db_id"] for r in rows})!=4:
        raise ValueError("Model outputs do not match frozen complete population")
    arms=("original","fk_corrected")
    summary={}
    for arm in arms:
        vals=[row["arms"][arm] for row in rows]
        summary[arm]={
            "strict_set_ex":sum(x["guarded"]["ex"] for x in vals),
            "direct_set_ex":sum(x["direct"]["ex"] for x in vals),
            "guarded_execution_ok":sum(x["guarded"]["execution_ok"] for x in vals),
            "tool_model_calls":sum(x["model_calls"] for x in vals),
            "total_latency_s":round(sum(x["seconds"] for x in vals),1),
        }
    rescued=[r["source_index"] for r in rows if not r["arms"]["original"]["guarded"]["ex"] and r["arms"]["fk_corrected"]["guarded"]["ex"]]
    regressed=[r["source_index"] for r in rows if r["arms"]["original"]["guarded"]["ex"] and not r["arms"]["fk_corrected"]["guarded"]["ex"]]
    by_db={}
    for db in sorted({x["db_id"] for x in rows}):
        selected=[x for x in rows if x["db_id"]==db]
        by_db[db]={
            "n":len(selected),
            "original":sum(x["arms"]["original"]["guarded"]["ex"] for x in selected),
            "fk_corrected":sum(x["arms"]["fk_corrected"]["guarded"]["ex"] for x in selected),
        }
    return {
        "status":"PROSPECTIVE_PAIRED_SCREEN_COMPLETE",
        "protocol":PROTOCOL,"cohort_sha256":FROZEN_SHA,
        "n":32,"physical_databases":4,
        "arms":summary,
        "rescues":len(rescued),"regressions":len(regressed),
        "rescued_ids":rescued,"regressed_ids":regressed,
        "net_ex_change":len(rescued)-len(regressed),
        "by_database":by_db,
        "not_promoted":True,
        "claim_boundary":"Small screening 32/4 physical BIRD TRAIN DBs previously unused, selected for FK metadata defect only; not hidden/public-independent and no stronger teacher. Do not claim broad transfer or use BIRD DEV300.",
        "cases":rows,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cohort",type=Path)
    p.add_argument("--meta",type=Path)
    p.add_argument("--shard",type=int)
    p.add_argument("--work",type=Path,default=Path("physical-db"))
    p.add_argument("--inputs",type=Path,nargs="+")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.inputs:
        result=aggregate(a.inputs)
        print("FK32_PROSPECTIVE_FINAL",json.dumps({k:v for k,v in result.items() if k!="cases"},sort_keys=True),flush=True)
    else:
        if a.cohort is None or a.meta is None or a.shard is None:
            p.error("Need cohort, frozen metadata and shard")
        result=run_shard(a.cohort,a.meta,a.shard,a.work)
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")


if __name__=="__main__":
    main()
