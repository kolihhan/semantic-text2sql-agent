"""Original Qwen3.5-9B baseline on BIRD TRAIN-only, DB-heldout SFT validation64.

This is the prospective baseline for a future unchanged-base QLoRA adapter.
NO training, NO treatment, NO BIRD DEV300, NO independent semantic judge.
Pinned original publisher SQLite and original source-verified JSONL.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha1,sha256
import json
from pathlib import Path
from time import perf_counter
from urllib.parse import quote
from urllib.request import urlopen

from evaluation.run_bird import _ollama_model_digest
from semantic_sql.contracts import SQLCandidate
from semantic_sql.direct import _strip_fence
from semantic_sql.execution import SQLExecutionError, execute_readonly
from semantic_sql.inference import run_guarded
from semantic_sql.providers import OllamaProvider

SOURCE_SHA = "cc112305bc9de33a03cea704f2edecfdfb45e62cc953ae498cea27fbed93b506"
MODEL_DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
MODEL = "qwen3.5:9b-q4_K_M"
URL_BASE="https://huggingface.co"
DATASET="Emulated-Inc/text-to-sql-with-evidence-training-pool"
REV="cd2c4ea321075aa53ce7dfe52deaaffa505d046a"
SHARDS=8
PER_SHARD=8

def read_inputs(path:Path)->list[dict]:
    data=path.read_bytes()
    if sha256(data).hexdigest()!=SOURCE_SHA:
        raise RuntimeError("Prepared original TRAIN validation cohort hash drift")
    rows=[json.loads(s) for s in data.decode("utf8").splitlines() if s.strip()]
    if len(rows)!=64 or len({r["source_index"] for r in rows})!=64:
        raise ValueError("Expected 64 distinct original published SQL validations")
    if len({r["source_db_id"] for r in rows})!=4:
        raise ValueError("Expected exactly 4 isolated physical BIRD TRAIN DBs")
    for r in rows:
        m=r.get("messages",[])
        if (len(m)!=3 or [x.get("role") for x in m]!=["system","user","assistant"]
            or sha256(m[1]["content"].encode("utf8")).hexdigest()!=r.get("input_sha256")):
            raise ValueError("Original SFT source/prompt integrity mismatch")
    return sorted(rows,key=lambda r:(r["source_db_id"],r["source_index"]))

def remote_manifest():
    with urlopen(f"{URL_BASE}/api/datasets/{DATASET}/tree/{REV}/raw/bird/train_databases?recursive=true&limit=1000",timeout=60) as f:
        rows=json.load(f)
    out={Path(r["path"]).stem:r for r in rows if r["path"].endswith(".sqlite")}
    if len(out)!=51:raise ValueError("Incomplete original BIRD TRAIN catalog")
    return out

def get_db(dbid:str, meta:dict, root:Path)->Path:
    url=f"{URL_BASE}/datasets/{DATASET}/resolve/{REV}/raw/bird/train_databases/{quote(dbid)}/{quote(dbid)}.sqlite"
    with urlopen(url,timeout=180) as response:
        raw=response.read(meta["size"]+1)
    if len(raw)!=meta["size"]:raise RuntimeError("database bytes length drift")
    if "lfs" in meta:
        if sha256(raw).hexdigest()!=meta["lfs"]["oid"]:raise RuntimeError("database SHA mismatch")
    elif sha1(f"blob {len(raw)}\0".encode()+raw).hexdigest()!=meta["oid"]:
        raise RuntimeError("database git blob SHA mismatch")
    root.mkdir(parents=True,exist_ok=True)
    path=root/(dbid+".sqlite")
    path.write_bytes(raw)
    return path

def split_prompt(full_user:str)->tuple[str,str,str|None]:
    if not full_user.startswith("Question: ") or "\nSchema context:\n" not in full_user:
        raise ValueError("Original direct generator prompt changed")
    question,remainder=full_user[10:].split("\nSchema context:\n",1)
    if "\nExternal evidence:\n" in remainder:
        schema,evidence=remainder.split("\nExternal evidence:\n",1)
    else:
        schema,evidence=remainder,None
    if not question or not schema:raise ValueError("Empty source question or schema")
    return question,schema,evidence

def gold_score(db:Path,predicted:str,gold_rows, gold_bag):
    if not predicted:return {"ex":False,"bag":False,"execution_ok":False}
    try:
        got=execute_readonly(db,predicted,max_rows=50_000)
        if got.truncated:return {"ex":False,"bag":False,"execution_ok":False}
        return {"ex":set(got.rows)==gold_rows,"bag":Counter(got.rows)==gold_bag,"execution_ok":True}
    except (SQLExecutionError,ValueError):
        return {"ex":False,"bag":False,"execution_ok":False}

def run_shard(validation:Path, shard:int,work:Path,model:str=MODEL):
    if not 0<=shard<SHARDS:raise ValueError("Bad shard")
    rows=read_inputs(validation)
    selected=rows[PER_SHARD*shard:PER_SHARD*(shard+1)]
    if len(selected)!=PER_SHARD or len({r["source_db_id"] for r in selected})!=1:
        raise RuntimeError("Expected 8 cases from exactly 1 validation DB per shard")
    if _ollama_model_digest("http://127.0.0.1:11434",model)!=MODEL_DIGEST:
        raise RuntimeError("Unchanged Qwen3.5-9B digest mismatch")
    dbid=selected[0]["source_db_id"]
    manifest=remote_manifest()
    db=get_db(dbid,manifest[dbid],work)
    provider=OllamaProvider(model=model,timeout_s=900.)
    outcomes=[]
    for record in selected:
        prompt=record["messages"][1]["content"]
        system=record["messages"][0]["content"]
        gold_sql=record["messages"][2]["content"]
        question,schema,evidence=split_prompt(prompt)
        gold=execute_readonly(db,gold_sql,max_rows=50_000)
        if gold.truncated:raise RuntimeError("Publisher gold truncated unexpectedly")
        gold_rows=set(gold.rows);gold_bag=Counter(gold.rows)
        started=perf_counter()
        direct_sql=_strip_fence(provider.complete_text(system=system,user=prompt))
        direct=gold_score(db,direct_sql,gold_rows,gold_bag)
        result=run_guarded(
            database=db,provider=provider,question=question,
            schema_context=schema,evidence=evidence,
            initial_candidate=SQLCandidate(sql=direct_sql,attempt=0),
            max_repairs=2,max_rows=50_000
        )
        guarded_sql=result.candidate.sql if result.candidate else ""
        guarded=gold_score(db,guarded_sql,gold_rows,gold_bag)
        outcomes.append({
            "source_index":record["source_index"],
            "db_id":dbid,
            "prompt_sha256":record["input_sha256"],
            "direct_sql":direct_sql,
            "guarded_sql":guarded_sql,
            "guarded_status":result.status,
            "repairs":result.candidate.attempt if result.candidate else None,
            "direct":direct,"guarded":guarded,
            "latency_s":round(perf_counter()-started,3)
        })
        print("QWEN9B_VALIDATION_BASE",shard,record["source_index"],
              "direct",direct["ex"],"guarded",guarded["ex"],flush=True)
    return {
        "status":"complete",
        "split":"BIRD TRAIN-only, DB-disjoint original publisher SFT-validation64",
        "validation_sha256":SOURCE_SHA,"model_digest":MODEL_DIGEST,
        "model":model,"shard":shard,"n":PER_SHARD,"database_id":dbid,
        "no_training":True,"no_dev300":True,"cases":outcomes,
    }

def aggregate(paths:list[Path]):
    if len(paths)!=SHARDS:raise ValueError("All 8 shards mandatory")
    shards=[json.loads(p.read_text(encoding="utf8")) for p in paths]
    if sorted(s["shard"] for s in shards)!=list(range(SHARDS)):
        raise RuntimeError("Duplicate/incomplete shard identities")
    if any(s.get("status")!="complete" or s.get("validation_sha256")!=SOURCE_SHA
           or s.get("model_digest")!=MODEL_DIGEST or len(s.get("cases",[]))!=8
           or s.get("no_training") is not True or s.get("no_dev300") is not True
           for s in shards):
        raise RuntimeError("Partial or mismatched independent validation evidence")
    rows=[r for s in shards for r in s["cases"]]
    if len(rows)!=64 or len({r["source_index"] for r in rows})!=64 or len({r["db_id"] for r in rows})!=4:
        raise RuntimeError("Sample count/DB identity gate failed")
    return {
        "status":"complete","n":64,"physical_train_validation_dbs":4,
        "model":MODEL,"model_digest":MODEL_DIGEST,
        "source_validation_sha256":SOURCE_SHA,
        "direct_ex":sum(r["direct"]["ex"] for r in rows),
        "guarded_ex":sum(r["guarded"]["ex"] for r in rows),
        "guarded_bag":sum(r["guarded"]["bag"] for r in rows),
        "direct_exec":sum(r["direct"]["execution_ok"] for r in rows),
        "guarded_exec":sum(r["guarded"]["execution_ok"] for r in rows),
        "guarded_vs_direct_rescues":sum(not r["direct"]["ex"] and r["guarded"]["ex"] for r in rows),
        "guarded_vs_direct_regressions":sum(r["direct"]["ex"] and not r["guarded"]["ex"] for r in rows),
        "total_qa_generation_s":round(sum(r["latency_s"] for r in rows),1),
        "claim_boundary":"Original 4 BIRD TRAIN DBs, fully disjoint from BIRD DEV and 16 train DBs; baseline only, no trained adapter, no inference uplift",
        "cases":rows,
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--validation",type=Path)
    parser.add_argument("--shard",type=int)
    parser.add_argument("--work",type=Path,default=Path("db-sources"))
    parser.add_argument("--inputs",type=Path,nargs="+")
    parser.add_argument("--output",type=Path,required=True)
    a=parser.parse_args()
    if a.inputs:
        result=aggregate(a.inputs)
        print("QWEN9B_TRAIN_ONLY_VALIDATION64_FINAL",
              json.dumps({k:v for k,v in result.items() if k!="cases"},sort_keys=True),flush=True)
    else:
        if a.validation is None or a.shard is None:parser.error("validation and shard required")
        result=run_shard(a.validation,a.shard,a.work)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")

if __name__=="__main__":
    main()
