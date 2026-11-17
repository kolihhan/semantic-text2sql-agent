"""Original Qwen3.5-9B CPU single-call timer + once-only candidate cache.

Three intentionally distinct stages:
  speed: measure real single SQL generation from 8 ORIGINAL TRAIN source prompts;
         not from DEV/HOLDOUT, no label or database needed, no repair.
  generate: source-verified frozen DEV500, exact original Qwen3.5-9B with
         K seeded stochastic candidates, prompt/context SHA, no gold read.
  score: execute cached candidates on SQLite after generation; write each
         shard's preselected consensus decision and EX labels (no model).
  aggregate: require all 500 manifest IDs, report actual paired confidence
         intervals; do not use HOLDOUT300 until a single locked final check.
No model replacement, SQL QLoRA, teacher, or extra repair.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from urllib.request import Request, urlopen

from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.direct import _SYSTEM
from semantic_sql.execution import SQLExecutionError
from evaluation.run_bird import _ollama_model_digest, _resolve_database
from evaluation.schema_context import select_schema_context
from evaluation.kcache_protocol import (
    PROTOCOL, MANIFEST, MODEL_DIGEST, MODEL_NAME, SOURCE_SHA256,
    candidate_seed, file_sha256, read_frozen_manifest, source_cases,
    per_case_offline, aggregate_offline
)

PINNED_TRAIN_BENCH_SOURCE_SHA = "ca68478128821b253dca9db600b101b9140fa645f2e38ed2e3493603db7efd8e"
TEMPERATURE = 0.7
NUM_PREDICT = 384
DEFAULT_K = 8
SHARDS = 16


def ollama_one(base_url: str, *, system: str, user: str, seed: int,
               temperature: float = TEMPERATURE, num_predict: int = NUM_PREDICT,
               timeout_s: int = 900) -> dict:
    payload = {"model": MODEL_NAME, "stream": False, "think": False,
               "keep_alive": "20m",
               "messages": [{"role":"system","content":system},
                            {"role":"user","content":user}],
               "options":{"temperature":temperature,"seed":seed,
                          "num_predict":num_predict}}
    req = Request(base_url.rstrip("/")+"/api/chat",
                  data=json.dumps(payload).encode("utf-8"),
                  headers={"Content-Type":"application/json"},method="POST")
    started=perf_counter()
    with urlopen(req,timeout=timeout_s) as response:
        result=json.loads(response.read())
    wall=round(perf_counter()-started,3)
    message=result.get("message") or {}
    text=str(message.get("content") or "").strip()
    # Protected type conventions: always produce SQL-only raw text.
    from semantic_sql.direct import _strip_fence
    return {"sql":_strip_fence(text),"wall_s":wall,
            "num_predict":num_predict,"seed":seed,
            "eval_count":int(result.get("eval_count") or 0),
            "prompt_eval_count":int(result.get("prompt_eval_count") or 0),
            "eval_duration_s":round(int(result.get("eval_duration") or 0)/1e9,4),
            "prompt_eval_duration_s":round(int(result.get("prompt_eval_duration") or 0)/1e9,4),
            "total_duration_s":round(int(result.get("total_duration") or 0)/1e9,4),
            "done_reason":str(result.get("done_reason") or "")}


def prepare_prompt(row: dict, dbroot: Path, catalog_cache: dict) -> tuple[str, str]:
    db_id=str(row["db_id"])
    if db_id not in catalog_cache:
        db=_resolve_database(dbroot,db_id)
        catalog_cache[db_id]=DatabaseCatalog.from_sqlite(db)
    schema,_=select_schema_context(catalog_cache[db_id],dbroot,db_id,str(row["question"]))
    user=f"Question: {row['question']}\nSchema context:\n{schema}"
    evidence=str(row.get("evidence") or "").strip()
    if evidence:
        user+="\nExternal evidence:\n"+evidence
    return user,sha256(user.encode()).hexdigest()


def source_shard(rows: list[dict], index: int, total: int) -> list[dict]:
    if not 0<=index<total or total<1 or total>64:
        raise ValueError("invalid predeclared source shard")
    return [row for i,row in enumerate(rows) if i%total==index]


def speed(base_url: str, source: Path, *, count: int = 8) -> dict:
    """Timing prompts already exposed in BIRD TRAIN; no DEV labels touched."""
    if file_sha256(source)!=PINNED_TRAIN_BENCH_SOURCE_SHA:
        raise ValueError("expected previously frozen real original TRAIN source prompts")
    rows=[json.loads(s) for s in source.read_text("utf-8").splitlines() if s.strip()]
    if len(rows)!=16 or not 4<=count<=16:
        raise ValueError("frozen TRAIN source requires 16 rows; sample count 4..16")
    prompts=[r["original_user"] for r in rows[:count]]
    if any(not p.startswith("Question: ") or "\nSchema context:\n" not in p for p in prompts):
        raise ValueError("source lacks authentic original user+schema prompts")
    model=_ollama_model_digest(base_url,MODEL_NAME)
    if model!=MODEL_DIGEST:
        raise ValueError("Qwen9B base digest drift")
    # Warm up one real same-model request; do not include it in latency.
    warm=ollama_one(base_url,system=_SYSTEM,user=prompts[0],
                    seed=candidate_seed("warmup",0))
    calls=[]
    for i,prompt in enumerate(prompts):
        one=ollama_one(base_url,system=_SYSTEM,user=prompt,
                       seed=candidate_seed(str(rows[i]["source_index"]),0))
        calls.append({"prompt_sha256":sha256(prompt.encode()).hexdigest(),
                      "wall_s":one["wall_s"],
                      "eval_count":one["eval_count"],
                      "prompt_eval_count":one["prompt_eval_count"],
                      "eval_duration_s":one["eval_duration_s"],
                      "prompt_eval_duration_s":one["prompt_eval_duration_s"],
                      "done_reason":one["done_reason"],
                      "sql_sha256":sha256(one["sql"].encode()).hexdigest()})
        print("REAL_SINGLE_SQL_QWEN9B_CPU",i,calls[-1]["wall_s"],
              calls[-1]["eval_count"],flush=True)
    observed=sorted(x["wall_s"] for x in calls)
    return {"status":"COMPLETE_SINGLE_GENERATION_CPU_MEASURED",
            "protocol":PROTOCOL,"model":MODEL_NAME,"model_digest":model,
            "no_Guarded_repair_calls":True,"no_gold_or_DEV_seen":True,
            "temperature":TEMPERATURE,"num_predict":NUM_PREDICT,
            "warmup_wall_s":warm["wall_s"],"n":len(calls),
            "median_wall_s":median(observed),"mean_wall_s":round(mean(observed),3),
            "p90_wall_s":observed[min(len(observed)-1,int(0.9*(len(observed)-1)))],
            "min_wall_s":observed[0],"max_wall_s":observed[-1],
            "calls":calls,
            "boundary":"Single original Qwen9B direct SQL generation measured after one warmup on eight already exposed BIRD TRAIN real schema prompts. Not Guarded generation+2 repairs, not a DEV score."}


def generate(source: Path, manifest_file: Path, dbroot: Path, base_url: str,
             *, shard: int, shards: int, k: int) -> dict:
    if k not in (8,16):
        raise ValueError("after timing freeze exact K=8 or 16; no per-task adaptive K")
    manifest=read_frozen_manifest(manifest_file)
    rows=source_shard(source_cases(source,manifest,split="dev"),shard,shards)
    digest=_ollama_model_digest(base_url,MODEL_NAME)
    if digest!=MODEL_DIGEST:
        raise ValueError("fixed original Qwen3.5-9B model digest changed")
    cache=[]
    catalogs={}
    for idx,row in enumerate(rows):
        case_id=str(row["question_id"])
        prompt,prompt_sha=prepare_prompt(row,dbroot,catalogs)
        samples=[]
        for n in range(k):
            generated=ollama_one(base_url,system=_SYSTEM,user=prompt,
                                 seed=candidate_seed(case_id,n))
            samples.append({"index":n,**generated})
        cache.append({
            "case_id":case_id,"db_id":str(row["db_id"]),
            "question_sha256":sha256(str(row["question"]).encode()).hexdigest(),
            "prompt_sha256":prompt_sha,"samples":samples,"k":k
        })
        print("QWEN9B_KCACHE_CASE",shard,idx+1,"/",len(rows),case_id,
              "K",k,"calls",len(samples),flush=True)
    return {
        "status":"CANDIDATE_CACHE_COMPLETE","protocol":PROTOCOL,"source_sha256":SOURCE_SHA256,
        "manifest_git_blob_sha1":__import__("evaluation.kcache_protocol",fromlist=["MANIFEST_GIT_BLOB_SHA1"]).MANIFEST_GIT_BLOB_SHA1,
        "model_digest":MODEL_DIGEST,"model":MODEL_NAME,
        "split":"dev","k":k,"shard":shard,"shards":shards,
        "temperature":TEMPERATURE,"num_predict":NUM_PREDICT,
        "no_gold_sql_in_cache":True,"no_repair":True,
        "cases":cache
    }


def validate_cache(cache: dict, rows: list[dict], *,
                   require_shard: bool = True) -> None:
    if (cache.get("status")!="CANDIDATE_CACHE_COMPLETE" or
        cache.get("protocol")!=PROTOCOL or
        cache.get("source_sha256")!=SOURCE_SHA256 or
        cache.get("model_digest")!=MODEL_DIGEST or
        cache.get("model")!=MODEL_NAME or
        cache.get("split")!="dev" or cache.get("no_gold_sql_in_cache") is not True or
        cache.get("no_repair") is not True or
        cache.get("k") not in (8,16)):
        raise ValueError("cache source/model/split contract drift")
    if require_shard:
        expected=source_shard(rows,cache["shard"],cache["shards"])
        if [r["case_id"] for r in cache["cases"]] != [str(r["question_id"]) for r in expected]:
            raise ValueError("cache question IDs do not match frozen shard")
    for row in cache["cases"]:
        if (row["k"]!=cache["k"] or len(row["samples"])!=cache["k"] or
            [s["index"] for s in row["samples"]]!=list(range(cache["k"]))):
            raise ValueError("missing/reordered one-time original model candidates")
        for i,s in enumerate(row["samples"]):
            if s.get("seed")!=candidate_seed(str(row["case_id"]),i):
                raise ValueError("candidate seed drift")


def score_shard(cache_file: Path, source: Path, manifest_file: Path,
                dbroot: Path) -> dict:
    m=read_frozen_manifest(manifest_file)
    rows=source_cases(source,m,split="dev")
    by_id={str(x["question_id"]):x for x in rows}
    cache=json.loads(cache_file.read_text("utf-8"))
    validate_cache(cache,rows)
    results=[]
    for c in cache["cases"]:
        original=by_id[c["case_id"]]
        db=_resolve_database(dbroot,c["db_id"])
        # Re-derive prompt identity from raw source + physical schema.
        prompt,prompt_sha=prepare_prompt(original,dbroot,{})
        if (c["db_id"]!=str(original["db_id"]) or
            c["question_sha256"]!=sha256(str(original["question"]).encode()).hexdigest() or
            c["prompt_sha256"]!=prompt_sha):
            raise ValueError("candidate cache prompt/source identity drift")
        result=per_case_offline(c,db,str(original["SQL"]))
        results.append(result)
    return {"status":"SCORED_FROM_IMMUTABLE_CACHE_ONLY","protocol":PROTOCOL,
            "source_sha256":SOURCE_SHA256,"model_digest":MODEL_DIGEST,
            "shard":cache["shard"],"shards":cache["shards"],"k":cache["k"],
            "model_calls":0,"selector_never_sees_gold":True,"cases":results}


def combine_scored(paths: list[Path], manifest_file: Path) -> dict:
    m=read_frozen_manifest(manifest_file)
    parts=[json.loads(p.read_text("utf-8")) for p in paths]
    if not parts:raise ValueError("no score shards")
    shards=parts[0].get("shards")
    k=parts[0].get("k")
    if (not isinstance(shards,int) or len(parts)!=shards or
        sorted(x.get("shard") for x in parts)!=list(range(shards)) or
        any(p.get("status")!="SCORED_FROM_IMMUTABLE_CACHE_ONLY" or
            p.get("protocol")!=PROTOCOL or
            p.get("source_sha256")!=SOURCE_SHA256 or
            p.get("model_digest")!=MODEL_DIGEST or p.get("k")!=k or
            p.get("selector_never_sees_gold") is not True or
            p.get("model_calls")!=0 for p in parts)):
        raise ValueError("not all frozen scored cache shards agree")
    rows=[row for part in parts for row in part["cases"]]
    final=aggregate_offline(rows,required_ids=set(m["dev_case_ids"]))
    return {"status":"COMPLETE_FROZEN_DEV500_KCACHE_OFFLINE_ANALYSIS",
            "manifest_git_blob_sha1":__import__("evaluation.kcache_protocol",fromlist=["MANIFEST_GIT_BLOB_SHA1"]).MANIFEST_GIT_BLOB_SHA1,
            "source_sha256":SOURCE_SHA256,"model_digest":MODEL_DIGEST,
            "results":final,"holdout300_used":False,"no_training":True,
            "holdout_is_question_only_disjoint_not_db_disjoint":True,
            "dev500_first32_previously_exposed":True}


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["speed","generate","score","aggregate"])
    p.add_argument("--source",type=Path)
    p.add_argument("--manifest",type=Path,default=Path(MANIFEST))
    p.add_argument("--database-root",type=Path)
    p.add_argument("--bench-source",type=Path)
    p.add_argument("--bench-count",type=int,default=8)
    p.add_argument("--base-url",default="http://127.0.0.1:11434")
    p.add_argument("--shard",type=int,default=0)
    p.add_argument("--shards",type=int,default=SHARDS)
    p.add_argument("--k",type=int,default=DEFAULT_K)
    p.add_argument("--cache",type=Path)
    p.add_argument("--inputs",type=Path,nargs="+")
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():
        raise FileExistsError("immutable result exists; no overwriting sampled candidates")
    if a.mode=="speed":
        if not a.bench_source:p.error("--bench-source required")
        result=speed(a.base_url,a.bench_source,count=a.bench_count)
    elif a.mode=="generate":
        if not a.source or not a.database_root:p.error("source+database-root required")
        result=generate(a.source,a.manifest,a.database_root,a.base_url,
                        shard=a.shard,shards=a.shards,k=a.k)
    elif a.mode=="score":
        if not a.source or not a.database_root or not a.cache:
            p.error("source+database-root+cache required")
        result=score_shard(a.cache,a.source,a.manifest,a.database_root)
    else:
        if not a.inputs:p.error("--inputs scores required")
        result=combine_scored(a.inputs,a.manifest)
        print("QWEN9B_FROZEN_DEV500_PASSATK_FINAL",
              json.dumps(result["results"],sort_keys=True),flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False,sort_keys=True)+"\n",
                     encoding="utf-8")
    if a.mode=="speed":
        print("QWEN9B_SINGLE_SQL_CPU_SPEED",
              json.dumps({k:v for k,v in result.items() if k!="calls"},sort_keys=True),
              flush=True)


if __name__=="__main__":
    main()
