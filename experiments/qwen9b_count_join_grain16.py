"""Pre-registered no-train 16-case BIRD TRAIN Qwen9B COUNT(JOIN) grain correction.

Four fresh physical source databases; excludes 44 prior BIRD TRAIN DBs and
all publisher DEV DBs. Source cases selected by original question containing
"how many", not by Qwen outputs or reference query features. Same unchanged
original Qwen3.5-9B and guarded 2-repair baseline. Treatment is PURE SQLite
deterministic, zero additional model calls; gold never used to choose rewrite.
Publisher source-local strict set/bag EX is not hidden or human-verified.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
from time import perf_counter

from evaluation.run_bird import _ollama_model_digest
from evaluation.schema_context import select_schema_context
from scripts.prepare_birdtrain_qwen9b_sft_source import (
    TRAIN_SHA,DEV_SHA,load_checked,fetch_manifest,fetch_database,normalized,
    validate_gold_sql)
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.contracts import SQLCandidate
from semantic_sql.direct import _SYSTEM,_strip_fence
from semantic_sql.execution import execute_readonly,SQLExecutionError
from semantic_sql.grain_guard import propose_join_entity_grain
from semantic_sql.inference import run_guarded
from semantic_sql.providers import OllamaProvider

MODEL="qwen3.5:9b-q4_K_M"
MODEL_DIGEST="56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
PROTOCOL="BIRD-TRAIN-SOURCE-DB45PLUS-QWEN9B-COUNT-JOIN-GRAIN16-V1"
SEED="birdtrain-count-join-question-conditioned-v1-20261117"
N_DB=4
PER_DB=4
N=N_DB*PER_DB
SHARDS=4
COUNT_Q=re.compile(r"\bhow\s+many\b",re.I)
PRIOR_SHA={
    "train":"b6851fb0f2b5027833aff45a0b5e01326359b245be1c812edc66986d0df1d402",
    "validation":"cc112305bc9de33a03cea704f2edecfdfb45e62cc953ae498cea27fbed93b506",
    "fk32":"ee1b5035302d36a1241952bd2e5acd3dfc2518d1fba2581f557e04e179960535",
    "value16":"4de8471341b38d0fb00ded1f17f265d33b528ec78b9779eb94937914df196955",
    "value16_replica":"faa4f7ba1de9923cbf8e51a500727dfe4c76a08c3f6366735cf21bff152655b7",
    "role8":"3be2fc49ba3cff4171e20afcaf70f1fc698c8be401ac332a29fa1f48b781168f",
    "preview8":"85c3e3abc45d90f23418dbc14261111abf90feadfef2b155988b77b40a4fe0ca",
    "column8":"e7e822dbefadeb57dd2ba5a79553d38f5b5148a4fab33b9c0563b2e624ca6d74"
}
PRIOR_N={"train":242,"validation":64,"fk32":32,"value16":16,
         "value16_replica":16,"role8":8,"preview8":8,"column8":8}
GATE={"min_net_set_ex":2,"max_regressions":0,"min_distinct_rescue_dbs":2,
      "min_actual_grain_corrections":2,"nondecrease_bag_ex":True}
MAX_DATABASE_BYTES=96*1024*1024

def stable_sha(records):
    return sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()

def hash_order(which,item):
    return sha256(f"{SEED}:{which}:{item}".encode()).hexdigest()

def prior_exclusions(prior,dev):
    if set(prior)!=set(PRIOR_SHA):
        raise ValueError("require all eight original source manifests")
    exclude={row["db_id"] for row in dev}
    dev_db=set(exclude)
    seen={normalized(row["question"]) for row in dev}
    counts={}
    for label,path in sorted(prior.items()):
        raw=path.read_bytes()
        if sha256(raw).hexdigest()!=PRIOR_SHA[label]:
            raise ValueError(f"prior physical source SHA changed: {label}")
        rows=[json.loads(x) for x in raw.decode().splitlines() if x.strip()]
        if len(rows)!=PRIOR_N[label]:
            raise ValueError(f"prior source count changed: {label}")
        field="source_db_id" if label in ("train","validation") else "db_id"
        local_dbs={row[field] for row in rows}
        if not local_dbs:
            raise ValueError("prior source without DB identities")
        counts[label]=len(local_dbs)
        exclude.update(local_dbs)
        for row in rows:
            if label in ("train","validation"):
                prompt=row["messages"][1]["content"]
                seen.add(normalized(prompt.split("\nSchema context:\n",1)[0].removeprefix("Question: ")))
            else:
                seen.add(normalized(row["question"]))
    if len(exclude-dev_db)!=44:
        raise ValueError(f"expected 44 previously scored TRAIN DBs, found {len(exclude-dev_db)}")
    return exclude,seen,counts

def freeze(train_file,dev_file,prior,work,out):
    train=load_checked(train_file,TRAIN_SHA)
    dev=load_checked(dev_file,DEV_SHA)
    excluded,seen,counts=prior_exclusions(prior,dev)
    manifest=fetch_manifest()
    grouped=defaultdict(list)
    for ix,row in enumerate(train):
        db=row.get("db_id")
        q=normalized(row.get("question") or "")
        if db not in manifest or db in excluded or not COUNT_Q.search(q) or q in seen:
            continue
        if manifest[db]["size"]>MAX_DATABASE_BYTES:continue
        seen.add(q)
        grouped[db].append((ix,row))
    ordered=sorted((db for db,v in grouped.items() if len(v)>=PER_DB),
                   key=lambda db:hash_order("database",db))
    accepted=[]
    source_dbs=[]
    for dbid in ordered:
        if len(source_dbs)==N_DB:break
        db=fetch_database(dbid,manifest[dbid],work)
        catalog=DatabaseCatalog.from_sqlite(db)
        rows=[]
        for ix,r in sorted(grouped[dbid],key=lambda item:hash_order("question",item[0]))[:60]:
            question=str(r["question"]).strip()
            gold_sql=str(r.get("SQL") or "")
            if validate_gold_sql(db,gold_sql) is not None:continue
            schema,_=select_schema_context(catalog,work,dbid,question)
            evidence=str(r.get("evidence") or "").strip()
            prompt=f"Question: {question}\nSchema context:\n{schema}"
            if evidence:
                prompt+=f"\nExternal evidence:\n{evidence}"
            if len(prompt)>16000:continue
            rows.append({"source_index":ix,"db_id":dbid,"question":question,
                         "gold_sql":gold_sql,"original_user":prompt,
                         "prompt_sha256":sha256(prompt.encode()).hexdigest()})
            if len(rows)==PER_DB:break
        if len(rows)==PER_DB:
            accepted.extend(rows)
            source_dbs.append(dbid)
    if len(accepted)!=N or len(source_dbs)!=N_DB:
        raise RuntimeError(f"not enough new eligible BIRD TRAIN source DBs: {source_dbs}, questions={len(accepted)}")
    if set(source_dbs)&excluded or len({x["source_index"] for x in accepted})!=N:
        raise ValueError("source reused prior DB or row")
    out.mkdir(parents=True,exist_ok=True)
    filename=out/"grain16.jsonl"
    filename.write_text("".join(json.dumps(x,ensure_ascii=False,separators=(",",":"))+"\n" for x in accepted))
    m={
        "status":"FROZEN_NO_MODEL_INFERENCE","protocol":PROTOCOL,
        "n":N,"shards":SHARDS,"physical_dbs":N_DB,
        "source_sha256":sha256(filename.read_bytes()).hexdigest(),
        "source_databases":source_dbs,
        "prior_source_sha256":PRIOR_SHA,"prior_db_counts":counts,
        "excluded_prior_train_db_count":44,"excluded_dev_db_count":len({x["db_id"] for x in dev}),
        "question_selection_only_asks_how_many":True,
        "no_selection_using_gold_execution_outcomes":True,
        "reference_sql_compiler_verified_only":True,
        "model_digest":MODEL_DIGEST,"gate":GATE,
        "no_qlora":True,"no_training":True,"not_hidden_or_dev_score":True,
        "public_corpus_and_historical_exposure_unknown":True
    }
    (out/"freeze.json").write_text(json.dumps(m,indent=2,sort_keys=True)+"\n")
    return m

def load_frozen(cohort,meta):
    m=json.loads(meta.read_text())
    raw=cohort.read_bytes()
    if (m.get("status")!="FROZEN_NO_MODEL_INFERENCE" or m.get("protocol")!=PROTOCOL
        or m.get("n")!=N or m.get("shards")!=SHARDS
        or m.get("excluded_prior_train_db_count")!=44
        or m.get("model_digest")!=MODEL_DIGEST or m.get("gate")!=GATE
        or m.get("source_sha256")!=sha256(raw).hexdigest()):
        raise ValueError("pre-inference SHA/model/source contract failed")
    rows=[json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if len(rows)!=N or len({x["source_index"] for x in rows})!=N:
        raise ValueError("question identities missing or duplicate")
    if len({x["db_id"] for x in rows})!=N_DB or len(m["source_databases"])!=N_DB:
        raise ValueError("incorrect new physical database count")
    if [rows[PER_DB*i]["db_id"] for i in range(N_DB)]!=m["source_databases"]:
        raise ValueError("physical database source order changed")
    for i,row in enumerate(rows):
        if (row["db_id"]!=m["source_databases"][i//PER_DB]
            or not COUNT_Q.search(row["question"])
            or not row["original_user"].startswith("Question: "+row["question"]+"\nSchema context:\n")
            or sha256(row["original_user"].encode()).hexdigest()!=row["prompt_sha256"]):
            raise ValueError("original publisher question source drift")
    return rows

def score(db,sql,gold_set,gold_bag):
    if not sql:
        return {"set_ex":False,"bag_ex":False,"executable":False}
    try:
        result=execute_readonly(db,sql,max_rows=50000)
        if result.truncated:
            return {"set_ex":False,"bag_ex":False,"executable":False}
        return {"set_ex":set(result.rows)==gold_set,
                "bag_ex":Counter(result.rows)==gold_bag,"executable":True}
    except (SQLExecutionError,ValueError):
        return {"set_ex":False,"bag_ex":False,"executable":False}

def one_case(db,row,provider):
    question=row["question"]
    prompt=row["original_user"]
    schema_and_ev=prompt[10:].split("\nSchema context:\n",1)[1]
    if "\nExternal evidence:\n" in schema_and_ev:
        schema,evidence=schema_and_ev.split("\nExternal evidence:\n",1)
    else:
        schema,evidence=schema_and_ev,None
    gold=execute_readonly(db,row["gold_sql"],max_rows=50000)
    if gold.truncated:raise ValueError("publisher gold result truncated")
    gold_set,gold_bag=set(gold.rows),Counter(gold.rows)
    start=perf_counter()
    initial=_strip_fence(provider.complete_text(system=_SYSTEM,user=prompt))
    guarded=run_guarded(database=db,provider=provider,question=question,
                        schema_context=schema,evidence=evidence,
                        initial_candidate=SQLCandidate(sql=initial,attempt=0),
                        max_repairs=2,max_rows=50000)
    sql=guarded.candidate.sql if guarded.candidate else ""
    baseline=score(db,sql,gold_set,gold_bag)
    llm_elapsed=round(perf_counter()-start,2)
    proposed=sql
    changed=False
    reason="base_not_executable"
    original_count=None; distinct_count=None
    begin=perf_counter()
    if baseline["executable"]:
        candidate=propose_join_entity_grain(db,sql,question)
        proposed=candidate.sql
        changed=candidate.changed
        reason=candidate.reason
        original_count=candidate.original_count
        distinct_count=candidate.distinct_count
    alternative=score(db,proposed,gold_set,gold_bag)
    guard_elapsed=round(perf_counter()-begin,3)
    return {
        "source_index":row["source_index"],"db_id":row["db_id"],
        "baseline_sql":sql,"treatment_sql":proposed,
        "baseline":baseline,"grain_guard":alternative,
        "grain_changed":changed,"grain_reason":reason,
        "raw_join_row_count":original_count,"distinct_parent_row_count":distinct_count,
        "baseline_elapsed_s":llm_elapsed,"deterministic_guard_s":guard_elapsed,
        "new_model_calls":0,"question":question,
        "question_sha256":row["prompt_sha256"]
    }

def run_shard(cohort,meta,shard,work):
    if shard not in range(SHARDS):raise ValueError("source shard must be [0,3]")
    cases=load_frozen(cohort,meta)[shard*PER_DB:(shard+1)*PER_DB]
    if len(cases)!=PER_DB or len({x["db_id"] for x in cases})!=1:
        raise ValueError("shard is not 4 questions from one DB")
    if _ollama_model_digest("http://127.0.0.1:11434",MODEL)!=MODEL_DIGEST:
        raise RuntimeError("fixed original Qwen model digest changed")
    manifest=fetch_manifest()
    db=fetch_database(cases[0]["db_id"],manifest[cases[0]["db_id"]],work)
    provider=OllamaProvider(model=MODEL,temperature=0.0,seed=42,num_predict=768,timeout_s=900)
    output=[]
    for row in cases:
        result=one_case(db,row,provider)
        output.append(result)
        print("BIRD_TRAIN_GRAIN16",shard,row["source_index"],
              result["baseline"]["set_ex"],result["grain_guard"]["set_ex"],
              result["grain_reason"],flush=True)
    return {"status":"PHYSICAL_4_CASES_COMPLETE","protocol":PROTOCOL,
            "shard":shard,"db_id":cases[0]["db_id"],"n":PER_DB,
            "source_sha256":sha256(cohort.read_bytes()).hexdigest(),
            "model_digest":MODEL_DIGEST,"seed":42,"temperature":0.0,"max_repairs":2,
            "new_training_steps":0,"cases":output}

def aggregate(paths):
    if len(paths)!=SHARDS:
        raise ValueError("all four physical Qwen shards required")
    shards=[json.loads(path.read_text()) for path in paths]
    if sorted(s.get("shard") for s in shards)!=list(range(SHARDS)):
        raise ValueError("duplicate/missing physical Qwen shard")
    if any(s.get("status")!="PHYSICAL_4_CASES_COMPLETE" or
           s.get("protocol")!=PROTOCOL or s.get("model_digest")!=MODEL_DIGEST
           or s.get("n")!=PER_DB or len(s.get("cases",[]))!=PER_DB
           or s.get("max_repairs")!=2 or s.get("new_training_steps")!=0 for s in shards):
        raise ValueError("model/source/protocol changed")
    if len({s["source_sha256"] for s in shards})!=1:
        raise ValueError("physical source SHA mismatch")
    rows=[r for s in sorted(shards,key=lambda x:x["shard"]) for r in s["cases"]]
    if len(rows)!=N or len({r["source_index"] for r in rows})!=N or len({r["db_id"] for r in rows})!=N_DB:
        raise ValueError("missing independent source IDs/DBs")
    rescues=[r for r in rows if not r["baseline"]["set_ex"] and r["grain_guard"]["set_ex"]]
    regressions=[r for r in rows if r["baseline"]["set_ex"] and not r["grain_guard"]["set_ex"]]
    totals={name:{
        "strict_set_ex":sum(r[name]["set_ex"] for r in rows),
        "strict_bag_ex":sum(r[name]["bag_ex"] for r in rows),
        "executable":sum(r[name]["executable"] for r in rows)}
        for name in ("baseline","grain_guard")}
    triggered=sum(r["grain_changed"] for r in rows)
    passed=(len(rescues)-len(regressions)>=GATE["min_net_set_ex"]
            and len(regressions)<=GATE["max_regressions"]
            and len({r["db_id"] for r in rescues})>=GATE["min_distinct_rescue_dbs"]
            and triggered>=GATE["min_actual_grain_corrections"]
            and totals["grain_guard"]["strict_bag_ex"]>=totals["baseline"]["strict_bag_ex"])
    return {
        "status":"COMPLETE_PUBLIC_SOURCE_DIAGNOSTIC_NO_AUTOPROMOTION",
        "protocol":PROTOCOL,"n":N,"physical_dbs":N_DB,
        "source_sha256":shards[0]["source_sha256"],
        "model_digest":MODEL_DIGEST,"totals":totals,
        "changed_grain_cases":triggered,
        "reason_counts":dict(Counter(r["grain_reason"] for r in rows)),
        "rescues":len(rescues),"regressions":len(regressions),
        "rescue_distinct_physical_dbs":len({r["db_id"] for r in rescues}),
        "new_model_calls":0,"actual_training_steps":0,
        "positive_screen_only":passed,"promotion_allowed":False,"gate":GATE,
        "claim_boundary":"Zero extra Qwen calls. Source-local strict set/bag EX on new publisher TRAIN DBs, not validated semantic oracle or official hidden DEV. No model swap or SQL QLoRA.",
        "cases":rows
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-only",action="store_true")
    p.add_argument("--train",type=Path)
    p.add_argument("--dev",type=Path)
    for label in PRIOR_SHA:
        p.add_argument("--prior-"+label.replace("_","-"),type=Path)
    p.add_argument("--cohort",type=Path)
    p.add_argument("--meta",type=Path)
    p.add_argument("--shard",type=int)
    p.add_argument("--inputs",type=Path,nargs="+")
    p.add_argument("--work",type=Path,default=Path("grain16-physical-sqlite"))
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError("immutable output exists")
    if a.source_only:
        prior={k:getattr(a,"prior_"+k) for k in PRIOR_SHA}
        if any(x is None for x in (a.train,a.dev,*prior.values())):
            p.error("all BIRD TRAIN/DEV and eight prior physical source cohorts required")
        result=freeze(a.train,a.dev,prior,a.work,a.out)
        print("GRAIN16_SOURCE_FROZEN",json.dumps(result,sort_keys=True),flush=True)
    elif a.inputs:
        result=aggregate(a.inputs)
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
        print("GRAIN16_PHYSICAL_AGGREGATE",json.dumps({k:v for k,v in result.items() if k!="cases"},sort_keys=True),flush=True)
    else:
        if a.cohort is None or a.meta is None or a.shard is None:
            p.error("cohort/meta/shard required")
        result=run_shard(a.cohort,a.meta,a.shard,a.work)
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")

if __name__=="__main__":
    main()
