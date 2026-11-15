"""Qwen9B native BATCH tool protocol with real physical SQLite vs Guarded.

The first trial had an artificial "multiple_tool_calls_unsupported" rejection
on four of the first eight questions. Correct the general Ollama tool protocol
and freeze NEW question IDs before inference. No gold in selection or calls.
Public BIRD DEV physical DB families may overlap; this is NOT hidden OOD.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256,file_digest
import json
import re
from pathlib import Path
from time import monotonic

from evaluation.qwen_grain_bird20_transfer import (
    BIRD_SOURCE_SHA,freeze as freeze_bird20,PRIOR16,PRIOR24
)
from evaluation.qwen_observed_sql_tools import (
    run_tool_agent,MODEL,DIGEST,MAX_STEPS,MAX_TOOLS
)
from evaluation.run_bird import CountingProvider,_resolve_database,_ollama_model_digest
from evaluation.schema_context import select_schema_context
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.contracts import SQLCandidate
from semantic_sql.execution import SQLExecutionError,execute_readonly
from semantic_sql.inference import run_guarded
from semantic_sql.providers import OllamaProvider

PROTOCOL="qwen9b-native-batched-tool-observation-versus-guarded-bird-dev12-v2"
N=12
DBS=6
SHARDS=3
MAX_ROWS=50_000


SEED="native-batched-tool-protocol-unseen-bird12-20261116-v1"


def freeze_fresh(source:Path,original:Path)->dict:
    """Select only new question IDs, with 1 analytic and 1 other in 6 DBs."""
    if sha256(source.read_bytes()).hexdigest()!=BIRD_SOURCE_SHA:
        raise ValueError("original publisher BIRD DEV source SHA mismatch")
    original100=json.loads(original.read_text())
    if (original100.get("status")!="complete"
        or original100.get("sample_size")!=100
        or len(original100.get("cases",[]))!=100):
        raise ValueError("historical BIRD original100 unavailable")
    first=freeze_bird20(source,original)
    excludes={str(x["case_id"]) for x in original100["cases"]}
    excludes |= PRIOR16|PRIOR24|{x["case_id"] for x in first["cases"]}
    if len(excludes)!=160:
        raise ValueError("prior question-exclusion cohort drift")
    data=json.loads(source.read_text(encoding="utf-8"))
    if len(data)!=1534:
        raise ValueError("BIRD source length unexpected")
    groups={}
    for i,row in enumerate(data):
        cid=str(row.get("question_id") or row.get("id") or i)
        question=" ".join(str(row.get("question") or "").split())
        db=str(row.get("db_id") or "")
        if not question or cid in excludes or not db:
            continue
        # Question-only comparison, not reference SQL/operators or old EX.
        kind="analytic" if re.search(
            r"\b(?:how many|number of|count|average|mean|sum|total|most|least|"
            r"per|each|highest|lowest|top|percentage|ratio)\b",question,re.I
        ) else "other"
        grouped=groups.setdefault(db,{"analytic":{},"other":{}})
        grouped[kind].setdefault(question.casefold(),{
            "case_id":cid,"database_id":db,"question":question,
            "evidence":str(row.get("evidence") or "")
        })
    rank=lambda value:sha256((SEED+":"+value).encode()).hexdigest()
    eligible=sorted(
        (db for db,bykind in groups.items()
         if bykind["analytic"] and bykind["other"]),
        key=lambda db:rank("db:"+db)
    )
    selected=[]
    for db in eligible[:DBS]:
        for kind in ("analytic","other"):
            candidates=sorted(groups[db][kind].values(),key=lambda x:rank("q:"+x["case_id"]))
            selected.append(candidates[0])
    if len(selected)!=N or len({c["database_id"] for c in selected})!=DBS:
        raise ValueError("new source-diverse question-only BIRD cohort not feasible")
    if len({x["case_id"] for x in selected})!=N:
        raise ValueError("duplicate BIRD new cases")
    cases_sha=sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest()
    return {
        "protocol":PROTOCOL,"source_sha":BIRD_SOURCE_SHA,
        "excluded_original100_previous16_24_and_grain20":len(excludes),
        "prior_bird20_sha":first["selected_cohort_sha256"],
        "cases_sha":cases_sha,
        "model":MODEL,"digest":DIGEST,
        "n":N,"db_count":DBS,
        "gold_used_to_select":False,
        "independent_question_ids_from_exposed_160":True,
        "source_note":"BIRD DEV public, 12 new IDs across 6 DB; DB and model-pretraining overlap possible",
        "cases":selected,
    }


def source_manifest(frozen:Path,source:Path,original:Path)->dict:
    observed=json.loads(frozen.read_text())
    expected=freeze_fresh(source,original)
    if observed!=expected:
        raise ValueError("frozen new BIRD12 cohort or selector changed after source freeze")
    return observed


def _model()->CountingProvider:
    return CountingProvider(OllamaProvider(
        model=MODEL,temperature=0,seed=42,
        num_predict=700,timeout_s=900
    ))


def _result(res,model_calls:int,sec:float)->dict:
    good=res.status=="ok" and not res.truncated
    return {
        "sql":res.candidate.sql if res.candidate else None,
        "status":res.status,"execution_success":bool(good),
        "truncated":res.truncated,"_rows":res.rows if good else (),
        "model_calls":model_calls,"elapsed_s":round(sec,2),
        "repairs":res.candidate.attempt if res.candidate else -1
    }


def run_shard(source:Path,original:Path,frozen:Path,root:Path,shard:int)->dict:
    if shard not in range(SHARDS):
        raise ValueError("bad source-pinned physical shard")
    manifest=source_manifest(frozen,source,original)
    if _ollama_model_digest("http://127.0.0.1:11434",MODEL)!=DIGEST:
        raise ValueError("original Qwen3.5-9B checkpoint checksum drift")
    original_rows=json.loads(source.read_text())
    identity={str(r.get("question_id") or r.get("id") or i):r
              for i,r in enumerate(original_rows)}
    selected=manifest["cases"][4*shard:4*(shard+1)]
    if len(selected)!=4 or len({r["database_id"] for r in selected})!=2:
        raise ValueError("BIRD 4-case shard source drift")
    cache={}
    records=[]
    for row in selected:
        case_id=row["case_id"];db_id=row["database_id"]
        original_row=identity[case_id]
        if original_row["db_id"]!=db_id or " ".join(original_row["question"].split())!=row["question"]:
            raise ValueError("physical source question identity mismatch")
        if db_id not in cache:
            p=_resolve_database(root,db_id)
            cache[db_id]=(p,DatabaseCatalog.from_sqlite(p))
        db,cat=cache[db_id]
        schema,_=select_schema_context(cat,root,db_id,row["question"])
        evidence=row["evidence"] or None
        model=_model()
        start=monotonic()
        control=run_guarded(
            database=db,provider=model,question=row["question"],
            schema_context=schema,evidence=evidence,
            max_repairs=2,max_rows=MAX_ROWS
        )
        base=_result(control,model.total_calls,monotonic()-start)
        start=monotonic()
        actual=run_tool_agent(row["question"],schema,db,evidence)
        if actual.sql is not None:
            provider=_model()
            revised=run_guarded(
                database=db,provider=provider,question=row["question"],
                schema_context=schema,evidence=evidence,
                initial_candidate=SQLCandidate(sql=actual.sql,attempt=0),
                max_repairs=2,max_rows=MAX_ROWS
            )
            new=_result(revised,actual.model_calls+provider.total_calls,monotonic()-start)
        else:
            new={
                "sql":None,"status":"agent_no_submission",
                "execution_success":False,"truncated":False,
                "_rows":(),"model_calls":actual.model_calls,
                "elapsed_s":round(monotonic()-start,2),"repairs":0
            }
        new["actual_tool_calls"]=actual.calls
        new["actual_schema_inspects"]=actual.inspect_calls
        new["actual_row_samples"]=actual.sample_calls
        new["actual_sql_trials"]=actual.trial_calls
        new["actual_tool_errors"]=actual.tool_errors
        new["tool_agent_stop_reason"]=actual.outcome
        with db.open("rb") as file:
            dbhash=file_digest(file,"sha256").hexdigest()
        records.append({
            **row,"schema_sha256":sha256(schema.encode()).hexdigest(),
            "db_sha256":dbhash,
            "arms":{"original":base,"observed_tools":new},
        })
        print("QWEN_OBSERVED_SQL_AGENT_GENERATED",shard,case_id,
              "calls",base["model_calls"],new["model_calls"],
              "tool_calls",actual.calls,"stopped",actual.outcome,flush=True)
    # Reference is NEVER read until every arm in the whole shard is generated.
    for record in records:
        row=identity[record["case_id"]]
        db=cache[record["database_id"]][0]
        try:
            gold=execute_readonly(db,str(row["SQL"]),max_rows=MAX_ROWS)
            valid=not gold.truncated
        except (SQLExecutionError,ValueError):
            gold,valid=None,False
        record["scorable"]=valid
        for name in ("original","observed_tools"):
            pred=record["arms"][name]
            rows=pred.pop("_rows")
            okay=valid and pred["execution_success"]
            pred["set_ex"]=bool(okay and set(map(tuple,rows))==set(map(tuple,gold.rows))) if valid else None
            pred["bag_ex"]=bool(okay and Counter(map(tuple,rows))==Counter(map(tuple,gold.rows))) if valid else None
    return {
        "protocol":PROTOCOL,"model":MODEL,"digest":DIGEST,
        "cohort_sha":manifest["cases_sha"],
        "source_sha":BIRD_SOURCE_SHA,"shard":shard,
        "cases":records,"sql_training":False,"goldblind_generation":True,
    }


def aggregate(parts:list[dict])->dict:
    if (len(parts)!=SHARDS
        or sorted(p["shard"] for p in parts)!=list(range(SHARDS))
        or any(p["protocol"]!=PROTOCOL or p["digest"]!=DIGEST
               or p["source_sha"]!=BIRD_SOURCE_SHA or
               len(p["cases"])!=N//SHARDS or not p["goldblind_generation"]
               for p in parts)):
        raise ValueError("incomplete or invalid physical Qwen tool-agent trial")
    if len({p["cohort_sha"] for p in parts})!=1:
        raise ValueError("original frozen manifest drift")
    rows=[r for p in sorted(parts,key=lambda x:x["shard"]) for r in p["cases"]]
    if len(rows)!=N or len({r["case_id"] for r in rows})!=N:
        raise ValueError("missing/repeated case IDs")
    valid=[r for r in rows if r["scorable"]]
    names=("original","observed_tools")
    measures={}
    for name in names:
        measures[name]={
            "strict_set_ex":sum(r["arms"][name]["set_ex"] for r in valid),
            "strict_bag_ex":sum(r["arms"][name]["bag_ex"] for r in valid),
            "executed":sum(r["arms"][name]["execution_success"] for r in rows),
            "model_calls":sum(r["arms"][name]["model_calls"] for r in rows),
            "latency_s":round(sum(r["arms"][name]["elapsed_s"] for r in rows),1),
        }
    rescues=[r for r in valid if not r["arms"]["original"]["set_ex"] and r["arms"]["observed_tools"]["set_ex"]]
    regress=[r for r in valid if r["arms"]["original"]["set_ex"] and not r["arms"]["observed_tools"]["set_ex"]]
    tools={
        name:sum(r["arms"]["observed_tools"][name] for r in rows)
        for name in ("actual_tool_calls","actual_schema_inspects","actual_row_samples",
                     "actual_sql_trials","actual_tool_errors")
    }
    candidate_positive=(
        len(valid)>=11 and len(rescues)-len(regress)>=2
        and not regress and len({r["database_id"] for r in rescues})>=2
        and measures["observed_tools"]["strict_bag_ex"]>=measures["original"]["strict_bag_ex"]
    )
    return {
        "protocol":PROTOCOL,"n":N,"scorable":len(valid),"dbs":DBS,
        "arms":measures,"tool_calls":tools,
        "rescues":[r["case_id"] for r in rescues],
        "regressions":[r["case_id"] for r in regress],
        "stop_reasons":dict(Counter(r["arms"]["observed_tools"]["tool_agent_stop_reason"] for r in rows)),
        "positive_source_screen_only":candidate_positive,
        "promotion_allowed":False,"training":False,
        "BIRD100_Qwen_Guarded_EX_unchanged":39,
        "independence_limit":"NEW disjoint BIRD DEV questions, some physical DB and model-pretraining overlap possible; not hidden source-independent",
        "cases":rows
    }


def main():
    a=argparse.ArgumentParser()
    for key in ("source","original","frozen","database-root","output"):
        a.add_argument("--"+key,type=Path)
    a.add_argument("--shard",type=int)
    a.add_argument("--inputs",type=Path,nargs="+")
    args=a.parse_args()
    if args.output is None or args.output.exists():
        raise FileExistsError("refuse overwrite")
    if args.inputs:
        report=aggregate([json.loads(p.read_text()) for p in args.inputs])
        print("QWEN_BATCHED_TOOLS_FRESH_BIRD12_FINAL",
              json.dumps({k:v for k,v in report.items() if k!="cases"},sort_keys=True),
              flush=True)
    elif args.frozen is None and args.shard is None:
        if args.source is None or args.original is None:
            a.error("source/original required to freeze unseen question-only cohort")
        report=freeze_fresh(args.source,args.original)
        print("QWEN_BATCHED_TOOLS_FRESH_COHORT_FROZEN",
              report["cases_sha"],flush=True)
    else:
        if any(v is None for v in (args.source,args.original,args.frozen,args.database_root,args.shard)):
            a.error("paired source/original/frozen/database-root/shard required")
        report=run_shard(args.source,args.original,args.frozen,args.database_root,args.shard)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")


if __name__=="__main__":
    main()
