"""BIRD TRAIN32 prospective question-frozen pilot: Guarded vs stateful LangGraph global-cardinality semantics.

Candidate selection is sealed before publisher SQL is opened. This is an
exploratory, question-enriched sample from historically exposed DB families;
not an independent or official DEV100 improvement.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import time

from evaluation.run_bird import _ollama_model_digest
from evaluation.schema_context import select_schema_context
from scripts.prepare_birdtrain_qwen9b_sft_source import (
    TRAIN_SHA, load_checked,fetch_manifest,fetch_database
)
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.direct import generate_direct_sql
from semantic_sql.execution import SQLExecutionError,execute_readonly
from semantic_sql.inference import run_guarded
from semantic_sql.providers import OllamaProvider
from semantic_sql.contracts import SQLCandidate
from experiments.bird_global_count_freeze32 import PROTOCOL as SOURCE_PROTOCOL

MODEL="qwen3.5:9b-q4_K_M"
DIGEST="56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
PROTOCOL="BIRD-QWEN9B-TRAIN32-GROUPED-GLOBAL-LANGGRAPH-PAIRED-V1"
MAX_DB_BYTES=96*1024*1024
N_SHARDS=4


class Meter:
    def __init__(self):
        self.inner=OllamaProvider(model=MODEL,timeout_s=900,temperature=0.0,
                                  seed=42,num_predict=512)
        self.calls=0
        self.model_wall_s=0.0

    def complete_text(self, *, system: str,user: str) -> str:
        start=time.monotonic()
        self.calls+=1
        try:
            return self.inner.complete_text(system=system,user=user)
        finally:
            self.model_wall_s+=time.monotonic()-start


def sources(source:Path, frozen:Path, *, inference:bool=False)->tuple[list[dict],dict]:
    m=json.loads(frozen.read_text(encoding="utf-8"))
    if inference:
        packed=json.loads(source.read_text(encoding="utf-8"))
        if (packed.get("status")!="BIRD_GLOBAL32_GOLD_FREE_QUESTION_ONLY"
            or packed.get("original_source_sha256")!=TRAIN_SHA
            or packed.get("selection_sha256")!=sha256(frozen.read_bytes()).hexdigest()
            or len(packed.get("cases",[]))!=32):
            raise ValueError("inference must use sealed NO-GOLD questions, not original BIRD publisher data")
        for row in packed["cases"]:
            if "SQL" in row or "sql" in row or "answer" in row or "gold" in row:
                raise ValueError("publisher SQL or answer leaked to Qwen job")
        max_index=max(x["source_index"] for x in packed["cases"])
        train=[{} for _ in range(max_index+1)]
        for item in packed["cases"]:
            train[item["source_index"]]=item
    else:
        train=load_checked(source,TRAIN_SHA)
    items=m.get("selection")
    if (m.get("protocol")!=SOURCE_PROTOCOL
        or m.get("original_source_sha256")!=TRAIN_SHA
        or m.get("status")!="SOURCE_ONLY_BIRD_TRAIN32_FROZEN_BEFORE_QWEN"
        or m.get("BIRD_HOLDOUT300_unused") is not True
        or m.get("last_two_protected_TRAIN_DBS_unused") is not True
        or m.get("no_Qwen_inference") is not True
        or not isinstance(items,list)
        or m.get("n") !=len(items)
        or len(items)!=32 or len({x.get("db_id") for x in items})!=m.get("n_db")):
        raise ValueError("pre-registered source-only cohort invalid")
    used=set()
    for row in items:
        ix=row.get("source_index")
        if type(ix) is not int or not 0<=ix<len(train) or ix in used:
            raise ValueError("bad frozen original BIRD source index")
        used.add(ix)
        original=train[ix]
        if original.get("db_id")!=row.get("db_id"):
            raise ValueError("physical BIRD database source changed")
        if sha256(original["question"].encode()).hexdigest()!=row.get("question_sha256"):
            raise ValueError("original publisher question identity changed")
        if sha256(str(original.get("evidence") or "").encode()).hexdigest()!=row.get("evidence_sha256"):
            raise ValueError("original publisher evidence identity changed")
        if row["db_id"] in ("simpson_episodes","works_cycles"):
            raise ValueError("protected BIRD source family")
    return train,m


def infer(source:Path,frozen:Path,physical:Path,shard:int,out:Path)->dict:
    train,m=sources(source,frozen,inference=True)
    if shard not in range(N_SHARDS):
        raise ValueError("bad shard")
    if _ollama_model_digest("http://localhost:11434",MODEL)!=DIGEST:
        raise ValueError("Qwen3.5-9B fixed digest drift")
    items=[(i,x) for i,x in enumerate(m["selection"]) if i%N_SHARDS==shard]
    metadata=fetch_manifest()
    dbids={x["db_id"] for _,x in items}
    for db in dbids:
        if db not in metadata or metadata[db]["size"]>MAX_DB_BYTES:
            raise ValueError("resource cap / pinned physical DB unavailable: "+db)
    cache={}
    records=[]
    for _,item in items:
        db=item["db_id"]
        if db not in cache:
            file=fetch_database(db,metadata[db],physical)
            cache[db]=(file,DatabaseCatalog.from_sqlite(file))
        file,catalog=cache[db]
        row=train[item["source_index"]]
        # The model sees ONLY question/evidence/schema; never publisher SQL.
        question=row["question"]
        evidence=str(row.get("evidence") or "")
        schema,_=select_schema_context(catalog,physical,db,question)
        meter=Meter()
        initial_start=time.monotonic()
        initial=generate_direct_sql(question,schema,meter,external_evidence=evidence or None)
        initial_wall_s=time.monotonic()-initial_start
        base_start=time.monotonic()
        base=run_guarded(
            database=file,provider=meter,question=question,schema_context=schema,
            evidence=evidence or None,initial_candidate=initial,
            max_repairs=2,max_rows=50000,
        )
        guarded_wall_s=time.monotonic()-base_start
        # EXACT SAME Qwen-generated original candidate and Guarded repairs.
        # Treatment uses actual native LangGraph state & conditional branches.
        treatment_sql=base.candidate.sql if base.candidate else ""
        probe_wall_s=0.0
        stage_names=[]
        stage_summaries=[]
        treatment_status=base.status
        if base.status=="ok" and base.candidate:
            start=time.monotonic()
            treated=run_guarded(
                database=file,provider=meter,question=question,
                schema_context=schema,evidence=evidence or None,
                initial_candidate=SQLCandidate(base.candidate.sql),
                max_repairs=0,max_rows=50000,grouped_global=True,
            )
            probe_wall_s=time.monotonic()-start
            treatment_sql=treated.candidate.sql if treated.candidate else treatment_sql
            treatment_status=treated.status
            stage_names=[s.name for s in treated.stages]
            stage_summaries=[s.summary for s in treated.stages
                             if s.name.startswith("grouped_")]
        if meter.calls>3:
            raise RuntimeError("Exceeded original Guarded 1+2 Qwen model-call cap")
        witness=bool(base.candidate and treatment_sql!=base.candidate.sql)
        records.append({
            "source_index":item["source_index"],"db_id":db,
            "question_sha256":item["question_sha256"],
            "original_initial_sql":initial.sql,
            "original_guarded_sql":base.candidate.sql if base.candidate else "",
            "treatment_sql":treatment_sql,
            "control_status":base.status,"treatment_status":treatment_status,
            "model_calls_shared":meter.calls,
            "model_wall_s_shared":round(meter.model_wall_s,3),
            "initial_generation_wall_s":round(initial_wall_s,3),
            "guarded_post_initial_wall_s":round(guarded_wall_s,3),
            "additional_graph_wall_s":round(probe_wall_s,3),
            "model_calls_added_by_graph":0,
            "counterexample_changed_sql":witness,
            "stage_names":stage_names,
            "stage_summaries":stage_summaries,
        })
        print("BIRD_GLOBAL32_QWEN_GOLD_FREE",shard,len(records),"/",len(items),
              db,"graph_changed",witness,"Qwen_calls",meter.calls,flush=True)
    obj={"protocol":PROTOCOL,"phase":"GOLD_BLIND_ORIGINAL_QWEN_PREDICTIONS_SEALED",
         "model_digest":DIGEST,"original_source_sha256":TRAIN_SHA,
         "selection_sha256":sha256(frozen.read_bytes()).hexdigest(),
         "shard":shard,"n":len(items),"cases":records,
         "no_gold_in_model_or_graph":True,
         "previous_database_exposure_uncertain":True,
         "HOLDOUT300_used":False,"protected_TRAIN_DBS_used":False}
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2)+"\n")
    return obj


def _result(db:Path,sql:str)->tuple[bool,dict]:
    if not sql:return False,{"status":"empty"}
    try:
        r=execute_readonly(db,sql,max_rows=50000)
        if r.truncated:return False,{"status":"truncated"}
        # Frozen set and bag compare; preserve numeric data types.
        return True,{"status":"ok",
                     "set":set(tuple(row) for row in r.rows),
                     "bag":Counter(tuple(row) for row in r.rows)}
    except (SQLExecutionError,TypeError):
        return False,{"status":"execution_failed"}


def score(source:Path,frozen:Path,physical:Path,generated:Path,shard:int,out:Path)->dict:
    train,m=sources(source,frozen)
    d=json.loads(generated.read_text(encoding="utf-8"))
    expected=[x for i,x in enumerate(m["selection"]) if i%N_SHARDS==shard]
    if (d.get("protocol")!=PROTOCOL
        or d.get("phase")!="GOLD_BLIND_ORIGINAL_QWEN_PREDICTIONS_SEALED"
        or d.get("model_digest")!=DIGEST
        or d.get("original_source_sha256")!=TRAIN_SHA
        or d.get("selection_sha256")!=sha256(frozen.read_bytes()).hexdigest()
        or d.get("shard")!=shard or d.get("n")!=len(expected)
        or len(d.get("cases",[]))!=len(expected)
        or d.get("no_gold_in_model_or_graph") is not True):
        raise ValueError("gold-blind source or Qwen cache not sealed")
    metadata=fetch_manifest()
    cache={}
    rows=[]
    for source_item,c in zip(expected,d["cases"]):
        ix=source_item["source_index"];db=source_item["db_id"]
        if (c["source_index"]!=ix or c["db_id"]!=db
                or c["question_sha256"]!=source_item["question_sha256"]):
            raise ValueError("cross-source identity mismatch")
        if db not in cache:
            cache[db]=fetch_database(db,metadata[db],physical)
        file=cache[db]
        # This is the ONLY mode that opens original publisher SQL.
        gold_ok,gold=_result(file,str(train[ix].get("SQL") or ""))
        control_ok,control=_result(file,c["original_guarded_sql"])
        treat_ok,treatment=_result(file,c["treatment_sql"])
        scorable=gold_ok
        rows.append({
            "source_index":ix,"db_id":db,"scorable":scorable,
            "control_set":bool(control_ok and control["set"]==gold["set"]) if scorable else None,
            "treatment_set":bool(treat_ok and treatment["set"]==gold["set"]) if scorable else None,
            "control_bag":bool(control_ok and control["bag"]==gold["bag"]) if scorable else None,
            "treatment_bag":bool(treat_ok and treatment["bag"]==gold["bag"]) if scorable else None,
            "model_calls_shared":c["model_calls_shared"],
            "model_wall_s_shared":c["model_wall_s_shared"],
            "additional_graph_wall_s":c["additional_graph_wall_s"],
            "graph_changed":c["counterexample_changed_sql"],
            "post_gold_canary":"labels_never_seen_in_inference",
        })
    scored={"protocol":PROTOCOL,"phase":"POST_SEAL_PUBLISHER_BIRD_STRICT_SCORE",
            "original_source_sha256":TRAIN_SHA,
            "selection_sha256":d["selection_sha256"],
            "shard":shard,"n":len(rows),"cases":rows}
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(scored,sort_keys=True,indent=2)+"\n")
    return scored


def aggregate(source:Path,frozen:Path,parts:list[Path],out:Path)->dict:
    _,m=sources(source,frozen)
    if len(parts)!=N_SHARDS:raise ValueError("all 4 scored shards required")
    seen=set();rows=[]
    for path in parts:
        d=json.loads(path.read_text())
        i=d.get("shard")
        if (not isinstance(i,int) or i in seen or i not in range(N_SHARDS)
            or d.get("protocol")!=PROTOCOL
            or d.get("phase")!="POST_SEAL_PUBLISHER_BIRD_STRICT_SCORE"
            or d.get("original_source_sha256")!=TRAIN_SHA):
            raise ValueError("invalid score shard")
        seen.add(i);rows+=d["cases"]
    if len(rows)!=32 or len({x["source_index"] for x in rows})!=32:
        raise ValueError("not the full frozen BIRD32")
    if {x["source_index"] for x in rows}!={x["source_index"] for x in m["selection"]}:
        raise ValueError("BIRD32 frozen question IDs mismatched")
    good=[x for x in rows if x["scorable"]]
    r=Counter()
    for x in good:
        a,b=x["control_set"],x["treatment_set"]
        r["control_set"]+=int(a);r["treatment_set"]+=int(b)
        r["control_bag"]+=int(x["control_bag"])
        r["treatment_bag"]+=int(x["treatment_bag"])
        r["rescues"]+=int(b and not a);r["regressions"]+=int(a and not b)
    changed=sum(x["graph_changed"] for x in rows)
    r["model_calls_shared"]=sum(x["model_calls_shared"] for x in rows)
    r["graph_additional_model_calls"]=0
    obj={
        "protocol":PROTOCOL,"status":"COMPLETE_BIRD_ORIGINAL_QWEN_GLOBAL_CARDINALITY_PAIRED_TRAIN32",
        "n":32,"scorable":len(good),"database_count":len({x["db_id"] for x in rows}),
        "counts":dict(r),"model_wall_shared_s":round(sum(x["model_wall_s_shared"] for x in rows),2),
        "graph_extra_wall_s":round(sum(x["additional_graph_wall_s"] for x in rows),2),
        "changed_sql_n":changed,
        "selected_net_SET_EX":r["rescues"]-r["regressions"],
        "positive_screen_only":bool(len(good)>=28 and r["rescues"]-r["regressions"]>=2
                                     and r["regressions"]==0),
        "promotion_allowed":False,
        "selection_pre_gold":True,
        "previously_exposed_DB_families":True,
        "not_independent_or_historical_DEV100":True,
        "protected_BIRD_HOLDOUT300_unused":True,"protected_two_TRAIN_DBS_unused":True,
        "token_and_USD_cost_measured":False,
        "warning":"Source-only BIRD32 selected before model; different questions from recent BIRD19/11, but database families have historical exposure. Not official BIRD DEV100 or independent transfer.",
    }
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")
    return obj


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("infer","score","aggregate"))
    for k in ("source","frozen","physical","generated","out"):
        p.add_argument("--"+k,type=Path)
    p.add_argument("--shard",type=int,default=0)
    p.add_argument("--parts",nargs="+",type=Path)
    a=p.parse_args()
    if a.mode=="infer":d=infer(a.source,a.frozen,a.physical,a.shard,a.out)
    elif a.mode=="score":d=score(a.source,a.frozen,a.physical,a.generated,a.shard,a.out)
    else:d=aggregate(a.source,a.frozen,a.parts,a.out)
    print("BIRD_ORIGINAL_QWEN_GLOBAL_CARDINALITY_PROOF",json.dumps(
        {k:v for k,v in d.items() if k!="cases"},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
