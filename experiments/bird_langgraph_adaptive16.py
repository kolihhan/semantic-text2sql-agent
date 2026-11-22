"""Matched BIRD TRAIN16 replay of existing Qwen candidate into opt-in LangGraph.

Primary SQL is reused from sealed original Qwen K4 control, not regenerated.
Both arms receive EXACT SAME original SQL; the adaptive branch can spend up to
one additional fixed Qwen3.5-9B call. Publisher ground truth is loaded ONLY in
score mode, after candidate SQL/state traces are sealed in the inference job.
This is an already-exposed BIRD TRAIN pilot, NOT independent promotion evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from evaluation.run_bird import _ollama_model_digest
from evaluation.schema_context import select_schema_context
from experiments.bird_qwen9b_physical_answer_contract16 import (
    PROTOCOL as SOURCE_PROTOCOL, N_DB, PER_DB, N, DIGEST, MODEL, TRAIN_SHA,
    checked_source, execute
)
from scripts.prepare_birdtrain_qwen9b_sft_source import load_checked
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.contracts import SQLCandidate
from semantic_sql.inference import run_guarded
from semantic_sql.providers import OllamaProvider

PROTOCOL = "BIRD-TRAIN16-MATCHED-LANGGRAPH-CONDITIONAL-PHYSICAL-V1"
SEALED_SOURCE_STATUS = "ORIGINAL_QWEN9B_CANDIDATES_AND_CHOICES_SEALED_PRE_GOLD"


class MeterProvider:
    def __init__(self):
        self.base = OllamaProvider(model=MODEL, timeout_s=900, temperature=0.7,
                                   seed=42, num_predict=384)
        self.calls = 0
        self.wall_s = 0.0

    def complete_text(self, *, system: str, user: str) -> str:
        start = time.monotonic()
        self.calls += 1
        try:
            return self.base.complete_text(system=system, user=user)
        finally:
            self.wall_s += time.monotonic()-start


def infer(source: Path, manifest: Path, dbroot: Path, sealed: Path, shard: int, out: Path) -> dict:
    rows, meta = checked_source(source,manifest)
    if shard not in range(N_DB):
        raise ValueError("out-of-range shard")
    db_id=meta["dbs"][shard]
    path=dbroot/db_id/(db_id+".sqlite")
    if not path.is_file():
        raise FileNotFoundError("Original BIRD physical DB missing")
    if _ollama_model_digest("http://127.0.0.1:11434",MODEL) != DIGEST:
        raise ValueError("Original fixed Qwen3.5-9B model digest changed")
    previous=json.loads(sealed.read_text())
    if (previous.get("protocol")!=SOURCE_PROTOCOL
        or previous.get("status")!=SEALED_SOURCE_STATUS
        or previous.get("model_digest")!=DIGEST
        or previous.get("source_sha256")!=meta["source_sha256"]
        or previous.get("shard")!=shard
        or previous.get("db_id")!=db_id
        or previous.get("n")!=PER_DB
        or previous.get("no_gold_in_model_or_selector") is not True):
        raise ValueError("Prior blinded Qwen control not sealed or not identical source")
    catalog=DatabaseCatalog.from_sqlite(path)
    cases=[]
    cohort=rows[shard*PER_DB:(shard+1)*PER_DB]
    for idx,(row,prev) in enumerate(zip(cohort,previous["cases"])):
        if (row["source_index"]!=prev["source_index"]
            or row["db_id"]!=prev["db_id"]
            or row["question_sha256"]!=prev["question_sha256"]):
            raise ValueError("Prior control identity mismatch")
        sql=prev["control_sql"][0]
        schema,_=select_schema_context(catalog,dbroot,db_id,row["question"])
        control=MeterProvider()
        treated=MeterProvider()
        kwargs=dict(
            database=path,question=row["question"],schema_context=schema,
            evidence=row["evidence"] or None,initial_candidate=SQLCandidate(sql),
            max_repairs=2,max_rows=100,
        )
        c=run_guarded(provider=control,adaptive_physical=False,**kwargs)
        t=run_guarded(provider=treated,adaptive_physical=True,**kwargs)
        cases.append({
            "source_index":row["source_index"],
            "question_sha256":row["question_sha256"],
            "db_id":db_id,
            "original_sql":sql,
            "control_sql":c.candidate.sql if c.candidate else "",
            "treatment_sql":t.candidate.sql if t.candidate else "",
            "control_status":c.status,
            "treatment_status":t.status,
            "control_calls":control.calls,
            "treatment_calls":treated.calls,
            "control_wall_s":round(control.wall_s,2),
            "treatment_wall_s":round(treated.wall_s,2),
            "adaptive_nodes":[x.name for x in t.stages],
            "stage_summaries":[x.summary for x in t.stages
                               if x.name.startswith("adaptive_")],
            "adaptive_probed":any(x.name in ("adaptive_data_probe","adaptive_schema_probe")
                                  for x in t.stages),
            "adaptive_fallback":any(x.name=="adaptive_fallback" for x in t.stages),
        })
        print("BIRD_ADAPTIVE_GOLD_BLIND_PAIR",shard,idx+1,PER_DB,
              "extra_qwen",treated.calls,"stage_count",len(t.stages),flush=True)
    obj={"protocol":PROTOCOL,"status":"BIRD_LANGGRAPH_CANDIDATE_SEALED_PRE_GOLD",
         "source_sha256":meta["source_sha256"],"model_digest":DIGEST,
         "shard":shard,"db_id":db_id,"n":PER_DB,"cases":cases,
         "no_gold_in_model_or_route":True,
         "previous_BIRD_question_and_DB_exposed":True,
         "DEV500_used":False,"HOLDOUT300_used":False}
    if out.exists():raise FileExistsError("immutable blind infer seal")
    out.write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False)+"\n")
    return obj


def score(source: Path,manifest: Path,dbroot: Path,sealed: Path,train: Path,shard:int,out:Path)->dict:
    rows,meta=checked_source(source,manifest)
    d=json.loads(sealed.read_text())
    db_id=meta["dbs"][shard]
    if (d.get("status")!="BIRD_LANGGRAPH_CANDIDATE_SEALED_PRE_GOLD"
        or d.get("protocol")!=PROTOCOL or d.get("model_digest")!=DIGEST
        or d.get("source_sha256")!=meta["source_sha256"] or d.get("shard")!=shard
        or d.get("db_id")!=db_id or d.get("n")!=PER_DB
        or d.get("no_gold_in_model_or_route") is not True):
        raise ValueError("Candidate outputs not sealed or source drift")
    source_rows=rows[shard*PER_DB:(shard+1)*PER_DB]
    gold=load_checked(train,TRAIN_SHA)
    path=dbroot/db_id/(db_id+".sqlite")
    scored=[]
    for row,case in zip(source_rows,d["cases"]):
        if (row["source_index"]!=case["source_index"]
            or row["question_sha256"]!=case["question_sha256"]
            or row["db_id"]!=case["db_id"]):
            raise ValueError("Source identity mismatch")
        ref=execute(path,gold[row["source_index"]]["SQL"])
        c=execute(path,case["control_sql"])
        t=execute(path,case["treatment_sql"])
        def eq(candidate,key):
            return bool(ref["ok"] and candidate["ok"]
                        and ref[key]==candidate[key]) if ref["ok"] else None
        scored.append({
            "source_index":row["source_index"],"db_id":db_id,
            "scorable":ref["ok"],
            "control_set":eq(c,"set_sha256"),"treatment_set":eq(t,"set_sha256"),
            "control_bag":eq(c,"bag_sha256"),"treatment_bag":eq(t,"bag_sha256"),
            "control_calls":case["control_calls"],
            "treatment_calls":case["treatment_calls"],
            "control_wall_s":case["control_wall_s"],
            "treatment_wall_s":case["treatment_wall_s"],
            "adaptive_probed":case["adaptive_probed"],
            "adaptive_fallback":case["adaptive_fallback"],
        })
    result={"protocol":PROTOCOL,"status":"POST_SEAL_BIRD_STRICT_PHYSICAL_SCORE",
            "source_sha256":meta["source_sha256"],"model_digest":DIGEST,
            "shard":shard,"db_id":db_id,"n":PER_DB,"cases":scored}
    if out.exists():raise FileExistsError("immutable score seal")
    out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    return result


def aggregate(manifest:Path,parts:list[Path],out:Path)->dict:
    m=json.loads(manifest.read_text())
    if len(parts)!=N_DB:raise ValueError("require all four physical DBs")
    cases=[];shards=set()
    for path in parts:
        p=json.loads(path.read_text()); sh=p.get("shard")
        if (sh in shards or not isinstance(sh,int) or sh not in range(N_DB)
            or p.get("protocol")!=PROTOCOL or p.get("status")!="POST_SEAL_BIRD_STRICT_PHYSICAL_SCORE"
            or p.get("source_sha256")!=m["source_sha256"] or p.get("model_digest")!=DIGEST
            or p.get("db_id")!=m["dbs"][sh] or p.get("n")!=PER_DB):
            raise ValueError("missing or mutated blinded source")
        shards.add(sh);cases+=p["cases"]
    if len(cases)!=N or len({x["source_index"] for x in cases})!=N:
        raise ValueError("not 16 unique original BIRD tasks")
    valid=[x for x in cases if x["scorable"]]
    c=Counter()
    for x in valid:
        a=x["control_set"];b=x["treatment_set"]
        c["control_set"]+=int(a);c["treatment_set"]+=int(b)
        c["control_bag"]+=int(x["control_bag"])
        c["treatment_bag"]+=int(x["treatment_bag"])
        c["rescues"]+=int(b and not a)
        c["regressions"]+=int(a and not b)
    result={"protocol":PROTOCOL,"status":"COMPLETE_REAL_BIRD_LANGGRAPH_PAIRED_TRAIN16",
            "n":N,"scorable":len(valid),"counts":dict(c),
            "selected_net":c["rescues"]-c["regressions"],
            "probe_tasks":sum(x["adaptive_probed"] for x in cases),
            "fallback_tasks":sum(x["adaptive_fallback"] for x in cases),
            "control_extra_model_calls":sum(x["control_calls"] for x in cases),
            "treatment_extra_model_calls":sum(x["treatment_calls"] for x in cases),
            "control_extra_model_wall_s":round(sum(x["control_wall_s"] for x in cases),1),
            "treatment_extra_model_wall_s":round(sum(x["treatment_wall_s"] for x in cases),1),
            "original_sql_shared_not_double_counted":True,
            "previously_exposed_BIRD_train_questions_and_db":True,
            "do_not_promote_or_compare_as_DEV100":True,
            "DEV500_used":False,"HOLDOUT300_used":False,
            "only_original_Qwen3_5_9B":True}
    if out.exists():raise FileExistsError("already finalized")
    out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode", choices=("infer","score","aggregate"))
    for name in ("source","manifest","dbroot","previous","sealed","train","out"):
        p.add_argument("--"+name,type=Path)
    p.add_argument("--shard",type=int,default=0)
    p.add_argument("--parts",nargs="+",type=Path)
    a=p.parse_args()
    if a.mode=="infer":
        value=infer(a.source,a.manifest,a.dbroot,a.previous,a.shard,a.out)
    elif a.mode=="score":
        value=score(a.source,a.manifest,a.dbroot,a.sealed,a.train,a.shard,a.out)
    else:
        value=aggregate(a.manifest,a.parts,a.out)
    print("BIRD_LANGGRAPH_PHYSICAL_REAL",json.dumps(
        {k:v for k,v in value.items() if k!="cases"},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
