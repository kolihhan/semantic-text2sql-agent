"""One fixed-Qwen3.5-9B K2 candidate-generation experiment with physical FK fanout.

Mechanism (one intentional information difference):
 Control candidate 2: original model/system prompt + original question/schema/evidence.
 Treatment candidate 2: SAME original model, system, seed, sampling budget, but
 adds measured, read-only child-FK key multiplicity for visible schema edges.
 Shared candidate 1 is bit-identical and generated ONCE.

Hypothesis: the measured relational grain prevents some generation errors that
purely declarative FK schema context cannot; specifically it can improve the
success of the SECOND candidate, not simply add more random samples.

Cohort selected before Qwen generation by physical schema/fanout availability,
NOT reference SQL patterns, baseline EX or model outputs. Source SQL validates
compilability only; used to score AFTER all candidate SQL are frozen.
Prospective BIRD TRAIN question-disjoint, physical DBs NOT guaranteed unseen.
NOT new BIRD DEV/OOD, no HOLDOUT300 or DEV500.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random
import sqlite3
import time

from evaluation.bounded_shard7_probe import bounded_outcome
from evaluation.kcache_protocol import MODEL_DIGEST, MODEL_NAME, majority_selector, candidate_seed
from evaluation.qwen9b_kcache import ollama_one
from evaluation.run_bird import _ollama_model_digest
from evaluation.schema_context import select_schema_context
from scripts.prepare_birdtrain_qwen9b_sft_source import (
    TRAIN_SHA,load_checked,fetch_manifest,fetch_database,normalized,validate_gold_sql)
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.direct import _SYSTEM

PROTOCOL="BIRD-TRAIN-QWEN35-9B-ORIGINAL-K2-VS-PHYSICAL-FK-MULTIPLICITY-K2-V1"
SOURCE_SELECTION_SEED="birdtrain-physical-fk-fanout-balanced-20261118"
N_DB=4
PER_DB=4
N=N_DB*PER_DB
SHARDS=4
MIN_FANOUT_CASES=8
TEMP=0.7
MAX_TOKENS=384
SQL_SECONDS=8.
MAX_DB_BYTES=96*1024*1024
ORIGINAL_TRAIN242_SHA="b6851fb0f2b5027833aff45a0b5e01326359b245be1c812edc66986d0df1d402"
ORIGINAL_VAL64_SHA="cc112305bc9de33a03cea704f2edecfdfb45e62cc953ae498cea27fbed93b506"
ORIGINAL_GRAIN16_SHA="ca68478128821b253dca9db600b101b9140fa645f2e38ed2e3493603db7efd8e"
GATE={"min_treatment_candidate2_net_ex":2,"max_treatment_candidate2_regressions":1,
      "min_rescue_databases":2,"min_pass2_net":2,
      "min_fanout_covered_source_cases":MIN_FANOUT_CASES}


def sha_order(context:str,value:str)->str:
    return sha256(f"{SOURCE_SELECTION_SEED}:{context}:{value}".encode()).hexdigest()


def _ident(s:str)->str:
    return '"'+s.replace('"','""')+'"'


def _read_capped(con:sqlite3.Connection,query:str)->tuple|None:
    limit=time.monotonic()+3.
    con.set_progress_handler(lambda:1 if time.monotonic()>limit else 0,10000)
    try:
        return con.execute(query).fetchone()
    except sqlite3.Error:
        return None
    finally:
        con.set_progress_handler(None,0)


def physical_fk_multiplicity(db:Path,catalog:DatabaseCatalog,schema_text:str)->tuple[str,int]:
    """Only physical row multiplicity for *already shown* FK schema edges.

    This is not a gold SQL lookup, does not propose any answer, does not reveal
    reference query or model outcome, and makes no unsupported entity guesses.
    """
    if not db.is_file() or not catalog.tables:
        raise ValueError("missing real source SQLite database")
    seen=set()
    for line in schema_text.splitlines():
        if "(" in line and ")" in line and not line.startswith(("FK ","- ")):
            candidate=line.split("(",1)[0].strip()
            if candidate in catalog.tables:
                seen.add(candidate)
    if not seen:
        return "",0
    lines=[]
    fanout=0
    uri="file:"+db.resolve().as_posix()+"?mode=ro"
    with sqlite3.connect(uri,uri=True) as con:
        con.execute("PRAGMA query_only=ON")
        for child,child_col,parent,parent_col in sorted(catalog.foreign_keys):
            if child not in seen or parent not in seen:
                continue
            statement=(f"SELECT COUNT(*), COUNT(DISTINCT {_ident(child_col)}) "
                       f"FROM {_ident(child)} WHERE {_ident(child_col)} IS NOT NULL")
            got=_read_capped(con,statement)
            if got is None:
                continue
            child_links,distinct_targets=map(int,got)
            if child_links>distinct_targets and distinct_targets>0:
                fanout+=1
            lines.append(
                f"- FK {child}.{child_col} -> {parent}.{parent_col}: "
                f"linked child records={child_links}; distinct parent-key values="
                f"{distinct_targets}; multiple child records per parent-key="
                f"{'yes' if child_links>distinct_targets else 'no'}"
            )
            if len(lines)>=8:
                break
    return "\n".join(lines),fanout


def _old_questions(train242:Path,val64:Path,grain16:Path)->tuple[set[int],set[str]]:
    files=((train242,ORIGINAL_TRAIN242_SHA,242),
           (val64,ORIGINAL_VAL64_SHA,64),
           (grain16,ORIGINAL_GRAIN16_SHA,16))
    ids=set();dbs=set()
    for path,sha,n in files:
        blob=path.read_bytes()
        if sha256(blob).hexdigest()!=sha:
            raise ValueError("old published source evidence SHA mismatch")
        rows=[json.loads(x) for x in blob.decode().splitlines() if x.strip()]
        if len(rows)!=n:
            raise ValueError("previously used source row count mismatch")
        for row in rows:
            ids.add(int(row["source_index"]))
            dbs.add(row.get("source_db_id") or row.get("db_id"))
    return ids,dbs


def freeze(train_path:Path,prior_train:Path,prior_val:Path,prior_grain:Path,
           work:Path,out:Path)->dict:
    train=load_checked(train_path,TRAIN_SHA)
    old_ids,prior_dbs=_old_questions(prior_train,prior_val,prior_grain)
    all_meta=fetch_manifest()
    by_db=defaultdict(list)
    for ix,row in enumerate(train):
        db=row.get("db_id")
        q=normalized(row.get("question",""))
        if ix in old_ids or not q or db not in all_meta or all_meta[db]["size"]>MAX_DB_BYTES:
            continue
        by_db[db].append((ix,row))
    dbs=sorted((db for db,rows in by_db.items() if len(rows)>=PER_DB),
               key=lambda db:(db in prior_dbs,sha_order("db",db)))
    chosen=[]
    visited=[]
    for dbid in dbs:
        db=fetch_database(dbid,all_meta[dbid],work)
        catalog=DatabaseCatalog.from_sqlite(db)
        local=[]
        for ix,row in sorted(by_db[dbid],key=lambda pair:sha_order("row",str(pair[0]))):
            question=str(row["question"]).strip()
            schema,_=select_schema_context(catalog,work,dbid,question)
            fk_note,fanout=physical_fk_multiplicity(db,catalog,schema)
            if not fk_note:
                continue
            original_user=f"Question: {question}\nSchema context:\n{schema}"
            evidence=str(row.get("evidence") or "").strip()
            if evidence:
                original_user+="\nExternal evidence:\n"+evidence
            treatment_user=(original_user+
                "\nPhysical FK multiplicity observations from THIS database (not SQL answer):\n"+
                fk_note+"\nUse the question to decide which entity or aggregate is requested.")
            if max(len(original_user),len(treatment_user))>16000:
                continue
            # Gold is not used to choose a *question type*, only to exclude
            # reference SQL that SQLite cannot compile; no scoring here.
            gold=str(row.get("SQL") or "").strip()
            if validate_gold_sql(db,gold) is not None:
                continue
            local.append({
                "source_index":ix,"db_id":dbid,"question":question,
                "original_user":original_user,"intervention_user":treatment_user,
                "gold_sql":gold,"physical_fk_fanout_positive":fanout>0,
                "physical_fk_edges_shown":fk_note.count("\n")+1,
                "original_prompt_sha256":sha256(original_user.encode()).hexdigest(),
                "treatment_prompt_sha256":sha256(treatment_user.encode()).hexdigest(),
            })
            if len(local)>=PER_DB:
                break
        if len(local)<PER_DB:
            continue
        chosen.extend(local)
        visited.append(dbid)
        if len(visited)>=N_DB:
            break
    if len(chosen)!=N or len(visited)!=N_DB:
        raise RuntimeError(f"insufficient candidate source DB/questions with real FK edges: {visited} n={len(chosen)}")
    fanout_n=sum(int(x["physical_fk_fanout_positive"]) for x in chosen)
    if fanout_n<MIN_FANOUT_CASES:
        raise RuntimeError(f"fail closed: only {fanout_n}/{N} cases with measured child-parent row fanout, require {MIN_FANOUT_CASES}")
    if len({r["source_index"] for r in chosen})!=N:
        raise ValueError("duplicate source question")
    if any(r["source_index"] in old_ids for r in chosen):
        raise ValueError("selected an exposed source question")
    out.mkdir(parents=True,exist_ok=True)
    p=out/"source16.jsonl"
    p.write_text("".join(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n"
                         for r in chosen),encoding="utf-8")
    report={
        "status":"SOURCE_QUESTION_FROZEN_BEFORE_MODEL_OR_EX",
        "protocol":PROTOCOL,
        "source_original_train_sha256":TRAIN_SHA,
        "source_sha256":sha256(p.read_bytes()).hexdigest(),
        "model_digest":MODEL_DIGEST,"n":N,"shards":SHARDS,
        "physical_databases":visited,
        "already_visited_db_count":sum(db in prior_dbs for db in visited),
        "unvisited_relative_to_prior3_source_count":sum(db not in prior_dbs for db in visited),
        "prior_excluded_questions":len(old_ids),
        "cases_with_measurable_fk_fanout":fanout_n,
        "cases_with_fk_evidence":N,
        "predeclared_gate":GATE,
        "new_model_calls":0,
        "source_selection_by_original_question_and_physical_FK_only":True,
        "no_selection_on_model_predictions_or_source_EX":True,
        "no_sql_qlora":True,
        "not_dev_score":True,"holdout300_unused":True,
        "physical_db_ood_not_claimed":True,
        "source_limits":"Candidate DBs may have been examined in earlier experiments beyond the 3 SHA-locked source artifacts. Independently new source questions are not necessarily new physical databases."
    }
    (out/"freeze.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report


def checked_rows(cohort:Path,meta:Path)->tuple[list[dict],dict]:
    m=json.loads(meta.read_text())
    raw=cohort.read_bytes()
    if (m.get("status")!="SOURCE_QUESTION_FROZEN_BEFORE_MODEL_OR_EX" or
        m.get("protocol")!=PROTOCOL or m.get("model_digest")!=MODEL_DIGEST or
        m.get("source_original_train_sha256")!=TRAIN_SHA or
        m.get("source_sha256")!=sha256(raw).hexdigest() or
        m.get("predeclared_gate")!=GATE or m.get("n")!=N or m.get("shards")!=SHARDS):
        raise ValueError("original source/model/predeclared gate drift")
    rows=[json.loads(x) for x in raw.decode().splitlines() if x.strip()]
    if len(rows)!=N or len({r["source_index"] for r in rows})!=N:
        raise ValueError("missing or duplicate frozen source question IDs")
    if len({r["db_id"] for r in rows})!=N_DB:
        raise ValueError("frozen physical database count mismatch")
    for row in rows:
        if (row["original_prompt_sha256"]!=sha256(row["original_user"].encode()).hexdigest() or
            row["treatment_prompt_sha256"]!=sha256(row["intervention_user"].encode()).hexdigest() or
            not row["physical_fk_edges_shown"]>=1):
            raise ValueError("baseline/treatment SQL prompt drift or missing physical FK")
    return rows,m


def infer_worker(cohort:Path,meta:Path,shard:int,dbroot:Path,base_url:str)->dict:
    if shard not in range(SHARDS):
        raise ValueError("invalid shard")
    rows,m=checked_rows(cohort,meta)
    cases=rows[shard*PER_DB:(shard+1)*PER_DB]
    if len(cases)!=PER_DB or len({r["db_id"] for r in cases})!=1:
        raise ValueError("not the exact four frozen source cases/physical DB for worker")
    if _ollama_model_digest(base_url,MODEL_NAME)!=MODEL_DIGEST:
        raise RuntimeError("Base Model digest changed: do not run")
    samples=[]
    for row in cases:
        ix=str(row["source_index"])
        seed0=candidate_seed(ix,0,salt="physical-fk-k2-v1")
        seed1=candidate_seed(ix,1,salt="physical-fk-k2-v1")
        a=ollama_one(base_url,system=_SYSTEM,user=row["original_user"],
                     seed=seed0,temperature=TEMP,num_predict=MAX_TOKENS)
        b=ollama_one(base_url,system=_SYSTEM,user=row["original_user"],
                     seed=seed1,temperature=TEMP,num_predict=MAX_TOKENS)
        c=ollama_one(base_url,system=_SYSTEM,user=row["intervention_user"],
                     seed=seed1,temperature=TEMP,num_predict=MAX_TOKENS)
        if not all(isinstance(x["sql"],str) for x in (a,b,c)):
            raise ValueError("original Qwen output not a string")
        samples.append({
            "source_index":row["source_index"],"db_id":row["db_id"],
            "original_prompt_sha256":row["original_prompt_sha256"],
            "treatment_prompt_sha256":row["treatment_prompt_sha256"],
            "base_first":a,"base_second":b,"physical_second":c,
        })
        print("SAME_MODEL_FK_GEN",shard,len(samples),"/",PER_DB,
              "fixed_digest",MODEL_DIGEST,"sample_calls",3,flush=True)
    return {
        "status":"ORIGINAL_FIXED_QWEN3_5_9B_CANDIDATE_POOL_COMPLETE",
        "protocol":PROTOCOL,"model_digest":MODEL_DIGEST,
        "source_sha256":m["source_sha256"],
        "worker":shard,"shards":SHARDS,"n":len(samples),
        "original_qwen_model_calls":PER_DB*3,
        "training_steps":0,"DEV500_used":False,"HOLDOUT300_used":False,
        "cases":samples
    }


def score_worker(cohort:Path,meta:Path,samples:dict,dbroot:Path,shard:int)->dict:
    from evaluation.run_bird import _resolve_database
    rows,m=checked_rows(cohort,meta)
    if (samples.get("status")!="ORIGINAL_FIXED_QWEN3_5_9B_CANDIDATE_POOL_COMPLETE" or
        samples.get("model_digest")!=MODEL_DIGEST or
        samples.get("source_sha256")!=m["source_sha256"] or
        samples.get("worker")!=shard or samples.get("n")!=PER_DB or
        samples.get("original_qwen_model_calls")!=PER_DB*3 or
        len(samples.get("cases",[]))!=PER_DB):
        raise ValueError("source-locked original model candidate cache not complete")
    scored=[]
    frozen_rows=rows[shard*PER_DB:(shard+1)*PER_DB]
    for row,c in zip(frozen_rows,samples["cases"]):
        if (row["source_index"]!=c["source_index"] or
            c["db_id"]!=row["db_id"] or
            c["original_prompt_sha256"]!=row["original_prompt_sha256"] or
            c["treatment_prompt_sha256"]!=row["treatment_prompt_sha256"]):
            raise ValueError("original same-Qwen samples not aligned to frozen source")
        db=_resolve_database(dbroot,row["db_id"])
        candidate_results=[bounded_outcome(db,c[key]["sql"],SQL_SECONDS)
                           for key in ("base_first","base_second","physical_second")]
        base_choice=majority_selector(candidate_results[:2])["index"]
        treatment_choice=majority_selector([candidate_results[0],candidate_results[2]])["index"]
        # Only now load/score publisher reference SQL; no labels to either
        # candidate generator or deployed selection decision.
        gold=bounded_outcome(db,row["gold_sql"],SQL_SECONDS)
        if not gold["valid"]:
            raise RuntimeError("reference SQL timed out; unscorable case no silent denominator")
        correctness=[x["valid"] and x["set_hash"]==gold["set_hash"] for x in candidate_results]
        scored.append({
            "source_index":row["source_index"],"db_id":row["db_id"],
            "first_correct":bool(correctness[0]),
            "normal_second_correct":bool(correctness[1]),
            "physical_second_correct":bool(correctness[2]),
            "normal_pass2_oracle":bool(correctness[0] or correctness[1]),
            "physical_pass2_oracle":bool(correctness[0] or correctness[2]),
            "normal_selected_correct":bool(correctness[base_choice]),
            "physical_selected_correct":bool(correctness[[0,2][treatment_choice]]),
            "normal_selected_index":base_choice,
            "physical_selected_index":[0,2][treatment_choice],
            "any_candidate_timeout":any(x.get("reason")=="query_timeout" for x in candidate_results),
            "physical_fk_fanout_positive":row["physical_fk_fanout_positive"],
            "generation_seconds":{
              k:c[k]["wall_s"] for k in ("base_first","base_second","physical_second")
            },
        })
    return {
        "status":"SCORE_ALL_ORIGINAL_FIXED_QWEN_K2_CANDIDATES",
        "protocol":PROTOCOL,"worker":shard,"shards":SHARDS,
        "model_digest":MODEL_DIGEST,"source_sha256":m["source_sha256"],
        "n":len(scored),"candidates_per_question_generated":3,
        "DEV500_used":False,"HOLDOUT300_used":False,
        "cases":scored
    }


def aggregate(parts:list[dict],source:Path,meta:Path)->dict:
    rows,m=checked_rows(source,meta)
    if len(parts)!=SHARDS or sorted(x["worker"] for x in parts)!=list(range(SHARDS)):
        raise ValueError("need all four distinct workers")
    records=[]
    for p in parts:
        if (p.get("status")!="SCORE_ALL_ORIGINAL_FIXED_QWEN_K2_CANDIDATES" or
            p.get("model_digest")!=MODEL_DIGEST or
            p.get("source_sha256")!=m["source_sha256"] or
            p.get("n")!=PER_DB or len(p.get("cases",[]))!=PER_DB or
            p.get("HOLDOUT300_used") is not False):
            raise ValueError("incomplete/untrusted original Qwen original source worker")
        records.extend(p["cases"])
    if len(records)!=N or sorted(x["source_index"] for x in records)!=sorted(x["source_index"] for x in rows):
        raise ValueError("frozen source IDs missing or repeated in final result")
    def count(key):return sum(bool(x[key]) for x in records)
    def pairs(a,b):
        gain=[x for x in records if x[b] and not x[a]]
        loss=[x for x in records if x[a] and not x[b]]
        return {"rescues":len(gain),"regressions":len(loss),
                "net":len(gain)-len(loss),
                "rescue_dbs":len({x["db_id"] for x in gain}),
                "regression_dbs":len({x["db_id"] for x in loss})}
    second=pairs("normal_second_correct","physical_second_correct")
    oracle=pairs("normal_pass2_oracle","physical_pass2_oracle")
    chosen=pairs("normal_selected_correct","physical_selected_correct")
    gate=(second["net"]>=GATE["min_treatment_candidate2_net_ex"] and
          second["regressions"]<=GATE["max_treatment_candidate2_regressions"] and
          second["rescue_dbs"]>=GATE["min_rescue_databases"] and
          oracle["net"]>=GATE["min_pass2_net"])
    return {
        "status":"COMPLETE_PROSPECTIVE_FIXED_MODEL_BIRD_TRAIN_FANOUT_K2" if gate else "FAILED_PREREGISTERED_FANOUT_GENERATION_SCREEN",
        "protocol":PROTOCOL,"model":MODEL_NAME,"model_digest":MODEL_DIGEST,
        "source_sha256":m["source_sha256"],"n":N,"physical_db_n":N_DB,
        "source_dbs":m["physical_databases"],
        "source_fanout_covered":m["cases_with_measurable_fk_fanout"],
        "first_correct":count("first_correct"),
        "normal_second_correct":count("normal_second_correct"),
        "physical_second_correct":count("physical_second_correct"),
        "normal_pass2_oracle":count("normal_pass2_oracle"),
        "physical_pass2_oracle":count("physical_pass2_oracle"),
        "normal_selected_EX":count("normal_selected_correct"),
        "physical_selected_EX":count("physical_selected_correct"),
        "second_candidate_paired":second,"pass2_oracle_paired":oracle,
        "goldblind_selected_paired":chosen,
        "positive_mechanism_screen":gate,"predeclared_gate":GATE,
        "calls_total_real_original_qwen":N*3,
        "per_arm_candidate_budget":2,
        "common_first_candidate_shared":True,
        "no_model_swap":True,"no_qlora":True,
        "holdout300_unused":True,"dev500_unused":True,
        "no_independent_source_generalization_claim":True,
        "boundary":"Same Qwen3.5-9B second-candidate paired physical FK multiplicity vs plain prompt, same seed/temp/384 tokens. Retrospective publisher strict EX after all original candidates; pass@2 oracle is never a deployable selection policy. 16 conditioned public TRAIN questions and historically visited physical databases do not establish full BIRD/OOD/medal improvements."
    }


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("freeze","infer","score","aggregate"))
    p.add_argument("--train",type=Path)
    p.add_argument("--prior-train",type=Path)
    p.add_argument("--prior-val",type=Path)
    p.add_argument("--prior-grain",type=Path)
    p.add_argument("--work",type=Path,default=Path("physical-train-sqlite"))
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--source",type=Path)
    p.add_argument("--meta",type=Path)
    p.add_argument("--samples",type=Path)
    p.add_argument("--db-root",type=Path)
    p.add_argument("--shard",type=int,default=0)
    p.add_argument("--inputs",type=Path,nargs="+")
    p.add_argument("--base-url",default="http://127.0.0.1:11434")
    a=p.parse_args()
    if a.out.exists():
        raise FileExistsError("never overwrite a frozen source/physical result")
    if a.mode=="freeze":
        if not all((a.train,a.prior_train,a.prior_val,a.prior_grain)):
            p.error("freeze requires pinned train and three source cohort proofs")
        result=freeze(a.train,a.prior_train,a.prior_val,a.prior_grain,a.work,a.out)
        print("FROZEN_FK_FANOUT_K2_TRAIN_SOURCE",json.dumps(result,sort_keys=True),flush=True)
        return
    if not a.source or not a.meta:
        p.error("source and meta mandatory for infer/score/aggregate")
    if a.mode=="infer":
        result=infer_worker(a.source,a.meta,a.shard,a.db_root or a.work,a.base_url)
    elif a.mode=="score":
        if not a.samples or not a.db_root:p.error("score needs samples and physical DB root")
        result=score_worker(a.source,a.meta,json.loads(a.samples.read_text()),a.db_root,a.shard)
    else:
        if not a.inputs:p.error("aggregate needs four worker result files")
        result=aggregate([json.loads(x.read_text()) for x in a.inputs],a.source,a.meta)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("PHYSICAL_FK_MULTIPLICITY_K2",json.dumps({k:v for k,v in result.items() if k!="cases"},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
