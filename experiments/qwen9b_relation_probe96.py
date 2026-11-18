"""Matched TWO-call original Qwen3.5-9B SQL generation with real SQLite relation probe.

Control: same base Qwen makes 2 independent direct SQL attempts, fixed seeds.
          Original K2 plurality tie rule selects FIRST candidate.
Treatment: SAME base Qwen makes 1 executable relation-preview SELECT, executes
           that SQL read-only against actual original SQLite, then uses observed
           columns+at most 5 rows to make 1 FINAL answer SQL.
Exactly TWO same-base model calls per arm. No teacher, training, gold in prompt,
DEV500/HOLDOUT300, or the 2 protected BIRD TRAIN physical databases.
Compare deployable first-control vs final-treatment strict physical SET/BAG EX;
also report control second candidate and oracle@2 as candidate-headroom bounds.
Historical source IDs: full 10 immutable source cohorts plus last 96 experiment.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random
import sqlite3
import time

from evaluation.run_bird import _ollama_model_digest
from evaluation.schema_context import select_schema_context
from scripts.freeze_birdtrain_residual_db_holdout import (
    PREVIOUS_SOURCE_SPECS,historical_source_ledger
)
from scripts.prepare_birdtrain_qwen9b_sft_source import (
    TRAIN_SHA,fetch_database,fetch_manifest,load_checked
)
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.direct import _SYSTEM,_strip_fence
from semantic_sql.execution import (
    _open_readonly_connection,validate_readonly_sql,SQLExecutionError
)
from semantic_sql.providers import OllamaProvider

PROTOCOL="FIXED-QWEN35-9B-TRAIN96-MATCHED-TWO-CALL-EXECUTABLE-RELATION-PROBE-V1"
MODEL="qwen3.5:9b-q4_K_M"
DIGEST="56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
PREVIOUS_96_SHA="cbe11b505524f446b576c60d3b751a5021fd4398deb7c1a7f70b2818e9f73398"
SELECT_SEED="two-call-physical-relational-subquery-probe-20261118"
N_DB=12
PER_DB=8
N=N_DB*PER_DB
SQL_SECONDS=8.
MAX_ROWS=50000
MAX_DB_BYTES=96*1024*1024
GATE={"min_net_correct":8,"min_rescue_dbs":4,"require_full_reference_n":96,
      "min_db_cluster_CI95_lower":0.0}

PROBE_SYSTEM=(
    "Generate ONE read-only SQLite SELECT *probe*, NOT the final answer. "
    "Find the relation, target entity keys, and filter semantics needed to "
    "answer the question. Project up to 5 representative physical rows. "
    "Use only actual schema columns, and finish with LIMIT 5. "
    "Do not calculate the final aggregate. Return SQL only."
)
FINAL_EVIDENCE=(
    "\nRelational evidence from a real, read-only SQLite probe "
    "(only a small sample; not proof of completeness). The probe SQL may "
    "contain errors or inspect an irrelevant relation, so defer to the user "
    "question and physical schema:\n"
    "{observation}\n"
    "Now produce ONE final read-only SQLite SELECT. "
    "Use the observed relation roles, join cardinality and entity keys "
    "only if relevant to the question. Check which entity is being counted "
    "or averaged, who initiated vs received the relation, AND/OR/NOT "
    "scope, numeric denominators, sorting/ties, and required output columns. "
    "Do not mention the probe or explanations. SQL only."
)


def source_freeze(trainfile:Path,source_dir:Path,previous96:Path,outdir:Path)->dict:
    original=load_checked(trainfile,TRAIN_SHA)
    paths={name:source_dir/f"{name}.jsonl" for name in PREVIOUS_SOURCE_SPECS}
    used_dbs,provenance=historical_source_ledger(paths)
    old_ids=set()
    for path in paths.values():
        old_ids.update(int(json.loads(line)["source_index"])
                       for line in path.read_text().splitlines() if line.strip())
    if len(old_ids)!=425 or len(used_dbs)!=49:
        raise ValueError("full historically seen source IDs/physical database drift")
    if sha256(previous96.read_bytes()).hexdigest()!=PREVIOUS_96_SHA:
        raise ValueError("last 96 real Qwen source SHA drift")
    prev=[json.loads(line) for line in previous96.read_text().splitlines() if line.strip()]
    if len(prev)!=96:
        raise ValueError("last exposed plan96 cohort missing")
    old_ids.update(int(row["source_index"]) for row in prev)
    if len(old_ids)!=519:
        raise ValueError("historical question exposure union must be 425+96-2=519")
    meta=fetch_manifest()
    if len(meta)!=51 or len(set(meta)-used_dbs)!=2:
        raise ValueError("original BIRD physical TRAIN source 49/51 audit broken")
    bydb={}
    for idx,item in enumerate(original):
        db=item.get("db_id")
        q=item.get("question")
        ev=item.get("evidence") or ""
        if (idx in old_ids or db not in used_dbs or
            meta[db]["size"]>MAX_DB_BYTES or
            not isinstance(q,str) or len(q.strip())<20 or
            not isinstance(ev,str) or len(q)+len(ev)>6000):
            continue
        bydb.setdefault(db,[]).append(idx)
    order=lambda s:sha256((SELECT_SEED+":"+str(s)).encode()).hexdigest()
    dbs=sorted((db for db,ids in bydb.items() if len(ids)>=PER_DB),key=order)[:N_DB]
    if len(dbs)!=N_DB:raise ValueError("not enough 12 old DBs with new questions")
    selected=[]
    for db in dbs:
        for idx in sorted(bydb[db],key=order)[:PER_DB]:
            item=original[idx]
            selected.append({
                "source_index":idx,"db_id":db,
                "question":item["question"],"evidence":item.get("evidence") or "",
                "question_sha256":sha256(item["question"].encode()).hexdigest()
            })
    assert len(selected)==N and len({r["source_index"] for r in selected})==N
    assert not any(r["db_id"] not in used_dbs or r["source_index"] in old_ids
                   for r in selected)
    outdir.mkdir(parents=True,exist_ok=False)
    source=outdir/"source96.jsonl"
    source.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in selected))
    result={
        "status":"FULL_HISTORY_EXCLUDED_QUESTION_ONLY_SOURCE_FROZEN_PRE_MODEL",
        "protocol":PROTOCOL,"model_digest":DIGEST,"train_sha256":TRAIN_SHA,
        "source_sha256":sha256(source.read_bytes()).hexdigest(),
        "n":N,"n_dbs":N_DB,"dbs":dbs,"per_db":PER_DB,
        "excluded_all_previous_source_indices":len(old_ids),
        "full_10_historical_SHA_checked":True,
        "original_49_exposed_DBs_only":True,
        "last_two_protected_DBs_unused":True,
        "source_has_no_gold":True,"DEV500_unused":True,"HOLDOUT300_unused":True,
        "equal_model_calls_per_arm_per_question":2,"gate":GATE,
        "historical_provenance":provenance["prior_sources"]
    }
    (outdir/"freeze.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    return result


def checked_source(src:Path,meta:Path)->tuple[list[dict],dict]:
    r=json.loads(meta.read_text())
    if (r.get("status")!="FULL_HISTORY_EXCLUDED_QUESTION_ONLY_SOURCE_FROZEN_PRE_MODEL"
        or r.get("protocol")!=PROTOCOL or r.get("model_digest")!=DIGEST
        or r.get("train_sha256")!=TRAIN_SHA
        or r.get("source_sha256")!=sha256(src.read_bytes()).hexdigest()
        or r.get("n")!=N or r.get("n_dbs")!=N_DB or r.get("gate")!=GATE
        or r.get("excluded_all_previous_source_indices")!=519
        or not r.get("last_two_protected_DBs_unused")
        or not r.get("source_has_no_gold")
        or r.get("DEV500_unused") is not True
        or r.get("HOLDOUT300_unused") is not True):
        raise ValueError("frozen source/model/holdout protocol drift")
    rows=[json.loads(x) for x in src.read_text().splitlines() if x.strip()]
    if len(rows)!=N or len({r["source_index"] for r in rows})!=N:
        raise ValueError("source missing 96 unique questions")
    for i,x in enumerate(rows):
        if (x["db_id"]!=r["dbs"][i//PER_DB]
            or x["question_sha256"]!=sha256(x["question"].encode()).hexdigest()
            or any(key in x for key in ("SQL","reference_sql","gold"))):
            raise ValueError("model input contains gold or wrong source/db")
    return rows,r


def execute(db:Path,sql:str,max_rows:int=MAX_ROWS,limit:float=SQL_SECONDS)->dict:
    start=time.monotonic()
    conn=None
    try:
        query=validate_readonly_sql(sql)
        conn,_=_open_readonly_connection(db)
        conn.set_progress_handler(lambda:int(time.monotonic()-start>limit),4000)
        cur=conn.execute(query)
        columns=[str(x[0]) for x in (cur.description or ())]
        rows=cur.fetchmany(max_rows+1)
        if len(rows)>max_rows:raise ValueError("row_cap")
        encoded=[json.dumps(list(row),ensure_ascii=False,default=str,
                            separators=(",",":")) for row in rows]
        a=sha256(json.dumps(sorted(set(encoded))).encode()).hexdigest()
        b=sha256(json.dumps(sorted(Counter(encoded).items())).encode()).hexdigest()
        return {"ok":True,"set_sha256":a,"bag_sha256":b,
                "columns":columns[:12],"rows":rows[:5],"n_rows":len(rows),"reason":"ok"}
    except (SQLExecutionError,sqlite3.Error,ValueError,OSError) as e:
        return {"ok":False,"set_sha256":None,"bag_sha256":None,
                "columns":[],"rows":[],"n_rows":None,
                "reason":"timeout" if time.monotonic()-start>=limit else type(e).__name__}
    finally:
        if conn:
            conn.set_progress_handler(None,0)
            conn.close()


def infer(src:Path,manifest:Path,dbroot:Path,shard:int,out:Path)->dict:
    rows,info=checked_source(src,manifest)
    if not 0<=shard<N_DB:raise ValueError("invalid physical DB worker")
    db=info["dbs"][shard]
    dpath=dbroot/db/f"{db}.sqlite"
    if not dpath.is_file():raise FileNotFoundError("original physical SQLite absent")
    if _ollama_model_digest("http://127.0.0.1:11434",MODEL)!=DIGEST:
        raise ValueError("original Qwen3.5-9B model SHA changed")
    catalog=DatabaseCatalog.from_sqlite(dpath)
    cases=[]
    def generate(provider,sys,user):
        st=time.monotonic()
        raw=provider.complete_text(system=sys,user=user)
        return _strip_fence(raw),round(time.monotonic()-st,2)
    q1=OllamaProvider(model=MODEL,timeout_s=900,temperature=0.7,seed=42,num_predict=384)
    q2=OllamaProvider(model=MODEL,timeout_s=900,temperature=0.7,seed=43,num_predict=384)
    for record in rows[shard*PER_DB:(shard+1)*PER_DB]:
        schema,_=select_schema_context(catalog,dbroot,db,record["question"])
        user=f"Question: {record['question']}\nSchema context:\n{schema}"
        if record["evidence"]:
            user+="\nExternal evidence:\n"+record["evidence"]
        # Predeclared 2 independent control calls. K2 tie -> first is deployed.
        base1,t1=generate(q1,_SYSTEM,user)
        base2,t2=generate(q2,_SYSTEM,user)
        # Exactly 2 treatment calls with original Qwen, no gold or stronger model.
        probe,tp=generate(q1,PROBE_SYSTEM,user)
        observed=execute(dpath,probe,max_rows=5,limit=5.)
        # No full-data-probe "zero rows means no answer" inference.
        fact={
            "SQL":probe[:1200],
            "execution":observed["reason"],
            "columns":observed["columns"],
            "example_rows":observed["rows"][:5],
            "rows_are_sample_only":True
        }
        final,tf=generate(q2,_SYSTEM,user+FINAL_EVIDENCE.format(
            observation=json.dumps(fact,ensure_ascii=False,default=str)[:2200]))
        cases.append({
            "source_index":record["source_index"],"db_id":db,
            "question_sha256":record["question_sha256"],
            "baseline_first_SQL":base1,"baseline_second_SQL":base2,
            "relation_probe_SQL":probe,"treatment_final_SQL":final,
            "probe_executable":observed["ok"],"probe_observed_rows":len(observed["rows"]),
            "model_calls_baseline":2,"model_calls_treatment":2,
            "baseline_wall_sec":round(t1+t2,2),
            "treatment_wall_sec":round(tp+tf,2),
            "model_digest":DIGEST
        })
        print("QWEN9B_EQUAL_BUDGET_REAL_PROBE",shard,len(cases),"/",PER_DB,
              "probe_ok",observed["ok"],flush=True)
    result={
        "status":"BOTH_ARM_REAL_SQL_GENERATIONS_SEALED_BEFORE_REFERENCE",
        "protocol":PROTOCOL,"source_sha256":info["source_sha256"],
        "model_digest":DIGEST,"shard":shard,"physical_db":db,"n":PER_DB,
        "n_baseline_calls":2*PER_DB,"n_treatment_calls":2*PER_DB,
        "gold_in_generation":False,"holdout_used":False,"cases":cases
    }
    if out.exists():raise FileExistsError("original generated SQL immutable")
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    return result


def score(src:Path,manifest:Path,dbroot:Path,shard:int,generated:Path,
          trainfile:Path,out:Path)->dict:
    rows,info=checked_source(src,manifest)
    gen=json.loads(generated.read_text())
    if (gen.get("status")!="BOTH_ARM_REAL_SQL_GENERATIONS_SEALED_BEFORE_REFERENCE"
        or gen.get("protocol")!=PROTOCOL or gen.get("model_digest")!=DIGEST
        or gen.get("source_sha256")!=info["source_sha256"]
        or gen.get("shard")!=shard or gen.get("n")!=PER_DB
        or gen.get("n_baseline_calls")!=2*PER_DB
        or gen.get("n_treatment_calls")!=2*PER_DB
        or gen.get("gold_in_generation") is not False):
        raise ValueError("candidate generation incomplete or labels not separated")
    actual=load_checked(trainfile,TRAIN_SHA)
    scored=[]
    for row,g in zip(rows[shard*PER_DB:(shard+1)*PER_DB],gen["cases"]):
        if (g["source_index"]!=row["source_index"]
            or g["db_id"]!=row["db_id"]
            or g["question_sha256"]!=row["question_sha256"]):
            raise ValueError("generated candidate not paired with frozen source")
        database=dbroot/row["db_id"]/f"{row['db_id']}.sqlite"
        a=execute(database,g["baseline_first_SQL"])
        b=execute(database,g["baseline_second_SQL"])
        t=execute(database,g["treatment_final_SQL"])
        # Publisher gold read only AFTER all original model SQLs are sealed.
        reference=execute(database,actual[row["source_index"]]["SQL"])
        valid=reference["ok"]
        def strict(outcome,key):
            return bool(outcome["ok"] and outcome[key]==reference[key]) if valid else None
        base1=strict(a,"set_sha256")
        base2=strict(b,"set_sha256")
        treat=strict(t,"set_sha256")
        scored.append({
            "source_index":row["source_index"],"db_id":row["db_id"],
            "reference_scorable":valid,"reference_failure":None if valid else reference["reason"],
            "baseline_selected_set_EX":base1,
            "baseline_second_set_EX":base2,
            "baseline_oracle_at2_set_EX":(base1 or base2) if valid else None,
            "treatment_selected_set_EX":treat,
            "baseline_selected_bag_EX":strict(a,"bag_sha256"),
            "treatment_selected_bag_EX":strict(t,"bag_sha256"),
            "baseline_candidate2_executable":b["ok"],
            "treatment_final_executable":t["ok"],
            "probe_executable":g["probe_executable"],
            "baseline_calls":g["model_calls_baseline"],
            "treatment_calls":g["model_calls_treatment"],
            "baseline_wall_sec":g["baseline_wall_sec"],
            "treatment_wall_sec":g["treatment_wall_sec"]
        })
    result={"status":"ORIGINAL_QWEN9B_PAIRED_PHYSICAL_STRICT_EX_AFTER_GOLD_BLIND_SQL",
            "protocol":PROTOCOL,"source_sha256":info["source_sha256"],
            "model_digest":DIGEST,"shard":shard,"db_id":info["dbs"][shard],
            "n":PER_DB,"DEV500_unused":True,"HOLDOUT300_unused":True,
            "cases":scored}
    if out.exists():raise FileExistsError("score evidence immutable")
    out.write_text(json.dumps(result,indent=2)+"\n")
    return result


def summarize(parts:list[Path],manifest:Path,out:Path)->dict:
    meta=json.loads(manifest.read_text())
    if len(parts)!=N_DB:raise ValueError("all 12 source DBA workers required")
    seen=set();records=[]
    for p in parts:
        x=json.loads(p.read_text())
        sh=x.get("shard")
        if (sh in seen or not isinstance(sh,int) or sh not in range(N_DB)
            or x.get("status")!="ORIGINAL_QWEN9B_PAIRED_PHYSICAL_STRICT_EX_AFTER_GOLD_BLIND_SQL"
            or x.get("protocol")!=PROTOCOL or x.get("model_digest")!=DIGEST
            or x.get("source_sha256")!=meta["source_sha256"]
            or x.get("db_id")!=meta["dbs"][sh] or x.get("n")!=PER_DB
            or len(x.get("cases",[]))!=PER_DB
            or x.get("DEV500_unused") is not True
            or x.get("HOLDOUT300_unused") is not True):
            raise ValueError("incorrect source Qwen model or missing output")
        seen.add(sh);records+=x["cases"]
    if seen!=set(range(N_DB)) or len(records)!=N or len({r["source_index"] for r in records})!=N:
        raise ValueError("not complete 96 original source frozen pairs")
    sc=[x for x in records if x["reference_scorable"]]
    wins=[x for x in sc if not x["baseline_selected_set_EX"] and x["treatment_selected_set_EX"]]
    losses=[x for x in sc if x["baseline_selected_set_EX"] and not x["treatment_selected_set_EX"]]
    delta=len(wins)-len(losses)
    groups={db:[int(x["treatment_selected_set_EX"])-int(x["baseline_selected_set_EX"])
                for x in sc if x["db_id"]==db] for db in meta["dbs"]}
    if any(not v for v in groups.values()):
        raise ValueError("cannot claim full DB cluster CI with missing DBs")
    rng=random.Random(20261118)
    ordered=sorted(groups)
    sims=[]
    for _ in range(4000):
        ds=[rng.choice(ordered) for _ in ordered]
        values=[v for d in ds for v in groups[d]]
        sims.append(sum(values)/len(values))
    sims.sort()
    ci={"lower95":sims[100],"upper95":sims[3899],
        "mean":delta/max(1,len(sc)),"n_db_clusters":N_DB}
    positive=(
        len(sc)==GATE["require_full_reference_n"]
        and delta>=GATE["min_net_correct"]
        and len({x["db_id"] for x in wins})>=GATE["min_rescue_dbs"]
        and ci["lower95"]>GATE["min_db_cluster_CI95_lower"]
    )
    report={
        "status":"COMPLETE_REAL_QWEN9B_TRAIN96_MATCHED_TWO_CALL_RELATION_PROBE",
        "protocol":PROTOCOL,"model_digest":DIGEST,
        "n":N,"scorable":len(sc),"censored":N-len(sc),
        "censored_ids":[x["source_index"] for x in records if not x["reference_scorable"]],
        "physical_dbs":N_DB,
        "baseline_selected_set_EX_correct":sum(x["baseline_selected_set_EX"] for x in sc),
        "baseline_second_set_EX_correct":sum(x["baseline_second_set_EX"] for x in sc),
        "baseline_oracle_at2_correct":sum(x["baseline_oracle_at2_set_EX"] for x in sc),
        "treatment_final_set_EX_correct":sum(x["treatment_selected_set_EX"] for x in sc),
        "baseline_selected_bag_EX_correct":sum(x["baseline_selected_bag_EX"] for x in sc),
        "treatment_bag_EX_correct":sum(x["treatment_selected_bag_EX"] for x in sc),
        "rescue_cases":len(wins),"regression_cases":len(losses),"net_correct":delta,
        "rescues_on_database_count":len({x["db_id"] for x in wins}),
        "paired_db_cluster_CI95":ci,
        "probe_success_count":sum(x["probe_executable"] for x in records),
        "baseline_second_SQL_exec_count":sum(x["baseline_candidate2_executable"] for x in records),
        "treatment_final_SQL_exec_count":sum(x["treatment_final_executable"] for x in records),
        "baseline_model_calls":sum(x["baseline_calls"] for x in records),
        "treatment_model_calls":sum(x["treatment_calls"] for x in records),
        "baseline_total_wall_s":round(sum(x["baseline_wall_sec"] for x in records),2),
        "treatment_total_wall_s":round(sum(x["treatment_wall_sec"] for x in records),2),
        "predeclared_gate":GATE,"accepted":positive,
        "DEV500_used":False,"HOLDOUT300_used":False,
        "last_2_protected_original_TRAIN_DBS_used":False,
        "source_independent_new_questions_but_not_new_physical_databases":True,
        "candidate_limit":"One source was selected as the deployed answer from each arm BEFORE scoring; control had K2 independent SQL, but ties revert to first, treatment had 1 tool query + one final SQL. Original K2 oracle@2 is a hindsight-only headroom bound, not a gold-blind selector."
    }
    if out.exists():raise FileExistsError("complete source scoreboard immutable")
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("freeze","infer","score","aggregate"))
    for key in ("train","history","prior96","out","source","manifest","dbroot","generated"):
        p.add_argument("--"+key,type=Path)
    p.add_argument("--shard",type=int,default=0)
    p.add_argument("--parts",type=Path,nargs="+")
    a=p.parse_args()
    if a.mode=="freeze":
        report=source_freeze(a.train,a.history,a.prior96,a.out)
    elif a.mode=="infer":
        report=infer(a.source,a.manifest,a.dbroot,a.shard,a.out)
    elif a.mode=="score":
        report=score(a.source,a.manifest,a.dbroot,a.shard,a.generated,a.train,a.out)
    else:
        report=summarize(a.parts,a.manifest,a.out)
    print("MATCHED_QWEN9B_TWO_CALL_RELATION_PROBE",json.dumps({
        k:v for k,v in report.items() if k not in ("cases","historical_provenance")
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
