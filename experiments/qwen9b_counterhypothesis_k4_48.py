"""Prospective equal-K4 original Qwen3.5-9B structural counterhypotheses on BIRD TRAIN48.

Control = four independent original SQL generations, seeds 42-45.
Treatment = two identical shared original generations (seeds 42-43),
            then two SAME-model SQL generations conditioned on the two original
            candidate SQLs, with complementary structural hypotheses (44-45).
Each arm owns exactly FOUR original Qwen calls per question; actual six calls
are sufficient because the two initial calls are shared.

The two counterhypotheses test structurally different *candidate generation*,
not an added system instruction to all default SQL generation. Selection uses
actual read-only SQLite output hashes by GOLD-BLIND plurality, earliest valid
tie break. All SQL/output signatures/choices sealed BEFORE publisher gold SQL
is fetched. Then separately score oracle@4 and deployed selected SET/BAG EX.
No new model/teacher/training/DEV500/HOLDOUT300 or two protected TRAIN DBs.
48 *new* TRAIN question identities across 8 exposed physical DBs are a SCREEN,
never a promotion result. All source IDs exclude 711 prior exposures.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random
import re
import time

from evaluation.run_bird import _ollama_model_digest
from evaluation.schema_context import select_schema_context
from scripts.freeze_birdtrain_residual_db_holdout import (
    PREVIOUS_SOURCE_SPECS, historical_source_ledger,
)
from scripts.prepare_birdtrain_qwen9b_sft_source import (
    TRAIN_SHA, fetch_manifest, load_checked,
)
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.direct import _SYSTEM, _strip_fence
from semantic_sql.providers import OllamaProvider
from experiments.qwen9b_semantic_roles96 import execute, MODEL, DIGEST, MAX_DB_BYTES

PROTOCOL="QWEN35-9B-TRAIN48-MATCHED-K4-STRUCTURAL-COUNTERHYPOTHESIS-V1"
N_DB=8
PER_DB=6
N=N_DB*PER_DB
BASE_SEEDS=(42,43,44,45)
COUNTER_SEEDS=(44,45)
OLD_PLAN_SHA="cbe11b505524f446b576c60d3b751a5021fd4398deb7c1a7f70b2818e9f73398"
OLD_PROBE_SHA="822a46c511806ed8056e13ccebf3d4c97c5e8d69046e3b1e86456a481abfeac4"
OLD_ROLE_SHA="07c1057a8db60dc4fa2047ad900ba4b488a2040515edd1278914fb4f106e427b"
SELECT_SEED="new-structural-counterhypothesis-train48-20261119"
GATE={"min_oracle_at4_net":4,"min_selected_net":3,
      "max_selected_regressions":2,"min_scorable":42,
      "min_rescue_dbs":3}
STRUCTURE_OPS=("count","distinct","sum","avg","min","max","group by",
               "having","order by","limit","join","left join","exists",
               "union","except","intersect","case","/")

def sha(path:Path)->str:
    return sha256(path.read_bytes()).hexdigest()

def freeze(train:Path,history:Path,plan:Path,probe:Path,
           role:Path,out:Path)->dict:
    all_rows=load_checked(train,TRAIN_SHA)
    paths={label:history/(label+".jsonl") for label in PREVIOUS_SOURCE_SPECS}
    used_dbs,provenance=historical_source_ledger(paths)
    old={int(r["source_index"]) for p in paths.values()
         for line in p.read_text().splitlines() if line.strip()
         for r in (json.loads(line),)}
    if len(old)!=425 or len(used_dbs)!=49:
        raise ValueError("historical BIRD TRAIN baseline source drift")
    for path,expected,n,expected_total in (
        (plan,OLD_PLAN_SHA,96,519),
        (probe,OLD_PROBE_SHA,96,615),
        (role,OLD_ROLE_SHA,96,711)):
        if sha(path)!=expected:
            raise ValueError("original question-only cohort SHA changed")
        records=[json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        ids={int(r["source_index"]) for r in records}
        if len(records)!=n or len(ids)!=n:
            raise ValueError("cohort has missing or duplicate question IDs")
        if path != plan and ids & old:
            raise ValueError("source was not independent of earlier cohort")
        old.update(ids)
        if len(old)!=expected_total:
            raise ValueError("exposure ledger changed: new cohort not independent")
    catalog=fetch_manifest()
    if len(catalog)!=51 or len(set(catalog)-used_dbs)!=2:
        raise ValueError("51 original physical database source manifest drift")
    bydb={}
    for i,row in enumerate(all_rows):
        q=row.get("question")
        ev=row.get("evidence") or ""
        db=row.get("db_id")
        if (i in old or db not in used_dbs or db=="retail_world"
            or catalog[db]["size"]>MAX_DB_BYTES
            or not isinstance(q,str) or len(q.strip())<20
            or not isinstance(ev,str) or len(q)+len(ev)>6000):
            continue
        bydb.setdefault(db,[]).append(i)
    order=lambda x:sha256((SELECT_SEED+":"+str(x)).encode()).hexdigest()
    dbs=sorted((db for db,ids in bydb.items() if len(ids)>=PER_DB),
               key=order)[:N_DB]
    if len(dbs)!=N_DB:
        raise ValueError("not enough old physical databases with NEW questions")
    source=[]
    for db in dbs:
        for i in sorted(bydb[db],key=order)[:PER_DB]:
            row=all_rows[i]
            source.append({"source_index":i,"db_id":db,
                           "question":row["question"],
                           "evidence":row.get("evidence") or "",
                           "question_sha256":sha256(row["question"].encode()).hexdigest()})
    if len(source)!=N or len({r["source_index"] for r in source})!=N:
        raise ValueError("bad new48 question selection")
    if out.exists():raise FileExistsError("immutable source freeze")
    out.mkdir()
    file=out/"source48.jsonl"
    file.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in source))
    report={"status":"NEW48_FULL711_HISTORY_SOURCE_FROZEN_PRE_QWEN",
            "protocol":PROTOCOL,"train_sha256":TRAIN_SHA,
            "source_sha256":sha(file),"model_digest":DIGEST,
            "n":N,"n_dbs":N_DB,"per_db":PER_DB,"dbs":dbs,
            "excluded_prior_source_indices":len(old),
            "original_49_exposed_database_families_only":True,
            "last_two_protected_physical_train_dbs_unused":True,
            "retail_world_excluded_for_publisher_reference_errors":True,
            "DEV500_unused":True,"HOLDOUT300_unused":True,
            "source_contains_no_gold_SQL":True,
            "per_arm_model_calls":4,"actual_unique_qwen_calls_per_task":6,
            "gate":GATE,"historical_provenance":provenance["prior_sources"]}
    (out/"freeze.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report

def checked_source(source:Path,manifest:Path)->tuple[list[dict],dict]:
    m=json.loads(manifest.read_text())
    if (m.get("status")!="NEW48_FULL711_HISTORY_SOURCE_FROZEN_PRE_QWEN"
        or m.get("protocol")!=PROTOCOL or m.get("model_digest")!=DIGEST
        or m.get("train_sha256")!=TRAIN_SHA or
        m.get("source_sha256")!=sha(source)
        or m.get("n")!=N or m.get("dbs") is None
        or m.get("n_dbs")!=N_DB or m.get("per_db")!=PER_DB
        or m.get("excluded_prior_source_indices")!=711
        or m.get("gate")!=GATE
        or m.get("last_two_protected_physical_train_dbs_unused") is not True
        or m.get("source_contains_no_gold_SQL") is not True
        or m.get("DEV500_unused") is not True
        or m.get("HOLDOUT300_unused") is not True):
        raise ValueError("original source/model/holdout protocol mismatch")
    rows=[json.loads(l) for l in source.read_text().splitlines() if l.strip()]
    if len(rows)!=N or len({r["source_index"] for r in rows})!=N:
        raise ValueError("source missing unique question identities")
    for i,x in enumerate(rows):
        if (x["db_id"]!=m["dbs"][i//PER_DB]
            or x["question_sha256"]!=sha256(x["question"].encode()).hexdigest()
            or any(k in x for k in ("SQL","gold","reference_sql"))):
            raise ValueError("source mismatch or gold leakage")
    return rows,m

def intent_class(question:str)->str:
    q=question.casefold()
    if re.search(r"\b(percentage|percent|ratio|proportion|fraction)\b",q):
        return "ratio"
    if re.search(r"\b(how many|number of|count|distinct|unique)\b",q):
        return "count"
    if re.search(r"\b(highest|lowest|top|maximum|minimum|most|least|first|last)\b",q):
        return "ranking"
    return "relation"

def shape_signature(sql:str)->str:
    """Coarse *nonsemantic* structure key; never treat as execution proof."""
    lowered=re.sub(r"\s+"," ",str(sql).lower())
    ops=[op for op in STRUCTURE_OPS
         if re.search(r"(?<!\w)"+re.escape(op)+r"(?!\w)",lowered)]
    tables=sorted(set(re.findall(
        r"\b(?:from|join)\s+([a-z_][a-z_0-9]*)",lowered)))
    projection=re.search(r"\bselect\s+(.*?)\s+from\b",lowered)
    select=(projection.group(1) if projection else "")[:300]
    select_class=("count" if "count(" in select else
                  "ratio" if "/" in select else
                  "aggregate" if any(x+"(" in select for x in ("sum","avg","min","max")) else
                  "records")
    cols=1+select.count(",") if select else 0
    return json.dumps([select_class,cols,tables,ops],sort_keys=True)

def choose_plurality(candidates:list[dict])->dict:
    if len(candidates)!=4:
        raise ValueError("exactly four candidates for every arm")
    valid=[(i,c["set_sha256"]) for i,c in enumerate(candidates)
           if c["ok"] and isinstance(c["set_sha256"],str)]
    if not valid:
        return {"index":0,"votes":0,"ok":False,"set_sha256":None,
                "bag_sha256":None}
    counter=Counter(sig for _,sig in valid)
    best=max(counter.values())
    chosen_idx=next(i for i,sig in valid if counter[sig]==best)
    selected=candidates[chosen_idx]
    return {"index":chosen_idx,"votes":best,"ok":True,
            "set_sha256":selected["set_sha256"],
            "bag_sha256":selected["bag_sha256"]}

def contrast_user(original_user:str, original_sqls:tuple[str,str], question:str,
                  hypothesis:int)->str:
    role=intent_class(question)
    if hypothesis==0:
        rule={
            "count":"Use a different entity COUNT grain (distinct target entity vs joined rows); consider whether count needs an EXISTS semijoin or GROUP BY instead of counting joined records.",
            "ratio":"Derive a different numerator/denominator entity-set definition; ensure the SELECT projection actually computes the ratio, not only ORDER BY it.",
            "ranking":"Use an alternative top-k/set-extremum formulation (NOT an unsupported group-by); verify ties and exactly which entity is returned.",
            "relation":"Explore an alternative physically defined FK path or direction for the question's requested endpoint; preserve all source predicates.",
        }[role]
    else:
        rule={
            "count":"Explore an alternative relation path with EXISTS or DISTINCT, keeping exact target entity grain and original evidence filters.",
            "ratio":"Re-derive numerator and denominator with explicit fact-level filters, using scalar subqueries or a CTE rather than rephrasing the same SQL.",
            "ranking":"Use a correlated extremum-subquery or window-compatible SQLite formulation; ensure projected name/id follows the requested entity.",
            "relation":"Use a different SELECT answer projection and justified table path while preserving the requested entity, ranking, scope, and explicit evidence.",
        }[role]
    base1,base2=original_sqls
    return (original_user+
        "\n\nTwo SQL hypotheses already generated by this same model; "
        "both are UNVERIFIED and may have the same structural mistake:\n"
        "CANDIDATE_A:\n"+base1[:2600]+"\nCANDIDATE_B:\n"+base2[:2600]+
        "\nSTRUCTURAL COUNTERHYPOTHESIS "+str(hypothesis+1)+" ("+role+"):\n"+rule+
        "\nProduce ONE alternative read-only SQLite SQL SELECT, with a materially "
        "different table path, selected entity grain or aggregation operator if "
        "the actual physical schema permits it. Use ONLY documented columns/tables. "
        "Never invent a join or delete explicit evidence filters merely to differ. "
        "If no valid structural alternative exists, output the best SQL rather than "
        "a fake alternative. SQL only, no analysis.")

def physical_candidates(database:Path,sqls:list[str])->list[dict]:
    out=[]
    for sql in sqls:
        e=execute(database,sql)
        out.append({"ok":e["ok"],"reason":e["reason"],
                    "set_sha256":e["set_sha256"],"bag_sha256":e["bag_sha256"],
                    "shape_signature":shape_signature(sql)})
    return out

def infer(source:Path,manifest:Path,dbroot:Path,shard:int,out:Path)->dict:
    rows,info=checked_source(source,manifest)
    if shard not in range(N_DB):raise ValueError("bad physical database shard")
    db=info["dbs"][shard]
    path=dbroot/db/(db+".sqlite")
    if not path.is_file():raise FileNotFoundError("physical original SQLite missing")
    if _ollama_model_digest("http://127.0.0.1:11434",MODEL)!=DIGEST:
        raise ValueError("original Qwen3.5-9B digest drift")
    catalog=DatabaseCatalog.from_sqlite(path)
    providers={
        s:OllamaProvider(model=MODEL,timeout_s=900,temperature=0.7,
                         seed=s,num_predict=384) for s in BASE_SEEDS
    }
    cases=[]
    for x in rows[shard*PER_DB:(shard+1)*PER_DB]:
        schema,_=select_schema_context(catalog,dbroot,db,x["question"])
        user="Question: "+x["question"]+"\nSchema context:\n"+schema
        if x["evidence"]:user+="\nExternal evidence:\n"+x["evidence"]
        def generate(seed,sys,usr):
            t=time.monotonic()
            sql=_strip_fence(providers[seed].complete_text(system=sys,user=usr))
            return sql,round(time.monotonic()-t,2)
        a1,t1=generate(42,_SYSTEM,user)
        a2,t2=generate(43,_SYSTEM,user)
        a3,t3=generate(44,_SYSTEM,user)
        a4,t4=generate(45,_SYSTEM,user)
        t_user0=contrast_user(user,(a1,a2),x["question"],0)
        t_user1=contrast_user(user,(a1,a2),x["question"],1)
        contrast_system=(_SYSTEM+"\nTreat the prior two candidates as "
            "hypotheses, not as examples to copy. Explore a defensible different "
            "SQL relational structure. Use no hidden reference SQL. SQL only.")
        b3,t5=generate(44,contrast_system,t_user0)
        b4,t6=generate(45,contrast_system,t_user1)
        control_sql=[a1,a2,a3,a4]
        treatment_sql=[a1,a2,b3,b4]
        # Read-only SQLite preflight/votes from generated SQL ONLY; no publisher gold.
        preflight=[
            *physical_candidates(path,[a1,a2,a3,a4]),
            *physical_candidates(path,[b3,b4])
        ]
        c0=preflight[:4]
        c1=[preflight[0],preflight[1],preflight[4],preflight[5]]
        control_choice=choose_plurality(c0)
        treatment_choice=choose_plurality(c1)
        cases.append({
            "source_index":x["source_index"],"db_id":db,
            "question_sha256":x["question_sha256"],
            "control_sql":control_sql,"treatment_sql":treatment_sql,
            "control_preflight":c0,"treatment_preflight":c1,
            "control_choice":control_choice,"treatment_choice":treatment_choice,
            "control_wall_s":round(t1+t2+t3+t4,2),
            "treatment_wall_s":round(t1+t2+t5+t6,2),
            "shared_qwen_calls":2,"control_k":4,"treatment_k":4,
            "unique_qwen_calls":6
        })
        print("GOLD_FREE_QWEN9B_K4_STRUCTURAL_CANDIDATES",
              shard,len(cases),"/",PER_DB,"counterfactual_structures",
              len({c["shape_signature"] for c in c1}),flush=True)
    result={"protocol":PROTOCOL,"status":"ORIGINAL_QWEN9B_CANDIDATES_AND_CHOICES_SEALED_PRE_GOLD",
            "model_digest":DIGEST,"source_sha256":info["source_sha256"],
            "shard":shard,"db_id":db,"n":PER_DB,"cases":cases,
            "no_gold_in_model_or_selector":True,"DEV500_unused":True,
            "HOLDOUT300_unused":True}
    if out.exists():raise FileExistsError("immutable generated SQL selection")
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    return result

def score(source:Path,manifest:Path,dbroot:Path,shard:int,
          generated:Path,train:Path,out:Path)->dict:
    rows,info=checked_source(source,manifest)
    sealed=json.loads(generated.read_text())
    if (sealed.get("protocol")!=PROTOCOL or
        sealed.get("status")!="ORIGINAL_QWEN9B_CANDIDATES_AND_CHOICES_SEALED_PRE_GOLD"
        or sealed.get("source_sha256")!=info["source_sha256"]
        or sealed.get("model_digest")!=DIGEST or sealed.get("shard")!=shard
        or sealed.get("n")!=PER_DB or sealed.get("no_gold_in_model_or_selector") is not True):
        raise ValueError("wrong model and pre-gold selection seal")
    gold=load_checked(train,TRAIN_SHA)
    physical_db=info["dbs"][shard]
    db=dbroot/physical_db/(physical_db+".sqlite")
    scored=[]
    for r,c in zip(rows[shard*PER_DB:(shard+1)*PER_DB],sealed["cases"]):
        if (r["source_index"]!=c["source_index"] or
            r["db_id"]!=c["db_id"] or r["question_sha256"]!=c["question_sha256"]):
            raise ValueError("source identity lost after score rehydration")
        reference=execute(db,gold[r["source_index"]]["SQL"])
        valid=reference["ok"]
        control=c["control_preflight"]
        treat=c["treatment_preflight"]
        def exact(x,key):
            return bool(x["ok"] and x[key]==reference[key]) if valid else None
        c_ex=[exact(x,"set_sha256") for x in control]
        t_ex=[exact(x,"set_sha256") for x in treat]
        c_bag=[exact(x,"bag_sha256") for x in control]
        t_bag=[exact(x,"bag_sha256") for x in treat]
        control_choice=c["control_choice"]
        treat_choice=c["treatment_choice"]
        # Recheck sealed selector has not been altered; gold never influences choice.
        if control_choice!=choose_plurality(control) or treat_choice!=choose_plurality(treat):
            raise ValueError("posthoc selector manipulation")
        scored.append({
            "source_index":r["source_index"],"db_id":r["db_id"],
            "reference_scorable":valid,
            "reference_failure":None if valid else reference["reason"],
            "control_selected_set_EX":c_ex[control_choice["index"]] if valid else None,
            "treatment_selected_set_EX":t_ex[treat_choice["index"]] if valid else None,
            "control_selected_bag_EX":c_bag[control_choice["index"]] if valid else None,
            "treatment_selected_bag_EX":t_bag[treat_choice["index"]] if valid else None,
            "control_oracle_at4":any(c_ex) if valid else None,
            "treatment_oracle_at4":any(t_ex) if valid else None,
            "control_executable":sum(x["ok"] for x in control),
            "treatment_executable":sum(x["ok"] for x in treat),
            "control_unique_structures":len({x["shape_signature"] for x in control}),
            "treatment_unique_structures":len({x["shape_signature"] for x in treat}),
            "control_wall_s":c["control_wall_s"],
            "treatment_wall_s":c["treatment_wall_s"],
            "calls_control":4,"calls_treatment":4,"calls_shared":2
        })
    result={"protocol":PROTOCOL,"status":"PUBLISHER_STRICT_PHYSICAL_SET_BAG_EX_SCORED_POST_SEAL",
            "model_digest":DIGEST,"source_sha256":info["source_sha256"],
            "shard":shard,"db_id":physical_db,"n":PER_DB,
            "cases":scored,"DEV500_unused":True,"HOLDOUT300_unused":True}
    if out.exists():raise FileExistsError("immutable source scorer")
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    return result

def summarize(parts:list[Path],manifest:Path,out:Path)->dict:
    meta=json.loads(manifest.read_text())
    if len(parts)!=N_DB or meta.get("protocol")!=PROTOCOL:
        raise ValueError("all 8 original physical DB score shards required")
    records=[]
    seen=set()
    for p in parts:
        d=json.loads(p.read_text())
        sh=d.get("shard")
        if (sh in seen or not isinstance(sh,int) or sh not in range(N_DB)
            or d.get("status")!="PUBLISHER_STRICT_PHYSICAL_SET_BAG_EX_SCORED_POST_SEAL"
            or d.get("protocol")!=PROTOCOL or d.get("model_digest")!=DIGEST
            or d.get("source_sha256")!=meta["source_sha256"]
            or d.get("db_id")!=meta["dbs"][sh] or d.get("n")!=PER_DB
            or len(d.get("cases",[]))!=PER_DB):
            raise ValueError("wrong physical Qwen or incomplete strict score")
        seen.add(sh)
        records+=d["cases"]
    if len(records)!=N or len({x["source_index"] for x in records})!=N:
        raise ValueError("wrong source union or incomplete question count")
    eligible=[x for x in records if x["reference_scorable"]]
    valid_count=len(eligible)
    c=Counter()
    for x in eligible:
        a=x["control_selected_set_EX"]
        b=x["treatment_selected_set_EX"]
        ao=x["control_oracle_at4"]
        bo=x["treatment_oracle_at4"]
        c["baseline_selected"]+=int(a)
        c["treatment_selected"]+=int(b)
        c["baseline_oracle"]+=int(ao)
        c["treatment_oracle"]+=int(bo)
        c["selected_rescues"]+=int(b and not a)
        c["selected_regressions"]+=int(a and not b)
        c["oracle_rescues"]+=int(bo and not ao)
        c["oracle_regressions"]+=int(ao and not bo)
        c["bag_baseline"]+=int(x["control_selected_bag_EX"])
        c["bag_treatment"]+=int(x["treatment_selected_bag_EX"])
    c["control_executable"] =sum(x["control_executable"] for x in records)
    c["treatment_executable"] =sum(x["treatment_executable"] for x in records)
    ci=None
    groups={db:[int(x["treatment_selected_set_EX"])-int(x["control_selected_set_EX"])
               for x in eligible if x["db_id"]==db] for db in meta["dbs"]}
    if all(groups.values()):
        rng=random.Random(20261119)
        ordered=sorted(groups)
        diffs=[]
        for _ in range(4000):
            chosen=[rng.choice(ordered) for _ in ordered]
            vals=[v for db in chosen for v in groups[db]]
            diffs.append(sum(vals)/len(vals))
        diffs.sort()
        ci={"lower95":diffs[100],"upper95":diffs[3899],
            "mean":sum(sum(v) for v in groups.values())/valid_count,
            "database_clusters":N_DB}
    oracle_net=c["oracle_rescues"]-c["oracle_regressions"]
    selected_net=c["selected_rescues"]-c["selected_regressions"]
    rescue_dbs=len({x["db_id"] for x in eligible if
                    x["treatment_selected_set_EX"] and
                    not x["control_selected_set_EX"]})
    screen=bool(valid_count>=GATE["min_scorable"]
                and oracle_net>=GATE["min_oracle_at4_net"]
                and selected_net>=GATE["min_selected_net"]
                and c["selected_regressions"]<=GATE["max_selected_regressions"]
                and rescue_dbs>=GATE["min_rescue_dbs"])
    result={"protocol":PROTOCOL,
            "status":"COMPLETE_REAL_QWEN9B_TRAIN48_EQUAL_K4_STRUCTURAL_COUNTERHYPOTHESIS",
            "n":N,"scorable":valid_count,"unscorable":N-valid_count,
            "censored_by_database":{db:sum(not x["reference_scorable"]
                             for x in records if x["db_id"]==db)
                            for db in meta["dbs"]},
            "original_qwen_digest":DIGEST,
            "source_sha256":meta["source_sha256"],
            "physical_dbs":N_DB,
            "counts":dict(c),"oracle_net":oracle_net,
            "selected_net":selected_net,
            "selected_rescue_database_count":rescue_dbs,
            "selected_DB_cluster_CI95":ci,
            "mean_structures_control":round(sum(x["control_unique_structures"] for x in records)/N,3),
            "mean_structures_treatment":round(sum(x["treatment_unique_structures"] for x in records)/N,3),
            "control_model_calls":4*N,
            "treatment_model_calls":4*N,
            "actual_unique_original_model_calls":6*N,
            "total_wall_control_s":round(sum(x["control_wall_s"] for x in records),1),
            "total_wall_treatment_s":round(sum(x["treatment_wall_s"] for x in records),1),
            "gate":GATE,"positive_screen_only":screen,"promotion_allowed":False,
            "source_new_question_not_new_database":True,
            "DEV500_used":False,"HOLDOUT300_used":False,
            "last_two_protected_original_TRAIN_DBS_used":False,
            "no_training_or_external_teacher":True,
            "candidate_oracle_not_deployable":True,
            "selector":"GOLD_BLIND_READONLY_SQLITE_RESULT_SET_PLURALITY_EARLIEST_VALID_ON_TIE"}
    if out.exists():raise FileExistsError("immutable original 48 scoreboard")
    out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("freeze","infer","score","aggregate"))
    for key in ("train","history","plan","probe","role","out","source",
                "manifest","dbroot","generated"):
        p.add_argument("--"+key,type=Path)
    p.add_argument("--shard",type=int,default=0)
    p.add_argument("--parts",type=Path,nargs="+")
    a=p.parse_args()
    if a.mode=="freeze":
        d=freeze(a.train,a.history,a.plan,a.probe,a.role,a.out)
    elif a.mode=="infer":
        d=infer(a.source,a.manifest,a.dbroot,a.shard,a.out)
    elif a.mode=="score":
        d=score(a.source,a.manifest,a.dbroot,a.shard,a.generated,a.train,a.out)
    else:
        d=summarize(a.parts,a.manifest,a.out)
    print("FIXED_QWEN35_STRUCTURAL_K4_SCREEN",json.dumps(
        {k:v for k,v in d.items() if k not in ("cases","historical_provenance")},
        sort_keys=True),flush=True)

if __name__=="__main__":
    main()
