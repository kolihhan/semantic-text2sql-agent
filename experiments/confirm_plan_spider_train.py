"""A0 vs frozen A5 on 100 SCHEMA-DISJOINT Spider TRAIN databases.

This is an independently chosen source shift from prior synthetic 4 schemas
and Spider DEV 20 schemas, with one question per unique Spider train schema.
It is NOT a pretrained-model holdout: public Spider TRAIN may be in XiYan
pretraining. No reference SQL influences cohort selection, planning, SQL
generation or any model call. All 50 SQL candidates are generated BEFORE
opening that shard's reference SQL for scoring. Same model/prompt except
A5's original 4-field gold-blind semantic plan.

Primary correctness proxy: exact row-multiset SQLite execution equality;
ordering, empties, execution failures and unscorable reference labels recorded.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from evaluation.schema_context import select_schema_context
from experiments.counterfactual_sql_plan_v1 import parse_plan,prompt
from experiments.spider_grounding_shift import FixedSeedOllamaProvider,MODEL,frozen_selection
from experiments.xiyan_dgsl_frozen100 import generate_xiyan,resolve_database
from experiments.xiyan_mschema_frozen100 import build_mschema
from experiments.audit_spider_exec_strict import score_readonly_pair
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.execution import execute_readonly,SQLExecutionError

SEED="xiyan-a5-100-spider-train-db-disjoint-from-previous-dev-20261113-v1"


def select_cases(train:list[dict],dev:list[dict],*,seed:str=SEED)->list[dict]:
    if not isinstance(train,list) or not isinstance(dev,list):
        raise ValueError("Spider train/dev must be JSON arrays")
    dev_dbs={str(x["db_id"]) for x in dev}
    candidates=[x for x in train if str(x.get("db_id") or "") not in dev_dbs]
    selection=frozen_selection(candidates,count=100,seed=seed)
    db_ids=[row["database_id"] for row in selection]
    if len(set(db_ids))!=100 or set(db_ids)&dev_dbs:
        raise ValueError("not exactly 100 independent training schemas outside prior DEV")
    if any("query" in row or "sql" in row for row in selection):
        raise ValueError("gold accidentally used in source selection")
    return selection


def run_shard(*,train_json:Path,dev_json:Path,db_root:Path,shard:int,
              model:str=MODEL)->dict:
    if shard not in range(4):
        raise ValueError("need 4x25 predeclared shards")
    train=json.loads(train_json.read_text(encoding="utf-8"))
    dev=json.loads(dev_json.read_text(encoding="utf-8"))
    selection=select_cases(train,dev)
    cohort_sha=sha256("\n".join(row["case_id"] for row in selection).encode()).hexdigest()
    chosen=selection[shard::4]
    if len(chosen)!=25 or len({row["database_id"] for row in chosen})!=25:
        raise ValueError("25 different database identities required")
    provider=FixedSeedOllamaProvider(model=model,timeout_s=900)
    generated=[]
    for index,row in enumerate(chosen,1):
        started=perf_counter()
        db=resolve_database(db_root,row["database_id"])
        catalog=DatabaseCatalog.from_sqlite(db)
        _,selected=select_schema_context(
            catalog,db_root,row["database_id"],row["question"])
        schema=build_mschema(
            catalog=catalog,database_id=row["database_id"],
            selected_tables=selected["selected_tables"],descriptions={})
        original=generate_xiyan(
            question=row["question"],schema_context=schema,evidence=None,
            provider=provider)
        plan_raw=provider.complete_text(
            system="Only JSON plans, no SQL.",
            user=prompt(row["question"],schema))
        plan=parse_plan(plan_raw)
        if plan is None:
            proposed=original
            disposition="invalid_plan_baseline_preserved"
        else:
            context=(schema+"\nProposed question-semantic plan (NOT a fact; "
                     "verify every assumption against the original question):\n"
                     +json.dumps(plan,sort_keys=True,ensure_ascii=False))
            proposed=generate_xiyan(
                question=row["question"],schema_context=context,evidence=None,
                provider=provider)
            disposition="planner_valid_new_candidate"
        generated.append({
            **row,"sql":{"a0":original,"a5":proposed},
            "planner_valid":plan is not None,
            "planner_action":disposition,
            "plan_sha256":sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest() if plan else None,
            "schema_sha256":sha256(schema.encode()).hexdigest(),
            "latency_s":round(perf_counter()-started,2),
            "labels_seen_before_model_calls":False,
        })
        print(f"SPIDER_TRAIN_A5_GENERATED shard={shard} {index}/25 db={row['database_id']}",flush=True)
    # Reference SQL accessed *only after* all generated candidates exist.
    scored=[]
    unscorable=[]
    for row in generated:
        reference=train[row["row_index"]]["query"]
        db=resolve_database(db_root,row["database_id"])
        try:
            gold=execute_readonly(db,reference,max_rows=50_000)
            if gold.truncated:
                raise ValueError("reference_truncated")
        except (SQLExecutionError,ValueError,TypeError) as exc:
            unscorable.append({"case_id":row["case_id"],"reason":type(exc).__name__})
            continue
        outcomes={
            arm:score_readonly_pair(db,sql,reference)
            for arm,sql in row["sql"].items()
        }
        scored.append({**row,"strict":outcomes})
    return {
        "schema":"spider-train-database-shift-a0-a5-shard-v1",
        "source":"Spider TRAIN public (possible pretrained overlap)",
        "target":"100 different database identities excluded from Spider DEV",
        "metric":"SQLite exact row-multiset execution",
        "seed":SEED,"model":model,
        "cohort_sha256":cohort_sha,"shard":shard,
        "requested":25,"scored":len(scored),
        "unscorable":unscorable,"cases":scored,
    }


def aggregate(shards:list[dict])->dict:
    if len(shards)!=4 or sorted(s.get("shard") for s in shards)!=[0,1,2,3]:
        raise ValueError("all distinct four shards required")
    if len({s["cohort_sha256"] for s in shards})!=1 or len({s["model"] for s in shards})!=1:
        raise ValueError("model/source cohort mismatch")
    if any(s["schema"]!="spider-train-database-shift-a0-a5-shard-v1"
           or s["requested"]!=25 for s in shards):
        raise ValueError("protocol mismatch")
    rows=[r for shard in shards for r in shard["cases"]]
    bad=[r for shard in shards for r in shard["unscorable"]]
    if len(rows)+len(bad)!=100 or len(rows)<90:
        raise ValueError("insufficient valid paired reference SQL for 100-case cohort")
    dbs=[r["database_id"] for r in rows]+[r["case_id"] for r in bad]
    if len({r["database_id"] for r in rows})!=len(rows):
        raise ValueError("duplicate database identities in scored sample")
    a0=sum(r["strict"]["a0"]["bag_equal"] for r in rows)
    a5=sum(r["strict"]["a5"]["bag_equal"] for r in rows)
    wins=sum(not r["strict"]["a0"]["bag_equal"] and r["strict"]["a5"]["bag_equal"] for r in rows)
    losses=sum(r["strict"]["a0"]["bag_equal"] and not r["strict"]["a5"]["bag_equal"] for r in rows)
    if a5-a0!=wins-losses:raise AssertionError("paired arithmetic drift")
    return {
        "schema":"spider-train-a5-database-shift-aggregate-v1",
        "requested":100,"sample_size":len(rows),"unscorable":bad,
        "database_count":len(rows),"cohort_sha256":shards[0]["cohort_sha256"],
        "model":shards[0]["model"],"a0_bag_correct":a0,"a5_bag_correct":a5,
        "wrong_to_correct":wins,"correct_to_wrong":losses,"net_gain":wins-losses,
        "planner_valid":sum(r["planner_valid"] for r in rows),
        "candidate_changed":sum(r["sql"]["a0"].strip()!=r["sql"]["a5"].strip() for r in rows),
        "both_empty":sum(r["strict"]["a0"]["both_empty"] for r in rows),
        "literal_ast_equal_a0":sum(r["strict"]["a0"]["exact_ast_equal"] for r in rows),
        "literal_ast_equal_a5":sum(r["strict"]["a5"]["exact_ast_equal"] for r in rows),
        "caveats":[
            "Spider TRAIN is public model-training data, NOT an unseen model benchmark",
            "100 databases are different from previous Spider DEV and synthetic experiment",
            "Prompt/hypothesis frozen after seeing previous synthetic 48-case result",
            "Exactly one source question per database; unstable database-level estimates",
            "SQLite exact bag equality is not full semantic correctness",
            "Only future independent BIRD/real-user correctness can warrant deployment",
        ],
        "cases":sorted(rows,key=lambda r:r["case_id"]),
    }


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path)
    p.add_argument("--dev",type=Path)
    p.add_argument("--database-root",type=Path)
    p.add_argument("--shard-index",type=int)
    p.add_argument("--model",default=MODEL)
    p.add_argument("--inputs",nargs="+",type=Path)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("refuse overwriting prior result")
    if a.inputs:
        report=aggregate([json.loads(f.read_text()) for f in a.inputs])
        print("SPIDER_TRAIN_A5_FINAL",json.dumps({
            k:v for k,v in report.items() if k!="cases"},sort_keys=True),flush=True)
    else:
        if any(v is None for v in (a.source,a.dev,a.database_root,a.shard_index)):
            p.error("need --source --dev --database-root --shard-index")
        report=run_shard(train_json=a.source,dev_json=a.dev,
                         db_root=a.database_root,shard=a.shard_index,model=a.model)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n")


if __name__=="__main__":
    main()
