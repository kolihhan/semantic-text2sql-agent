"""Original real fixed Qwen9B BIRD TRAIN-VAL64 *source-only* relational witness census.

This stops BEFORE publisher gold SQL/EX evaluation unless independent
relation/entity coverage is meaningful across multiple physical databases.
The reference message[2] and original model .direct/.guarded score labels
are NEVER opened by this script. Runs no model and cannot claim EX gain.
"""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
import argparse
import json
from pathlib import Path

from evaluation.entity_count_relational_witness import propose_entity_count_witness

SOURCE_SHA="cc112305bc9de33a03cea704f2edecfdfb45e62cc953ae498cea27fbed93b506"
MODEL_DIGEST="56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
DBS=("movie_3","music_tracker","superstore","world")
PROTOCOL="BIRD-TRAIN64-QWEN35-9B-REAL-PHYSICAL-ENTITY-COUNT-WITNESS-NOGOLD-V1"
PREDECLARED_MIN_GUARDED_CANDIDATES=4
PREDECLARED_MIN_DATABASES=2


def read_source(path:Path)->list[dict]:
    raw=path.read_bytes()
    if sha256(raw).hexdigest()!=SOURCE_SHA:
        raise ValueError("original TRAIN VAL64 source hash drift")
    rows=[json.loads(x) for x in raw.decode().splitlines() if x.strip()]
    if len(rows)!=64 or len({x["source_index"] for x in rows})!=64:
        raise ValueError("original source 64 row identities drift")
    if tuple(sorted({x["source_db_id"] for x in rows}))!=DBS:
        raise ValueError("original four physical databases drift")
    for x in rows:
        if x["messages"][1]["role"]!="user" or sha256(
            x["messages"][1]["content"].encode()
        ).hexdigest()!=x["input_sha256"]:
            raise ValueError("original model user prompt SHA drift")
    return rows


def original_predictions(path:Path)->list[dict]:
    data=json.loads(path.read_text())
    if (data.get("status")!="complete" or data.get("n")!=64
        or data.get("source_validation_sha256")!=SOURCE_SHA
        or data.get("model_digest")!=MODEL_DIGEST
        or data.get("direct_ex")!=27 or data.get("guarded_ex")!=31
        or len(data.get("cases",[]))!=64):
        raise ValueError("original fixed Qwen9B baseline identity drift")
    return data["cases"]


def physical_worker(source:Path,model:Path,db:Path,shard:int)->dict:
    if shard not in range(4) or db.name!=DBS[shard]+".sqlite" or not db.is_file():
        raise ValueError("wrong original physical database for source worker")
    rows=read_source(source)
    models=original_predictions(model)
    indexed={m["source_index"]:m for m in models}
    if len(indexed)!=64 or set(indexed)!={r["source_index"] for r in rows}:
        raise ValueError("incomplete source/model identity")
    chosen=[r for r in rows if r["source_db_id"]==DBS[shard]]
    if len(chosen)!=16:
        raise ValueError("expected 16 source questions on physical DB")
    cases=[]
    for r in chosen:
        m=indexed[r["source_index"]]
        prompt=r["messages"][1]["content"]
        if m["db_id"]!=DBS[shard] or m["prompt_sha256"]!=sha256(prompt.encode()).hexdigest():
            raise ValueError("model case/prompt SHA not source aligned")
        row={"source_index":r["source_index"],"db_id":DBS[shard]}
        for arm in ("direct","guarded"):
            sql=m[arm+"_sql"]
            w=propose_entity_count_witness(db,prompt,sql)
            row[arm]={
                "changed":w.changed,"reason":w.reason,
                "model_sql_sha256":sha256(sql.encode()).hexdigest(),
                "alternate_sql_sha256":sha256(w.sql.encode()).hexdigest(),
                "physical_count_rows":w.original_row_count,
                "physical_count_distinct_entities":w.distinct_entity_count,
                "entity_table":w.entity_table,
                "entity_column":w.entity_column,
            }
        cases.append(row)
        print("REAL_SQL_PHYSICAL_ENTITY_WITNESS",
              DBS[shard],len(cases),"/16",
              "guarded",row["guarded"]["changed"],flush=True)
    return {
        "status":"PHYSICAL_SOURCE_ONLY_WORKER_DONE",
        "protocol":PROTOCOL,"source_SHA256":SOURCE_SHA,
        "fixed_model_digest":MODEL_DIGEST,"shard":shard,
        "physical_db":DBS[shard],"n":len(cases),
        "model_calls":0,"training_steps":0,
        "publisher_reference_SQL_read":False,
        "model_score_labels_used":False,
        "DEV500_used":False,"HOLDOUT300_used":False,
        "cases":cases,
    }


def aggregate(parts:list[dict],source:Path)->dict:
    original=read_source(source)
    if len(parts)!=4 or sorted(p["shard"] for p in parts)!=list(range(4)):
        raise ValueError("all four original physical workers required")
    all_rows=[]
    for p in parts:
        if (p.get("status")!="PHYSICAL_SOURCE_ONLY_WORKER_DONE"
            or p.get("protocol")!=PROTOCOL or p.get("source_SHA256")!=SOURCE_SHA
            or p.get("fixed_model_digest")!=MODEL_DIGEST
            or p.get("n")!=16 or len(p.get("cases",[]))!=16
            or p.get("physical_db")!=DBS[p["shard"]]
            or p.get("model_calls")!=0 or p.get("training_steps")!=0
            or p.get("publisher_reference_SQL_read") is not False
            or p.get("model_score_labels_used") is not False
            or p.get("DEV500_used") is not False
            or p.get("HOLDOUT300_used") is not False):
            raise ValueError("incomplete or contaminated source worker")
        all_rows.extend(p["cases"])
    if len(all_rows)!=64 or {x["source_index"] for x in all_rows}!={r["source_index"] for r in original}:
        raise ValueError("not all original Qwen TRAIN64 source cases present")
    arms={}
    for arm in ("direct","guarded"):
        changed=[x for x in all_rows if x[arm]["changed"]]
        reasons=Counter(x[arm]["reason"] for x in all_rows)
        arms[arm]={
            "changed_count":len(changed),
            "changed_db_count":len({x["db_id"] for x in changed}),
            "changed_db_ids":sorted({x["db_id"] for x in changed}),
            "changed_original_indices":sorted(x["source_index"] for x in changed),
            "reasons":dict(sorted(reasons.items())),
            "sample_count":64,
        }
    g=arms["guarded"]
    positive=(g["changed_count"]>=PREDECLARED_MIN_GUARDED_CANDIDATES
              and g["changed_db_count"]>=PREDECLARED_MIN_DATABASES)
    return {
        "status":"ORIGINAL_QWEN9B_SOURCE_ONLY_ENTITY_KEY_CARDINALITY_SCREEN_COMPLETE",
        "protocol":PROTOCOL,"source_SHA256":SOURCE_SHA,
        "fixed_model_digest":MODEL_DIGEST,
        "n":64,"physical_dbs":list(DBS),
        "direct":arms["direct"],"guarded":g,
        "feasibility_positive":positive,
        "predeclared_min_guarded_changes":PREDECLARED_MIN_GUARDED_CANDIDATES,
        "predeclared_min_changed_physical_dbs":PREDECLARED_MIN_DATABASES,
        "do_not_evaluate_publisher_labels_if_gate_fails":not positive,
        "real_model_calls":0,"training_steps":0,
        "publisher_reference_SQL_read":False,
        "model_score_labels_used":False,
        "DEV500_used":False,"HOLDOUT300_used":False,
        "claim_boundary":"Only a physical COUNT(*) vs COUNT(DISTINCT entity-id) counterexample and potential SQL alternative, not its semantic correctness or EX. Public BIRD TRAIN real-model historical outputs; never tune on their original publisher EX. If too few triggers across DBs, do not spend new Qwen calls or inspect source labels.",
    }


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("worker","aggregate"))
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--model",type=Path)
    p.add_argument("--database",type=Path)
    p.add_argument("--shard",type=int)
    p.add_argument("--inputs",type=Path,nargs="+")
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():
        raise FileExistsError("do not overwrite immutable original source evidence")
    if args.mode=="worker":
        if args.model is None or args.database is None or args.shard is None:
            p.error("worker requires original model, DB, source shard")
        result=physical_worker(args.source,args.model,args.database,args.shard)
    else:
        if not args.inputs:
            p.error("aggregate requires four workers")
        result=aggregate([json.loads(f.read_text()) for f in args.inputs],args.source)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("QWEN9B_ENTITY_COUNT_REAL_PHYSICAL_SCREEN",
          json.dumps({k:v for k,v in result.items() if k!="cases"},sort_keys=True),
          flush=True)


if __name__=="__main__":
    main()
