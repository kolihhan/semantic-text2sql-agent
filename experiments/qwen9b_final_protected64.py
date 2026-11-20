"""Real fixed Qwen3.5-9B BIRD TRAIN96: physical-schema-checked relation plan before SQL.

Hypothesis: Qwen solves multi-relation query construction more accurately when
it first names relevant real tables, join columns, answer unit, predicates and
aggregation, *then* produces SQL. This is a new generative candidate (not a
posthoc literal/COUNT tweak), no teachers, SQL LoRA, or base-model changes.

One original direct SQL vs one plan->SQL, identical base model/SQL system and
decoding. Treatment costs an extra plan-model call: this is NOT equal-call
compute; explicitly report it. Gold is absent from frozen model prompts and
NEVER read until generated SQL artifacts have been uploaded/locked.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
import time

from evaluation.run_bird import _ollama_model_digest
from evaluation.schema_context import select_schema_context
from scripts.prepare_birdtrain_qwen9b_sft_source import (
    TRAIN_SHA, fetch_manifest, fetch_database, load_checked
)
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.direct import _SYSTEM, _strip_fence
from semantic_sql.providers import OllamaProvider

MODEL="qwen3.5:9b-q4_K_M"
DIGEST="56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
PROTO="FINAL-QWEN35-9B-PROTECTED2DB64-GROUNDED-PLAN-V1"
LEDGER_SHA="c9f1098a81541a393dfad19bc4b4600e6c421902149375c830c7673f600c632d"
PREV_TRAIN_SHA="b6851fb0f2b5027833aff45a0b5e01326359b245be1c812edc66986d0df1d402"
PREV_VAL_SHA="cc112305bc9de33a03cea704f2edecfdfb45e62cc953ae498cea27fbed93b506"
PREV_FAN_SHA="62eb19b49d43f5e40a5fc2c79d095b9a6294caea781aeebf6b71cda57df5c04c"
SEED="fixed-qwen9b-grounded-relation-plan-TRAIN96-prospective-20261118"
DBS=2
PER_DB=32
N=DBS*PER_DB
MAX_DB_BYTES=128*1024*1024
SQL_SECONDS=8.
ROWS_CAP=50000
PLAN_KEYS={"tables","join_keys","answer_unit","aggregation","filters","ordering"}
GATE={"min_net_correct":2,"min_rescue_databases":2,"max_regressions":2,
      "min_scorable":56}
PLAN_SYSTEM=("Produce ONLY JSON, not SQL. Work out the relational meaning of a "
             "user's SQLite question. Do not invent tables or columns. For unknown "
             "relationships return [] and for unknown answer unit say UNKNOWN.")
PLAN_REQUEST="""Before SQL, produce a small relation-level plan. Output exactly one JSON
object with keys:
"tables": list of relevant existing physical table names;
"join_keys": list of [left "table.column", right "table.column"] pairs;
"answer_unit": short noun/column describing what one answer row represents;
"aggregation": a short precise expression of COUNT/DISTINCT, SUM, AVG, MIN,
MAX, GROUP BY, or NONE, including the real entity grain and denominator;
"filters": list of short AND/OR/NOT clauses in plain English, including
literal values or numeric thresholds from the question;
"ordering": short rule for top/bottom, ranking, ties, or NONE.
Only use physical schema columns in join_keys. Do not assume an unknown FK.
Preserve who performs vs receives a relation and any numeric denominators.
Use UNKNOWN/[] when unsupported. Be concise and do not output SQL.

"""
SQL_PLAN_PREAMBLE=(
    "\nSource-schema-checked intermediate query plan (tentative; the user "
    "question and supplied schema remain authoritative). Translate its "
    "answer grain, relation direction, filters and aggregation into one "
    "valid read-only SQLite SELECT. Avoid assuming row count equals "
    "entity count after a many-side JOIN. If any part contradicts the "
    "question, trust the question instead of the draft plan:\n"
)


def checked_source(path:Path)->list[dict]:
    return load_checked(path,TRAIN_SHA)


def previous_ids(path:Path,digest:str)->set[int]:
    if sha256(path.read_bytes()).hexdigest()!=digest:
        raise ValueError("historical source SHA mismatch")
    rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    return {int(x["source_index"]) for x in rows}


def freeze(trainfile: Path, ledgerfile: Path, _oldtrain: Path,
           _oldval: Path, _fanout: Path, out: Path) -> dict:
    """FINAL source-frozen 2-DB evaluation. Never select by SQL or correctness."""
    train=checked_source(trainfile)
    if sha256(ledgerfile.read_bytes()).hexdigest()!=LEDGER_SHA:
        raise ValueError("original DB ledger SHA drift")
    ledger=json.loads(ledgerfile.read_text())
    protected=json.loads(Path(
        "manifests/bird-train-last-two-db-protected-holdout-20261118.json"
    ).read_text())
    records=protected["holdout_database_records"]
    if (protected["status"]!="SEALED_BEFORE_ANY_ORIGINAL_QWEN_EVALUATION"
        or protected["source_frozen_manifest_SHA256"]!=LEDGER_SHA
        or protected["original_BIRD_TRAIN_source_SHA256"]!=TRAIN_SHA
        or protected["verified_prior_source_question_ids"]!=425
        or len(records)!=2 or any(len(x["protected_original_source_indices"])!=32 for x in records)
        or {x["db_id"] for x in records}
           != {x["db_id"] for x in ledger["unexposed_database_summaries"]}
        or any(x["physical_SQLite_size_bytes"]>MAX_DB_BYTES for x in records)):
        raise ValueError("sealed previously unexposed physical DB contract mismatch")
    rows=[]
    for item in records:
        db=item["db_id"]
        for ix in item["protected_original_source_indices"]:
            if not isinstance(ix,int) or ix<0 or ix>=len(train):
                raise ValueError("invalid protected source index")
            source=train[ix]
            question=source.get("question")
            evidence=source.get("evidence") or ""
            if (source.get("db_id")!=db or not isinstance(question,str)
                or not question.strip() or not isinstance(evidence,str)):
                raise ValueError("protected source question changed")
            rows.append({
                "source_index":ix,"db_id":db,"question":question,
                "evidence":evidence,
                "question_sha256":sha256(question.encode()).hexdigest()
            })
    dbs=[r["db_id"] for r in records]
    if len(rows)!=N or len({r["source_index"] for r in rows})!=N:
        raise ValueError("incomplete source-frozen protected cohort")
    out.mkdir(parents=True,exist_ok=False)
    path=out/"questions.jsonl"
    path.write_text("".join(
        json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\\n" for r in rows
    ),encoding="utf-8")
    report={
        "status":"FROZEN_SOURCE_NO_GOLD_BEFORE_REAL_QWEN_GENERATION",
        "protocol":PROTO,"model_digest":DIGEST,
        "original_BIRD_TRAIN_sha256":TRAIN_SHA,
        "original_source_sha256":sha256(path.read_bytes()).hexdigest(),
        "source_n":N,"db_n":DBS,"per_db":PER_DB,"source_dbs":dbs,
        "historically_exposed_question_ids_excluded":425,
        "protected_2_ood_train_DBs_used":True,
        "DEV500_used":False,"HOLDOUT300_used":False,
        "publisher_SQL_in_frozen_questions":False,
        "protected_original_source_index_count":N,
        "source_selection_presealed_20261118":True,
        "planned_model_calls_max":N*3,
        "experimental_compute":"1 original Qwen direct SQL versus Qwen relation plan then Qwen SQL; 2x to 3x total calls per task",
        "gate":GATE
    }
    (out/"freeze.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\\n")
    return report

def checked_rows(path:Path,meta:Path)->tuple[list[dict],dict]:
    m=json.loads(meta.read_text())
    if (m.get("status")!="FROZEN_SOURCE_NO_GOLD_BEFORE_REAL_QWEN_GENERATION"
        or m.get("protocol")!=PROTO or m.get("model_digest")!=DIGEST
        or m.get("original_BIRD_TRAIN_sha256")!=TRAIN_SHA
        or m.get("original_source_sha256")!=sha256(path.read_bytes()).hexdigest()
        or m.get("source_n")!=N or m.get("db_n")!=DBS or m.get("gate")!=GATE
        or m.get("protected_2_ood_train_DBs_used") is not True
        or m.get("HOLDOUT300_used") is not False):
        raise ValueError("original Qwen source/model/gate frozen manifest drift")
    rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    if (len(rows)!=N or len({r["source_index"] for r in rows})!=N
        or len({r["db_id"] for r in rows})!=DBS):
        raise ValueError("partial source cohort")
    for i,r in enumerate(rows):
        if r["question_sha256"]!=sha256(r["question"].encode()).hexdigest():
            raise ValueError("question provenance drift")
        if r["db_id"]!=m["source_dbs"][i//PER_DB]:
            raise ValueError("not exactly one original DB per worker")
        if "SQL" in r or "gold" in r or "reference" in r:
            raise ValueError("publisher reference accidentally enters Qwen prompt source")
    return rows,m


def validate_plan(text:str,catalog:DatabaseCatalog)->dict|None:
    # Only validated real physical tables/columns may enter the plan->SQL.
    try:
        item=json.loads(text.strip())
    except (ValueError,TypeError):
        return None
    if not isinstance(item,dict) or set(item)!=PLAN_KEYS:
        return None
    tables=item["tables"]
    if (not isinstance(tables,list) or len(tables)>6 or len(set(map(str,tables)))!=len(tables)
        or any(not isinstance(t,str) or t not in catalog.tables for t in tables)):
        return None
    cols={f"{t}.{col}" for t,info in catalog.tables.items() for col in info.columns}
    joins=item["join_keys"]
    if not isinstance(joins,list) or len(joins)>8:
        return None
    for pair in joins:
        if (not isinstance(pair,list) or len(pair)!=2
            or any(not isinstance(x,str) or x not in cols for x in pair)
            or any(x.split(".",1)[0] not in tables for x in pair)):
            return None
    filters=item["filters"]
    if (not isinstance(filters,list) or len(filters)>8
        or any(not isinstance(f,str) or not f.strip() or len(f)>190 for f in filters)):
        return None
    for key in ("answer_unit","aggregation","ordering"):
        value=item[key]
        if not isinstance(value,str) or not value.strip() or len(value)>190:
            return None
    # Avoid treating an empty, hallucinated or tautological plan as evidence.
    if not tables:
        return None
    return item


def infer(source:Path,meta:Path,shard:int,dbroot:Path,out:Path)->dict:
    rows,m=checked_rows(source,meta)
    if shard not in range(DBS):
        raise ValueError("invalid physical DB shard")
    chosen=rows[shard*PER_DB:(shard+1)*PER_DB]
    db=chosen[0]["db_id"]
    dbpath=dbroot/db/f"{db}.sqlite"
    if not dbpath.is_file():
        raise FileNotFoundError("source original database missing")
    if _ollama_model_digest("http://127.0.0.1:11434",MODEL)!=DIGEST:
        raise ValueError("fixed Qwen3.5-9B digest changed")
    catalog=DatabaseCatalog.from_sqlite(dbpath)
    provider=OllamaProvider(model=MODEL,timeout_s=900,temperature=0.,seed=42,num_predict=512)
    cases=[]
    for row in chosen:
        schema,_=select_schema_context(catalog,dbroot,db,row["question"])
        user=f"Question: {row['question']}\nSchema context:\n{schema}"
        if row["evidence"]:
            user+=f"\nExternal evidence:\n{row['evidence']}"
        started=time.monotonic()
        base=_strip_fence(provider.complete_text(system=_SYSTEM,user=user))
        base_t=time.monotonic()-started
        raw=provider.complete_text(system=PLAN_SYSTEM,user=PLAN_REQUEST+user)
        plan=validate_plan(raw,catalog)
        if plan is None:
            treatment=base
            extra_sql_calls=0
            treat_t=0.
        else:
            start_sql=time.monotonic()
            treatment=_strip_fence(provider.complete_text(
                system=_SYSTEM,
                user=user+SQL_PLAN_PREAMBLE+json.dumps(plan,ensure_ascii=False,sort_keys=True)
            ))
            extra_sql_calls=1
            treat_t=time.monotonic()-start_sql
        cases.append({
            "source_index":row["source_index"],"db_id":db,
            "question_sha256":row["question_sha256"],
            "base_SQL":base,"plan_SQL":treatment,
            "plan_valid":plan is not None,
            "plan_sha256":sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest() if plan else None,
            "SQL_changed":base!=treatment,
            "qwen_calls":2+extra_sql_calls,
            "baseline_generation_seconds":round(base_t,2),
            "plan_plus_generation_seconds":round(time.monotonic()-started-base_t,2),
            "final_SQL_generation_seconds":round(treat_t,2)
        })
        print("GROUNDED_PLAN_QWEN9B_REAL_SQL",db,len(cases),"/",PER_DB,
              "valid_plan",plan is not None,"calls",2+extra_sql_calls,flush=True)
    result={"status":"COMPLETE_GOLD_BLIND_REAL_QWEN9B_SQL_GENERATION",
            "protocol":PROTO,"source_sha256":m["original_source_sha256"],
            "model_digest":DIGEST,"shard":shard,"db_id":db,"n":PER_DB,
            "total_model_calls":sum(x["qwen_calls"] for x in cases),
            "DEV500_used":False,"HOLDOUT300_used":False,
            "gold_sql_used_to_generate":False,"cases":cases}
    if out.exists():
        raise FileExistsError("no overwriting actual model generations")
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    return result


def sql_result(db:Path,sql:str)->dict:
    # Closed read-only original SQLite, single SELECT/CTE and a hard progress
    # timer; canonical SET and BAG using historical scoring normalization.
    if not isinstance(sql,str) or not re.match(r"(?is)^\s*(SELECT|WITH)\b",sql):
        return {"valid":False,"reason":"not_select","set":None,"bag":None}
    stop=time.monotonic()+SQL_SECONDS
    conn=None
    try:
        conn=sqlite3.connect(db.resolve().as_uri()+"?mode=ro",uri=True)
        conn.execute("PRAGMA query_only=ON")
        conn.set_progress_handler(lambda:int(time.monotonic()>stop),2000)
        # SQLite single statement API rejects multi-statement injection.
        cursor=conn.execute(sql)
        rows=cursor.fetchmany(ROWS_CAP+1)
        if len(rows)>ROWS_CAP:
            raise ValueError("row_cap")
        canon=[json.dumps(list(row),ensure_ascii=False,sort_keys=True,default=str,
                          separators=(",",":")) for row in rows]
        ss=sha256(json.dumps(sorted(set(canon))).encode()).hexdigest()
        bb=sha256(json.dumps(sorted(Counter(canon).items())).encode()).hexdigest()
        return {"valid":True,"reason":"ok","set":ss,"bag":bb}
    except (sqlite3.Error,ValueError,OSError) as exc:
        return {"valid":False,"reason":"timeout" if time.monotonic()>stop else type(exc).__name__,
                "set":None,"bag":None}
    finally:
        if conn is not None:
            conn.set_progress_handler(None,0)
            conn.close()


def score(source:Path,meta:Path,generations:Path,train_path:Path,
          shard:int,dbroot:Path,out:Path)->dict:
    rows,m=checked_rows(source,meta)
    pred=json.loads(generations.read_text())
    if (pred.get("status")!="COMPLETE_GOLD_BLIND_REAL_QWEN9B_SQL_GENERATION"
        or pred.get("protocol")!=PROTO
        or pred.get("source_sha256")!=m["original_source_sha256"]
        or pred.get("model_digest")!=DIGEST
        or pred.get("shard")!=shard or pred.get("n")!=PER_DB
        or len(pred.get("cases",[]))!=PER_DB
        or pred.get("gold_sql_used_to_generate") is not False):
        raise ValueError("Qwen9B actual SQL generation not frozen before publisher labels")
    actual=checked_source(train_path)
    result=[]
    cases=rows[shard*PER_DB:(shard+1)*PER_DB]
    for row,generated in zip(cases,pred["cases"]):
        if (generated["source_index"]!=row["source_index"]
            or generated["db_id"]!=row["db_id"]
            or generated["question_sha256"]!=row["question_sha256"]):
            raise ValueError("source question/model SQL misalignment")
        db=dbroot/row["db_id"]/f"{row['db_id']}.sqlite"
        outcomes={name:sql_result(db,generated[key])
                  for name,key in (("baseline","base_SQL"),("plan","plan_SQL"))}
        # Gold reference first accessed HERE, after the immutable model SQL
        # generation file has been uploaded and no model call remains.
        gold=sql_result(db,str(actual[row["source_index"]]["SQL"]))
        usable=bool(gold["valid"])
        result.append({
            "source_index":row["source_index"],"db_id":row["db_id"],
            "reference_scorable":usable,
            "reference_reason":None if usable else gold["reason"],
            "base_EX":bool(outcomes["baseline"]["valid"] and
                           outcomes["baseline"]["set"]==gold["set"]) if usable else None,
            "plan_EX":bool(outcomes["plan"]["valid"] and
                           outcomes["plan"]["set"]==gold["set"]) if usable else None,
            "base_BAG":bool(outcomes["baseline"]["valid"] and
                           outcomes["baseline"]["bag"]==gold["bag"]) if usable else None,
            "plan_BAG":bool(outcomes["plan"]["valid"] and
                           outcomes["plan"]["bag"]==gold["bag"]) if usable else None,
            "base_SQL_executable":outcomes["baseline"]["valid"],
            "plan_SQL_executable":outcomes["plan"]["valid"],
            "plan_valid":generated["plan_valid"],
            "SQL_changed":generated["SQL_changed"],
            "qwen_calls":generated["qwen_calls"],
            "baseline_seconds":generated["baseline_generation_seconds"],
            "additional_plan_and_SQL_seconds":generated["plan_plus_generation_seconds"]
        })
    return {"status":"PAIRED_STRICT_SQLITE_EX_AFTER_SEALED_QWEN_GENERATION",
            "protocol":PROTO,"model_digest":DIGEST,
            "source_sha256":m["original_source_sha256"],
            "shard":shard,"db_id":m["source_dbs"][shard],
            "n":PER_DB,"gold_seen_by_model":False,"DEV500_used":False,
            "HOLDOUT300_used":False,"cases":result}


def aggregate(inputs:list[Path],meta:Path,out:Path)->dict:
    m=json.loads(meta.read_text())
    if len(inputs)!=DBS:
        raise ValueError("must aggregate both protected physical DB workers")
    parts=[json.loads(x.read_text()) for x in inputs]
    if sorted(x.get("shard") for x in parts)!=list(range(DBS)):
        raise ValueError("missing or duplicated original 2 DB shards")
    records=[]
    for part in parts:
        if (part.get("status")!="PAIRED_STRICT_SQLITE_EX_AFTER_SEALED_QWEN_GENERATION"
            or part.get("protocol")!=PROTO or part.get("model_digest")!=DIGEST
            or part.get("source_sha256")!=m["original_source_sha256"]
            or part.get("db_id")!=m["source_dbs"][part["shard"]]
            or part.get("n")!=PER_DB or len(part.get("cases",[]))!=PER_DB
            or part.get("DEV500_used") is not False
            or part.get("HOLDOUT300_used") is not False):
            raise ValueError("original model/source SQL shards incomplete")
        records+=part["cases"]
    if len(records)!=N or len({r["source_index"] for r in records})!=N:
        raise ValueError("incomplete source question identifiers")
    usable=[r for r in records if r["reference_scorable"]]
    rescue=[r for r in usable if not r["base_EX"] and r["plan_EX"]]
    regress=[r for r in usable if r["base_EX"] and not r["plan_EX"]]
    changed=sum(r["SQL_changed"] for r in records)
    diff=sum(r["plan_EX"]-r["base_EX"] for r in usable)
    if diff!=len(rescue)-len(regress):
        raise ValueError("incorrect paired EX transition counts")
    import random
    rng=random.Random(20261118)
    by_db={db:[int(r["plan_EX"])-int(r["base_EX"]) for r in usable if r["db_id"]==db]
           for db in m["source_dbs"]}
    if any(not xs for xs in by_db.values()):
        raise ValueError("missing physical DB cluster for paired CI")
    groups=sorted(by_db)
    draws=[]
    for _ in range(4000):
        chosen=[rng.choice(groups) for _ in groups]
        values=[v for db in chosen for v in by_db[db]]
        draws.append(sum(values)/len(values))
    draws.sort()
    ci={"estimate":diff/len(usable),"ci95_lower":draws[100],
        "ci95_upper":draws[3899],"n_databases":len(groups),
        "method":"fixed-seed 4000-round paired physical-DB-cluster bootstrap"}
    accepted=(len(usable)>=GATE["min_scorable"]
              and diff>=GATE["min_net_correct"]
              and len(regress)<=GATE["max_regressions"]
              and len({r["db_id"] for r in rescue})>=GATE["min_rescue_databases"]
              and sum(r["plan_BAG"] for r in usable)>=sum(r["base_BAG"] for r in usable))
    report={
        "status":"COMPLETE_REAL_QWEN9B_PROTECTED2DB64_SEMANTIC_PLAN",
        "protocol":PROTO,"model_digest":DIGEST,
        "source_sha256":m["original_source_sha256"],
        "n":N,"scorable":len(usable),"censored":N-len(usable),
        "physical_DB_n":DBS,"source_physical_dbs":m["source_dbs"],
        "baseline_strict_EX":sum(r["base_EX"] for r in usable),
        "plan_strict_EX":sum(r["plan_EX"] for r in usable),
        "baseline_BAG_EX":sum(r["base_BAG"] for r in usable),
        "plan_BAG_EX":sum(r["plan_BAG"] for r in usable),
        "rescues":len(rescue),"regressions":len(regress),"net_correct":diff,
        "rescue_DB_n":len({r["db_id"] for r in rescue}),
        "regression_DB_n":len({r["db_id"] for r in regress}),
        "paired_db_cluster_CI95":ci,
        "plan_valid_n":sum(r["plan_valid"] for r in records),
        "different_SQL_n":changed,
        "model_calls_total":sum(r["qwen_calls"] for r in records),
        "baseline_model_calls":N,
        "treatment_model_calls":sum(r["qwen_calls"]-1 for r in records),
        "additional_plan_and_SQL_wall_seconds":round(sum(r["additional_plan_and_SQL_seconds"] for r in records),2),
        "predeclared_gate":GATE,"positive_screen":accepted,
        "DEV500_used":False,"HOLDOUT300_used":False,
        "protected_two_unseen_TRAIN_DBS_used":True,
        "independent_DB_generalization_proven":False,
        "claim_boundary":"FINAL one-shot on 64 source-presealed questions from two previously untouched BIRD TRAIN physical databases; fixed Qwen3.5-9B and frozen previously tested plan; unmatched 1 vs 2 generation-call budget; two DB clusters are too few to generalize. Not Guarded max2 repair, BIRD DEV, or HOLDOUT300."
    }
    if out.exists():
        raise FileExistsError("immutable original Qwen9B EX evidence")
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("freeze","infer","score","aggregate"))
    p.add_argument("--train",type=Path)
    p.add_argument("--ledger",type=Path)
    p.add_argument("--prior-train",type=Path)
    p.add_argument("--prior-val",type=Path)
    p.add_argument("--prior-fanout",type=Path)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--source",type=Path)
    p.add_argument("--meta",type=Path)
    p.add_argument("--generated",type=Path)
    p.add_argument("--dbroot",type=Path)
    p.add_argument("--shard",type=int,default=0)
    p.add_argument("--inputs",type=Path,nargs="+")
    a=p.parse_args()
    if a.mode=="freeze":
        result=freeze(a.train,a.ledger,a.prior_train,a.prior_val,a.prior_fanout,a.out)
    elif a.mode=="infer":
        result=infer(a.source,a.meta,a.shard,a.dbroot,a.out)
    elif a.mode=="score":
        result=score(a.source,a.meta,a.generated,a.train,a.shard,a.dbroot,a.out)
        if a.out.exists():raise FileExistsError("source scoring already frozen")
        a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    else:
        result=aggregate(a.inputs,a.meta,a.out)
    print("FINAL_QWEN9B_PROTECTED64",json.dumps({
        k:v for k,v in result.items() if k not in ("cases",)
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
