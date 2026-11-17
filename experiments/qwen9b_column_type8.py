"""Preregistered, gold-blind Qwen3.5-9B semantic reviewer with SQL-referenced raw column physical type/values.

Fixed original Qwen9B first SQL and existing two-repair Guarded baseline.
Treatment sees original question/schema, baseline SQL, and actual raw SQLite
values and declared types of columns the baseline SQL physically references. Never a gold query, gold label or
stronger model. Tested on four source-disjoint BIRD TRAIN DBs, public only.

No QLoRA, no model change, no DEV/HOLDOUT, no automatic deployment.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
from time import perf_counter

from evaluation.run_bird import _ollama_model_digest
from evaluation.schema_context import select_schema_context
from scripts.prepare_birdtrain_qwen9b_sft_source import (
    TRAIN_SHA,DEV_SHA,load_checked,fetch_manifest,fetch_database,
    normalized,validate_gold_sql)
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.contracts import SQLCandidate
from semantic_sql.direct import _SYSTEM,_strip_fence
from semantic_sql.execution import execute_readonly,SQLExecutionError
from semantic_sql.inference import run_guarded
from semantic_sql.providers import OllamaProvider
from semantic_sql.verifier import verify_sql_preflight

MODEL="qwen3.5:9b-q4_K_M"
DIGEST="56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
PROTOCOL="QWEN9B-BIRD-TRAIN-NEWDB4X2-REFERENCED-COLUMN-EVIDENCE-V1"
SEED="birdtrain-referenced-column-physical-type-value-v1-20261117"
N=8
SHARDS=4
PER_SHARD=2
# First six are immutable source byte SHAs from prior TRAIN-only physical studies.
SOURCE_SHA={
 "prior_train":"b6851fb0f2b5027833aff45a0b5e01326359b245be1c812edc66986d0df1d402",
 "prior_validation":"cc112305bc9de33a03cea704f2edecfdfb45e62cc953ae498cea27fbed93b506",
 "prior_fk32":"ee1b5035302d36a1241952bd2e5acd3dfc2518d1fba2581f557e04e179960535",
 "prior_value16":"4de8471341b38d0fb00ded1f17f265d33b528ec78b9779eb94937914df196955",
 "prior_value16_replica":"faa4f7ba1de9923cbf8e51a500727dfe4c76a08c3f6366735cf21bff152655b7",
 "prior_role8":"3be2fc49ba3cff4171e20afcaf70f1fc698c8be401ac332a29fa1f48b781168f",
 "prior_preview8":"85c3e3abc45d90f23418dbc14261111abf90feadfef2b155988b77b40a4fe0ca",
}
SOURCE_COUNTS={"prior_train":242,"prior_validation":64,"prior_fk32":32,
               "prior_value16":16,"prior_value16_replica":16,"prior_role8":8,"prior_preview8":8
SOURCE_COLUMNS={"prior_train":"source_db_id","prior_validation":"source_db_id"}
MAX_DB_BYTES=96*1024*1024
QUESTION_SCOPE=re.compile(r"\b(how many|number of|count|average|percentage|ratio|most|least|highest|lowest|biggest|smallest|more than|less than|top)\b",re.I)
RULE=(
 "You are a conservative SQL semantic auditor. Original fixed Qwen3.5-9B "
 "already wrote an executable SQLite SELECT. You receive ONLY the original "
 "question, schema, the submitted SQL, and sampled RAW stored values plus "
 "physical SQLite typeof() and declared storage type of columns that the "
 "SQL itself actually references. These raw values are source evidence, "
 "NOT a gold answer. Look specifically for arithmetic over formatted TEXT "
 "money, comparison of categorical TEXT to numeric literals, date encodings, "
 "aggregation unit and filter scope. Do NOT add unrelated source columns "
 "or broaden filters just because other values resemble the question. "
 "If and ONLY if a concrete physical type/value conflict makes the SQL "
 "semantically wrong, output one corrected read-only SQLite SELECT with "
 "no fences or explanation. Otherwise repeat the SQL verbatim."
)
RULE_SHA=sha256(RULE.encode()).hexdigest()
GATE={"min_net_set_ex":2,"max_set_regressions":0,"min_database_rescues":2,
      "nondecreasing_bag":True}
MAX_SOURCE_EXAMINED_DB=30

def ranked(kind,item):
    return sha256(f"{SEED}:{kind}:{item}".encode()).hexdigest()

def _prior_exclusion(prior:dict[str,Path], dev:list[dict]):
    if set(prior)!=set(SOURCE_SHA):
        raise ValueError("all six prior source artifacts required")
    seen={normalized(r["question"]) for r in dev}
    excluded_db={r["db_id"] for r in dev}
    for label,path in prior.items():
        blob=path.read_bytes()
        if sha256(blob).hexdigest()!=SOURCE_SHA[label]:
            raise ValueError(f"{label}: original source SHA mismatch")
        rows=[json.loads(line) for line in blob.decode().splitlines() if line.strip()]
        if len(rows)!=SOURCE_COUNTS[label]:
            raise ValueError(f"{label}: count changed")
        field=SOURCE_COLUMNS.get(label,"db_id")
        dbs={r[field] for r in rows}
        if not dbs:
            raise ValueError(f"{label}: original DB identities absent")
        excluded_db.update(dbs)
        for r in rows:
            if "question" in r:
                seen.add(normalized(r["question"]))
            elif label in ("prior_train","prior_validation"):
                prompt=r["messages"][1]["content"]
                seen.add(normalized(prompt.split("\nSchema context:\n",1)[0].removeprefix("Question: ")))
    # 16+4+4+4+4+4 distinct BIRD TRAIN physical DBs, plus 11 DEV.
    known_dev={r["db_id"] for r in dev}
    if len(excluded_db-known_dev)!=40:
        raise ValueError(f"expected 40 previously exposed BIRD TRAIN DBs, got {len(excluded_db-known_dev)}")
    return excluded_db,seen

def freeze(train_file,dev_file,prior,root,out):
    train=load_checked(train_file,TRAIN_SHA)
    dev=load_checked(dev_file,DEV_SHA)
    excluded_db,seen=_prior_exclusion(prior,dev)
    manifest=fetch_manifest()
    grouped=defaultdict(list)
    for ix,r in enumerate(train):
        db=r.get("db_id")
        q=normalized(r.get("question",""))
        if (db not in manifest or db in excluded_db or
            not QUESTION_SCOPE.search(q) or q in seen or
            manifest[db]["size"]>MAX_DB_BYTES):
            continue
        seen.add(q)
        grouped[db].append((ix,r))
    eligible=sorted((db for db,rows in grouped.items() if len(rows)>=PER_SHARD),
                    key=lambda db:ranked("db",db))
    selected=[]
    rows=[]
    checks=0
    for db in eligible:
        if len(selected)==SHARDS or checks>=MAX_SOURCE_EXAMINED_DB:break
        checks+=1
        physical=fetch_database(db,manifest[db],root)
        catalog=DatabaseCatalog.from_sqlite(physical)
        accepted=[]
        for ix,r in sorted(grouped[db],key=lambda x:ranked("case",x[0]))[:30]:
            question=str(r["question"]).strip()
            gold_sql=str(r.get("SQL") or "")
            if validate_gold_sql(physical,gold_sql) is not None:
                continue
            schema,_=select_schema_context(catalog,root,db,question)
            evidence=str(r.get("evidence") or "").strip()
            prompt=f"Question: {question}\nSchema context:\n{schema}"
            if evidence:prompt+=f"\nExternal evidence:\n{evidence}"
            if len(prompt)>16000:continue
            accepted.append({"source_index":ix,"db_id":db,"question":question,
                             "gold_sql":gold_sql,"original_user":prompt,
                             "prompt_sha256":sha256(prompt.encode()).hexdigest()})
            if len(accepted)==PER_SHARD:break
        if len(accepted)<PER_SHARD:continue
        selected.append(db)
        rows.extend(accepted)
    if len(rows)!=N or len(selected)!=SHARDS:
        raise RuntimeError(f"not enough fresh physical DBs: {selected}, {len(rows)}")
    out.mkdir(parents=True,exist_ok=True)
    file=out/"column-type8.jsonl"
    file.write_text("".join(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n" for r in rows),encoding="utf-8")
    report={
        "status":"FROZEN_BEFORE_ANY_NEW_MODEL_INFERENCE","protocol":PROTOCOL,
        "train_sha":TRAIN_SHA,"dev_sha":DEV_SHA,
        "prior_artifact_shas":SOURCE_SHA,
        "source_sha256":sha256(file.read_bytes()).hexdigest(),
        "source_dbs":selected,"excluded_prior_train_dbs":40,
        "exclude_all_11_dev_dbs":True,"n":N,"shards":SHARDS,
        "selection_seed":SEED,"questions_conditioned_on_numeric_relation":True,
        "publisher_sql_compiler_checked_only":True,
        "gold_not_used_for_treatment":True,"model_digest":DIGEST,
        "review_rule_sha256":RULE_SHA,"gate":GATE,
        "not_GAIA":True,"no_training":True,"no_stronger_model":True,
        "public_train_data_not_hidden":True,
        "unknown_other_exposures":True,
        "not_general_population":True,
    }
    (out/"freeze.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report

def load_frozen(path,meta):
    raw=path.read_bytes()
    m=json.loads(meta.read_text())
    if (m.get("status")!="FROZEN_BEFORE_ANY_NEW_MODEL_INFERENCE"
        or m.get("protocol")!=PROTOCOL or m.get("n")!=N or m.get("shards")!=SHARDS
        or sha256(raw).hexdigest()!=m.get("source_sha256")
        or m.get("model_digest")!=DIGEST
        or m.get("review_rule_sha256")!=RULE_SHA or m.get("gate")!=GATE
        or m.get("excluded_prior_train_dbs")!=40):
        raise RuntimeError("frozen source/model/review contract drift")
    rows=[json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if len(rows)!=N or len({r["source_index"] for r in rows})!=N:
        raise ValueError("not exactly 8 new unique source IDs")
    if (len({r["db_id"] for r in rows})!=SHARDS
        or [rows[i]["db_id"] for i in range(0,N,2)]!=m["source_dbs"]
        or any(rows[2*i]["db_id"]!=rows[2*i+1]["db_id"] for i in range(SHARDS))):
        raise ValueError("four physical source DBs not grouped correctly")
    for r in rows:
        if (not QUESTION_SCOPE.search(r["question"])
            or sha256(r["original_user"].encode()).hexdigest()!=r["prompt_sha256"]
            or not r["original_user"].startswith("Question: "+r["question"]+"\nSchema context:\n")):
            raise ValueError("question selection/prompt integrity drift")
    return rows

def score(db,sql,gold_set,gold_bag):
    if not sql:return {"ex":False,"bag":False,"exec":False}
    try:
        result=execute_readonly(db,sql,max_rows=50000)
        if result.truncated:return {"ex":False,"bag":False,"exec":False}
        return {"ex":set(result.rows)==gold_set,
                "bag":Counter(result.rows)==gold_bag,"exec":True}
    except (SQLExecutionError,ValueError):
        return {"ex":False,"bag":False,"exec":False}

def _ident(value):
    return '"' + str(value).replace('"','""') + '"'

def _source_column_evidence(db,sql,question,max_columns=10):
    """SQLite authorizer: only source columns READ by the exact proposed SQL.

    No fuzzy matching of unrelated database values, oracle labels or answers.
    Fetch <=4 actual stored values and declared storage type of <=10 columns.
    """
    conn=sqlite3.connect(f"{Path(db).resolve().as_uri()}?mode=ro",uri=True)
    try:
        conn.execute("PRAGMA query_only = ON")
        used=set()
        def capture(code,table,column,_db,_trigger):
            if (code==sqlite3.SQLITE_READ and isinstance(table,str)
                and isinstance(column,str) and column
                and not table.startswith("sqlite_")):
                used.add((table,column))
            return sqlite3.SQLITE_OK
        conn.set_authorizer(capture)
        try:
            conn.execute("EXPLAIN QUERY PLAN " + sql).fetchall()
        finally:
            conn.set_authorizer(None)
        def rank(pair):
            table,col=pair
            q=re.sub(r"[^a-z0-9]","",question.casefold())
            k=re.sub(r"[^a-z0-9]","",col.casefold())
            relevant=bool(k and k in q and len(k)>=3)
            risk=bool(re.search(r"(?i)salary|price|cost|rate|amount|len|length|year|date|score|percent|rating|genre|category|sentiment|polar|profit|duration|unit|size",col))
            return (-int(relevant),-int(risk),table.casefold(),col.casefold())
        results=[]
        for table,col in sorted(used,key=rank)[:max_columns]:
            try:
                types={r[1]:str(r[2]) for r in conn.execute("PRAGMA table_info("+_ident(table)+")")}
                if col not in types:
                    continue
                qcol=_ident(col);qtable=_ident(table)
                samples=conn.execute(
                    f"SELECT DISTINCT typeof({qcol}), CAST({qcol} AS TEXT) "
                    f"FROM {qtable} WHERE {qcol} IS NOT NULL LIMIT 4").fetchall()
                if not samples:
                    continue
                results.append({"table":table,"column":col,
                                "declared_type":types[col],
                                "distinct_raw_examples":[{"sqlite_type":str(t),
                                                          "literal":str(v)[:96]}
                                                         for t,v in samples]})
            except (sqlite3.Error,ValueError):
                continue
        return {"source":"SQLite read-only SQL-referenced columns only",
                "columns":results,"sample_limit_per_column":4,
                "gold_or_publisher_sql_used":False,
                "column_count":len(results)}
    finally:
        conn.close()

def _prompt(question,original_user,sql,source_columns):
    return json.dumps({"question":question,
                       "original_question_and_schema":original_user,
                       "baseline_sql":sql,
                       "physical_sqlite_referenced_column_evidence":source_columns},
                       ensure_ascii=False,sort_keys=True)

def one_case(db,row,provider):
    question=row["question"]
    prompt=row["original_user"]
    parts=prompt[10:].split("\nSchema context:\n",1)
    schema_and_ev=parts[1]
    if "\nExternal evidence:\n" in schema_and_ev:
        schema,evidence=schema_and_ev.split("\nExternal evidence:\n",1)
    else:
        schema,evidence=schema_and_ev,None
    gold=execute_readonly(db,row["gold_sql"],max_rows=50000)
    if gold.truncated:
        raise ValueError("gold truncated")
    gold_set,gold_bag=set(gold.rows),Counter(gold.rows)
    started=perf_counter()
    initial=_strip_fence(provider.complete_text(system=_SYSTEM,user=prompt))
    guarded=run_guarded(database=db,provider=provider,question=question,
                       schema_context=schema,evidence=evidence,
                       initial_candidate=SQLCandidate(sql=initial,attempt=0),
                       max_repairs=2,max_rows=50000)
    baseline=guarded.candidate.sql if guarded.candidate else ""
    baseline_result=score(db,baseline,gold_set,gold_bag)
    baseline_elapsed=round(perf_counter()-started,2)
    status="no_executable_base"
    after=baseline
    source_columns=None
    reviewer_elapsed=0
    if baseline_result["exec"]:
        source_columns=_source_column_evidence(db,baseline,question)
        if source_columns and source_columns["columns"]:
            review_start=perf_counter()
            revision=_strip_fence(provider.complete_text(system=RULE,
                         user=_prompt(question,prompt,baseline,source_columns)))
            reviewer_elapsed=round(perf_counter()-review_start,2)
            if not revision or revision.strip().rstrip(";")==baseline.strip().rstrip(";"):
                status="reviewer_abstained"
            else:
                valid=verify_sql_preflight(db,revision)
                if valid.ok and score(db,revision,gold_set,gold_bag)["exec"]:
                    after=revision
                    status="accepted_semantic_revision"
                else:
                    status="revision_rejected_unexecutable"
    tested=score(db,after,gold_set,gold_bag)
    return {
      "source_index":row["source_index"],"db_id":row["db_id"],
      "baseline_sql":baseline,"treatment_sql":after,
      "baseline":baseline_result,"reviewed":tested,
      "review_status":status,
      "evidence_columns":source_columns["column_count"] if source_columns else 0,
      "evidence_columns_with_raw_text":sum(
          any(v["sqlite_type"]=="text" for v in col["distinct_raw_examples"])
          for col in source_columns["columns"]) if source_columns else 0,
      "reviewer_extra_calls":int(bool(source_columns and source_columns["columns"])),
      "baseline_elapsed_s":baseline_elapsed,"reviewer_elapsed_s":reviewer_elapsed,
      "question":question,
      "source_prompt_sha256":row["prompt_sha256"],
    }

def run_shard(cohort,meta,shard,work):
    if shard not in range(SHARDS):raise ValueError("bad shard")
    rows=load_frozen(cohort,meta)[shard*PER_SHARD:(shard+1)*PER_SHARD]
    if _ollama_model_digest("http://127.0.0.1:11434",MODEL)!=DIGEST:
        raise RuntimeError("model digest changed")
    manifest=fetch_manifest()
    dbid=rows[0]["db_id"]
    db=fetch_database(dbid,manifest[dbid],work)
    provider=OllamaProvider(model=MODEL,temperature=0.0,seed=42,num_predict=768,timeout_s=900)
    cases=[]
    for row in rows:
        case=one_case(db,row,provider)
        cases.append(case)
        print("COLUMN_TYPE8",shard,row["source_index"],case["baseline"]["ex"],
              case["reviewed"]["ex"],case["review_status"],flush=True)
    return {"status":"COMPLETE_PHYSICAL_PAIRED","protocol":PROTOCOL,
            "shard":shard,"db_id":dbid,
            "source_sha256":sha256(cohort.read_bytes()).hexdigest(),
            "model_digest":DIGEST,"review_rule_sha256":RULE_SHA,"seed":42,
            "temperature":0,"max_repairs":2,"n":PER_SHARD,"cases":cases}

def aggregate(files):
    if len(files)!=SHARDS:raise ValueError("four complete physical shard files required")
    shards=[json.loads(p.read_text()) for p in files]
    if sorted(s.get("shard") for s in shards)!=list(range(SHARDS)):
        raise ValueError("duplicate/missing shard")
    if any(s.get("status")!="COMPLETE_PHYSICAL_PAIRED" or
           s.get("protocol")!=PROTOCOL or s.get("model_digest")!=DIGEST or
           s.get("review_rule_sha256")!=RULE_SHA or s.get("n")!=PER_SHARD or
           len(s.get("cases",[]))!=PER_SHARD or s.get("max_repairs")!=2
           for s in shards):
        raise ValueError("model/source/model-run protocol drift")
    if len({s["source_sha256"] for s in shards})!=1:
        raise ValueError("source SHA mismatch")
    rows=[c for s in sorted(shards,key=lambda s:s["shard"]) for c in s["cases"]]
    if len({r["source_index"] for r in rows})!=N or len({s["db_id"] for s in shards})!=SHARDS:
        raise ValueError("not 8 unique tasks/4 new source databases")
    rescue=[r for r in rows if not r["baseline"]["ex"] and r["reviewed"]["ex"]]
    regress=[r for r in rows if r["baseline"]["ex"] and not r["reviewed"]["ex"]]
    rescue_dbs={r["db_id"] for r in rescue}
    totals={arm:{"strict_set_ex":sum(r[arm]["ex"] for r in rows),
                 "strict_bag_ex":sum(r[arm]["bag"] for r in rows),
                 "execution_success":sum(r[arm]["exec"] for r in rows)}
            for arm in ("baseline","reviewed")}
    passed=(len(rescue)-len(regress)>=GATE["min_net_set_ex"]
            and len(regress)<=GATE["max_set_regressions"]
            and len(rescue_dbs)>=GATE["min_database_rescues"]
            and totals["reviewed"]["strict_bag_ex"]>=totals["baseline"]["strict_bag_ex"])
    return {
      "status":"COMPLETE_PHYSICAL_8_PUBLIC_DIAGNOSTIC_NO_PROMOTION",
      "n":N,"dbs":SHARDS,"protocol":PROTOCOL,
      "model_digest":DIGEST,"source_sha256":shards[0]["source_sha256"],
      "review_rule_sha256":RULE_SHA,"same_model_review_only":True,
      "actual_new_qwen_training_steps":0,"no_qlora":True,"dev_holdout_used":False,
      "totals":totals,"rescues":len(rescue),"regressions":len(regress),
      "net_set_ex":len(rescue)-len(regress),
      "rescue_db_count":len(rescue_dbs),
      "review_accepted":sum(r["review_status"]=="accepted_semantic_revision" for r in rows),
      "review_extra_calls":sum(r["reviewer_extra_calls"] for r in rows),
      "total_review_elapsed_s":round(sum(r["reviewer_elapsed_s"] for r in rows),2),
      "total_physically_sampled_columns":sum(r["evidence_columns"] for r in rows),
      "positive_screen_only":passed,"promotion_allowed":False,
      "gate":GATE,"cases":rows,
      "limitations":["8 source-conditioned public BIRD TRAIN tasks from four new physical databases",
                     "publisher labels compile checked, not independently human semantic adjudicated",
                     "full upstream BIRD evaluator not executed","no hidden/DEV score"],
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-only",action="store_true")
    p.add_argument("--train",type=Path)
    p.add_argument("--dev",type=Path)
    p.add_argument("--prior-train",type=Path)
    p.add_argument("--prior-validation",type=Path)
    p.add_argument("--prior-fk32",type=Path)
    p.add_argument("--prior-value16",type=Path)
    p.add_argument("--prior-value16-replica",type=Path)
    p.add_argument("--prior-role8",type=Path)
    p.add_argument("--prior-preview8",type=Path)
    p.add_argument("--cohort",type=Path)
    p.add_argument("--meta",type=Path)
    p.add_argument("--shard",type=int)
    p.add_argument("--work",type=Path,default=Path("source-sqlite"))
    p.add_argument("--inputs",nargs="+",type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError("immutable source/result already exists")
    if a.source_only:
        prior={k:getattr(a,k.replace("-","_")) for k in SOURCE_SHA}
        if any(q is None for q in (a.train,a.dev,*prior.values())):
            p.error("source-only requires pinned train/dev plus all six prior source manifests")
        result=freeze(a.train,a.dev,prior,a.work,a.out)
        print("COLUMN_TYPE8_SOURCE_FROZEN",json.dumps(result,sort_keys=True),flush=True)
    elif a.inputs:
        result=aggregate(a.inputs)
        print("COLUMN_TYPE8_PAIRED_COMPLETE",json.dumps({k:v for k,v in result.items() if k!="cases"},sort_keys=True),flush=True)
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False)+"\n")
    else:
        if a.cohort is None or a.meta is None or a.shard is None:
            p.error("need cohort, meta, shard or --inputs")
        result=run_shard(a.cohort,a.meta,a.shard,a.work)
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False)+"\n")

if __name__=="__main__":
    main()
