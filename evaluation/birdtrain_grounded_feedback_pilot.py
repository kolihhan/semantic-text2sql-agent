"""Development-only blind Qwen9B semantic-review A/B on EXISTING BIRD TRAIN val64.
Four previously exposed train databases, three unselected questions each.
Do not claim OOD: protected final 2 TRAIN DBs and DEV500/DEV300 remain untouched.
"""
from __future__ import annotations
import argparse
from collections import defaultdict, Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
import time

PROTO="BIRD_TRAIN_EXPOSED_VAL64_SEMANTIC_FEEDBACK_3ARM_V1"
SOURCE_SHA="cc112305bc9de33a03cea704f2edecfdfb45e62cc953ae498cea27fbed93b506"
MODEL="qwen3.5:9b-q4_K_M"
MODEL_DIGEST="56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
SEED="exposed-validation64-four-dbs-three-each-no-correctness-20261121"
GENERIC_SYSTEM=("Check the SQL against the question and physical schema: output, grain, "
                "aggregations, joins, filters, ranking. Revise only if a clear semantic "
                "contradiction is proven. Otherwise return the original unchanged. "
                "Return exactly one read-only SQLite SELECT/CTE; SQL only, no prose.")
EVIDENCE_SYSTEM=("Conservative SQL editor. The appended parsed SQL clauses and physical "
                 "keys describe what the SQL does; they do NOT say what the question means "
                 "and are NOT proof of an error. Compare them to the user's requested "
                 "answer unit, relations, constraints and ranking. Only fix a direct "
                 "question/schema contradiction, preserving all other conditions and "
                 "fields. If unsure, return ORIGINAL unchanged. Return one SQLite "
                 "read-only SELECT/CTE; SQL only, no explanations.")
def sha(s):return sha256(s.encode() if isinstance(s,str) else s).hexdigest()
def load(path):return json.loads(path.read_text(encoding="utf-8"))
def write_new(path,p):
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(p,ensure_ascii=False,sort_keys=True,indent=2)+"\n")
def source_rows(path):
    if sha(path.read_bytes())!=SOURCE_SHA:raise ValueError("exposed validation64 SHA drift")
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows)!=64 or len({r["source_index"] for r in rows})!=64:raise ValueError("validation64 roster drift")
    if len({r["source_db_id"] for r in rows})!=4:raise ValueError("expected 4 exposed validation DBs")
    return rows
def project_input(r):
    # Explicitly DO NOT read r["messages"][2], containing publisher gold SQL.
    s=r["messages"][1]["content"]
    if not s.startswith("Question: ") or "\nSchema context:\n" not in s:raise ValueError("input format drift")
    q=s[len("Question: "):].split("\nSchema context:\n",1)[0]
    evidence=s.split("\nExternal evidence:\n",1)[1] if "\nExternal evidence:\n" in s else ""
    return q,evidence
def freeze(src,out):
    bydb=defaultdict(list)
    for r in source_rows(src):
        q,e=project_input(r)
        bydb[r["source_db_id"]].append({"source_index":r["source_index"],
            "db_id":r["source_db_id"],"question":q,"evidence":e,
            "question_sha256":sha(q),"evidence_sha256":sha(e)})
    chosen=[]
    for db in sorted(bydb):
        pool=sorted(bydb[db],key=lambda x:sha(SEED+"|"+db+"|"+str(x["source_index"])))
        chosen+=pool[:3]
    if len(chosen)!=12:raise ValueError("not 4x3")
    result={"protocol":PROTO,"phase":"SOURCE_FROZEN_NO_GOLD",
        "exposure":"PREVIOUSLY_EXPOSED_TRAIN_VALIDATION64_NOT_INDEPENDENT",
        "source_sha256":SOURCE_SHA,"model_digest":MODEL_DIGEST,"selection_seed":SEED,
        "n":len(chosen),"n_physical_dbs":len(bydb),"cases":chosen,
        "reference_sql_excluded":True,"heldout_two_train_db_accessed":False,
        "DEV500_accessed":False,"DEV300_accessed":False}
    write_new(out,result)
    print("SEMANTIC_FEEDBACK_FREEZE",json.dumps({"n":12,"dbs":sorted(bydb),"gold_free":True}),flush=True)
def db_file(root,id):
    f=root/id/(id+".sqlite")
    if not f.is_file():raise FileNotFoundError(f)
    return f
def structure(sql,db):
    """Pure AST/counted-key facts; NO case-ID heuristics, no guessed error label."""
    import sqlglot
    from sqlglot import exp
    try:root=sqlglot.parse_one(sql,read="sqlite")
    except sqlglot.ParseError:return "SQL AST unverified; no error implied"
    node=root.find(exp.Select)
    if node is None:return "SQL AST SELECT unavailable; no error implied"
    tables=sorted({t.name for t in root.find_all(exp.Table)})[:12]
    observed=["Tables: "+", ".join(tables),
       "Projection: "+", ".join(x.sql(dialect="sqlite") for x in node.expressions[:10])]
    for name,key in (("WHERE","where"),("GROUP","group"),("HAVING","having"),
                     ("ORDER","order"),("LIMIT","limit"),("OFFSET","offset")):
        x=node.args.get(key)
        observed.append(name+": "+(x.sql(dialect="sqlite")[:900] if x else "(absent)"))
    aggs=sorted({x.sql(dialect="sqlite") for x in node.find_all(exp.AggFunc)})[:12]
    observed.append("Aggregates: "+(", ".join(aggs) if aggs else "(none)"))
    joins=node.args.get("joins") or []
    observed.append("Joins: "+"; ".join(x.sql(dialect="sqlite")[:450] for x in joins[:8]))
    cx=sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro",uri=True)
    try:
        names={x[0] for x in cx.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table in tables:
            if table not in names:continue
            t=table.replace('"','""')
            pk=[x[1] for x in cx.execute(f'PRAGMA table_info("{t}")') if x[5]>0]
            fk=[f"{x[3]}->{x[2]}.{x[4]}" for x in cx.execute(f'PRAGMA foreign_key_list("{t}")')]
            observed.append(f"Physical {table}: primary keys {pk}; declared foreign keys {fk[:10]}")
    finally:cx.close()
    observed.append("These are structural observations, NOT semantic correctness labels.")
    return "\n".join(observed)[:6000]
def validate_revision(original,answer,db):
    from semantic_sql.direct import _strip_fence
    from semantic_sql.verifier import verify_sql_preflight
    sql=_strip_fence(answer).strip()
    if not sql or not verify_sql_preflight(db,sql).ok:return original,"invalid_fallback"
    canon=lambda s:re.sub(r"\s+"," ",s).strip().rstrip(";")
    return (original,"unchanged") if canon(sql)==canon(original) else (sql,"changed")
def infer(src,root,out):
    from semantic_sql.catalog import DatabaseCatalog
    from semantic_sql.direct import generate_direct_sql
    from semantic_sql.inference import run_guarded
    from semantic_sql.providers import OllamaProvider
    from semantic_sql.verifier import verify_sql_preflight
    from evaluation.run_bird import CountingProvider,_ollama_model_digest
    from evaluation.schema_context import select_schema_context
    freeze=load(src)
    if freeze.get("protocol")!=PROTO or freeze.get("phase")!="SOURCE_FROZEN_NO_GOLD" or freeze.get("n")!=12:
        raise ValueError("frozen/source mismatch")
    if _ollama_model_digest("http://localhost:11434",MODEL)!=MODEL_DIGEST:
        raise ValueError("Qwen9B digest mismatch")
    result={"protocol":PROTO,"phase":"INCOMPLETE_BLIND","cases":[],
            "source_sha256":SOURCE_SHA,"model_digest":MODEL_DIGEST,
            "prompt_sha256":{"generic":sha(GENERIC_SYSTEM),"evidence":sha(EVIDENCE_SYSTEM)},
            "reference_sql_unavailable":True}
    write_new(out,result)
    for item in freeze["cases"]:
        db=db_file(root,item["db_id"])
        cat=DatabaseCatalog.from_sqlite(db)
        schema,_=select_schema_context(cat,root,item["db_id"],item["question"])
        p=CountingProvider(OllamaProvider(model=MODEL,timeout_s=600,
                          temperature=0,seed=42,num_predict=384))
        started=time.monotonic()
        first=generate_direct_sql(item["question"],schema,p,external_evidence=item["evidence"] or None)
        baseline=run_guarded(database=db,provider=p,question=item["question"],
                schema_context=schema,evidence=item["evidence"] or None,
                initial_candidate=first,max_repairs=2,max_rows=50000)
        baseline_seconds=round(time.monotonic()-started,2)
        sql=baseline.candidate.sql if baseline.candidate else ""
        valid=(baseline.status=="ok" and bool(sql) and verify_sql_preflight(db,sql).ok)
        calls=p.total_calls
        treatment={"generic":sql,"evidence":sql}
        statuses={"generic":"baseline_invalid_skip","evidence":"baseline_invalid_skip"}
        timings={"generic":0.0,"evidence":0.0}
        if valid:
            prompt=f"Question:\n{item['question']}\nSchema context:\n{schema}\n"
            if item["evidence"]:prompt+=f"External evidence:\n{item['evidence']}\n"
            prompt+=f"Original SQL:\n{sql}\n"
            for arm,system,text in (("generic",GENERIC_SYSTEM,prompt),
                    ("evidence",EVIDENCE_SYSTEM,prompt+"\nObserved SQL AST/physical metadata:\n"+structure(sql,db))):
                start=time.monotonic()
                try:
                    answer=p.complete_text(system=system,user=text)
                    treatment[arm],statuses[arm]=validate_revision(sql,answer,db)
                except Exception as e:
                    statuses[arm]="model_error:"+type(e).__name__
                timings[arm]=round(time.monotonic()-start,2)
        result["cases"].append({"source_index":item["source_index"],"db_id":item["db_id"],
            "question_sha256":item["question_sha256"],"baseline_sql":sql,
            "baseline_status":baseline.status,"preflight_ok":valid,
            "baseline_calls":calls,"baseline_latency_s":baseline_seconds,
            "generic_sql":treatment["generic"],"evidence_sql":treatment["evidence"],
            "generic_status":statuses["generic"],"evidence_status":statuses["evidence"],
            "generic_latency_s":timings["generic"],"evidence_latency_s":timings["evidence"],
            "total_model_calls":p.total_calls})
        out.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+"\n")
        print("BLIND_FEEDBACK",len(result["cases"]),item["db_id"],baseline.status,
              statuses["generic"],statuses["evidence"],flush=True)
    result["phase"]="ALL12_GOLD_BLIND_SEALED"
    out.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+"\n")
    print("BLIND_FEEDBACK_SEALED",len(result["cases"]),flush=True)
def score(frozen,blind,source,db_root,out):
    from semantic_sql.execution import execute_readonly,SQLExecutionError
    f=load(frozen); b=load(blind)
    if f.get("protocol")!=PROTO or b.get("protocol")!=PROTO or b.get("phase")!="ALL12_GOLD_BLIND_SEALED":
        raise ValueError("unsealed blind inference; no publisher gold permitted")
    if len(f["cases"])!=12 or len(b["cases"])!=12 or b.get("model_digest")!=MODEL_DIGEST:
        raise ValueError("source/model count drift")
    labels={r["source_index"]:r for r in source_rows(source)}
    rows=[]
    for q,guess in zip(f["cases"],b["cases"]):
        if q["source_index"]!=guess["source_index"] or q["db_id"]!=guess["db_id"]:
            raise ValueError("question/DB identity drift")
        source_row=labels[q["source_index"]]
        question,evidence=project_input(source_row)
        if sha(question)!=q["question_sha256"] or sha(evidence)!=q["evidence_sha256"]:
            raise ValueError("question/evidence drift")
        db=db_file(db_root,q["db_id"])
        def execute(sql):
            try:
                x=execute_readonly(db,sql,max_rows=50000)
                return None if x.truncated else x
            except (SQLExecutionError,TypeError,ValueError):return None
        # First and ONLY now open publisher reference SQL, from existing validation64.
        gold=execute(source_row["messages"][2]["content"])
        preds={arm:execute(guess[arm+"_sql"]) for arm in ("baseline","generic","evidence")}
        def valid(x,metric):
            if x is None or gold is None:return False
            return (set(x.rows)==set(gold.rows)) if metric=="set" else (Counter(x.rows)==Counter(gold.rows))
        rows.append({"source_index":q["source_index"],"db_id":q["db_id"],
           "scorable":gold is not None,"baseline_preflight":guess["preflight_ok"],
           "generic_status":guess["generic_status"],"evidence_status":guess["evidence_status"],
           "baseline_calls":guess["baseline_calls"],"total_calls":guess["total_model_calls"],
           "latency_s":{a:guess[a+"_latency_s"] for a in ("baseline","generic","evidence")},
           "set":{a:valid(preds[a],"set") for a in preds},
           "bag":{a:valid(preds[a],"bag") for a in preds}})
    data=[x for x in rows if x["scorable"]]
    stats={}
    for metric in ("set","bag"):
        stats[metric]={}
        for arm in ("baseline","generic","evidence"):
            stats[metric][arm]={"correct":sum(x[metric][arm] for x in data),
                "rescues":sum(x[metric][arm] and not x[metric]["baseline"] for x in data),
                "regressions":sum(x[metric]["baseline"] and not x[metric][arm] for x in data)}
    stats["extra_calls_each_review"]=sum(x["baseline_preflight"] for x in rows)
    stats["baseline_calls"]=sum(x["baseline_calls"] for x in rows)
    stats["actual_total_calls"]=sum(x["total_calls"] for x in rows)
    stats["changed"]={arm:sum(x[arm+"_status"]=="changed" for x in rows) for arm in ("generic","evidence")}
    stats["model_errors"]={arm:sum(x[arm+"_status"].startswith("model_error") for x in rows) for arm in ("generic","evidence")}
    report={"protocol":PROTO,"phase":"EXPOSED_DB_DEVELOPMENT_SCORED","n":len(rows),
       "scorable":len(data),"dbs":len({x["db_id"] for x in rows}),
       "scores":stats,"cases":rows,"gold_unseen_until_all_blind_calls_sealed":True,
       "not_new_ood_evidence":True,"promotion_forbidden":True,
       "DEV500_unopened":True,"DEV300_unopened":True,"final2_TRAIN_DBs_unopened":True}
    write_new(out,report)
    print("SEMANTIC_FEEDBACK_PILOT_SCORED",json.dumps({k:v for k,v in report.items() if k!="cases"},sort_keys=True),flush=True)
def main():
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=("freeze","infer","score"))
    for x in ("source","frozen","blind","db_root","out"):
        p.add_argument("--"+x.replace("_","-"),type=Path)
    a=p.parse_args()
    if not a.out:raise ValueError("out required")
    if a.phase=="freeze":freeze(a.source,a.out)
    elif a.phase=="infer":infer(a.frozen,a.db_root,a.out)
    else:score(a.frozen,a.blind,a.source,a.db_root,a.out)
if __name__=="__main__":main()
