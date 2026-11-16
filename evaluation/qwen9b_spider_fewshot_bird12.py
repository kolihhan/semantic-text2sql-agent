"""Original Qwen3.5-9B: TEST genuine Spider TRAIN structural few-shot retrieval.

Only intervention: add two independently annotated Spider TRAIN question/SQL
examples to the FIRST generation prompt. Original selected schema, question,
pinned Qwen9B digest/seed/temperature, and <=2 verifier repairs stay identical.
No SQL finetune. No BIRD labels (or prior generated SQL) select exemplars.
The old Qwen BIRD100 and prior exposed 12 are excluded before inference.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from time import monotonic

from evaluation.qwen_schema_grounding_pair import (
    MODEL, MODEL_DIGEST, SOURCE_SHA, PinnedQwen, checked_source, case_id, legacy_tokens,
)
from evaluation.run_bird import _ollama_model_digest, _resolve_database
from evaluation.schema_context import select_schema_context
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.execution import SQLExecutionError, execute_readonly
from semantic_sql.inference import run_guarded

PROTOCOL="same-qwen9b-spider-train-structure-two-examples-bird12-v1"
SEED="original-qwen9b-source-independent-spider-fewshot-20261116"
N=12
SHARDS=4
PER_SHARD=3
MAX_CONTEXT_CHARS=12000
STOP={"a","an","the","with","or","and","is","are","for","in","on",
      "of","to","that","whose","which","from","what","get","list","show",
      "give","find","return","all","have","has","who"}
PATTERNS={
    "count":r"\b(how many|number of|count of|count)\b",
    "mean":r"\b(average|mean)\b",
    "sum":r"\b(total|sum)\b",
    "rank":r"\b(most|least|highest|lowest|top|largest|smallest|first|last)\b",
    "group":r"\b(each|every|per)\b",
    "compare":r"\b(more than|less than|above|below|at least|at most|greater than|fewer than)\b",
    "negative":r"\b(without|not |except |never )",
    "time":r"\b(before|after|year|month|date|since|between)\b",
    "distinct":r"\b(distinct|different|unique)\b",
}


def _normal(text):
    return " ".join(str(text).split())


def words(text):
    return {t.rstrip("s") for t in re.findall(r"[a-z0-9]+",str(text).lower())
            if len(t)>2 and t not in STOP}


def ops(question):
    q=_normal(question).lower()
    return frozenset(k for k,p in PATTERNS.items() if re.search(p,q))


def train_candidates(rows):
    records=[]
    seen=set()
    for row in rows:
        if not isinstance(row,dict):
            continue
        q=_normal(row.get("question",""))
        sql=_normal(row.get("query",""))
        db=_normal(row.get("db_id",""))
        if not q or not sql or not db or len(q)>220 or len(sql)>500:
            continue
        if not re.match(r"^(SELECT|WITH)\b",sql,re.I) or ";" in sql.strip().rstrip(";"):
            continue
        if not ops(q):  # Transfer SQL operators, not generic word overlap.
            continue
        identity=sha256(f"{db}\n{q}\n{sql}".encode()).hexdigest()
        if identity in seen:
            continue
        seen.add(identity)
        records.append({"source":"Spider TRAIN","db_id":db,
                        "question":q,"sql":sql.rstrip(";"),"fingerprint":identity})
    if len(records)<300 or len({r["db_id"] for r in records})<25:
        raise ValueError("Spider original TRAIN source unexpectedly small or wrong split")
    return records


def retrieve(question,db_id,train,limit=2):
    target_ops=ops(question)
    if not target_ops:
        return []
    wt=words(question)
    scored=[]
    for item in train:
        if item["db_id"]==db_id or _normal(item["question"]).casefold()==_normal(question).casefold():
            continue
        common=target_ops & ops(item["question"])
        if not common:
            continue
        overlap=len(wt & words(item["question"]))
        score=10*len(common)+min(overlap,4)
        stable=sha256((SEED+":"+item["fingerprint"]).encode()).hexdigest()
        scored.append((-score,stable,item))
    scored.sort(key=lambda x:(x[0],x[1]))
    selected,dbs=[],set()
    for _,_,item in scored:
        if item["db_id"] in dbs:
            continue
        selected.append(item)
        dbs.add(item["db_id"])
        if len(selected)>=limit:
            return selected
    return []


class TwoExampleFirstCall:
    """Same Qwen model; change only the first generation prompt."""
    def __init__(self,base,examples):
        self.base=base
        self.examples=examples
        self.calls=0

    def complete_text(self,*,system,user):
        self.calls+=1
        if self.calls>1:
            return self.base.complete_text(system=system,user=user)
        prefix=(
            "Two annotated SQL examples from DIFFERENT databases are shown only "
            "to illustrate generic SQL operators and relational structure. "
            "They use unrelated schemas: NEVER copy table names, column names, "
            "literal values or answers from the examples. Use ONLY the target "
            "question and its physical schema for actual SQL.\n"
        )
        for index,example in enumerate(self.examples,1):
            prefix+=(
                f"\nExample {index} [other database]:\n"
                f"Question: {example['question']}\n"
                f"SQL: {example['sql']}\n"
            )
        prefix+="\n---\nTarget SQL task (use only its schema):\n"
        return self.base.complete_text(system=system,user=prefix+user)


def source_manifest(source,root,reference,excluded_source,train_source,train_sha):
    rows=checked_source(source)
    original=json.loads(reference.read_text())
    if original.get("status")!="complete" or original.get("sample_size")!=100:
        raise ValueError("original BIRD100 exclusion invalid")
    excluded={str(x["case_id"]) for x in original["cases"]}
    extra=json.loads(excluded_source.read_text())
    if (extra.get("case_count")!=12 or len(extra.get("cases",[]))!=12
        or extra.get("protocol")!="original-qwen9b-full-physical-schema-vs-lexical-bird12-v1"):
        raise ValueError("prior exposed BIRD12 exclusion invalid")
    excluded |= {str(x["case_id"]) for x in extra["cases"]}
    if len(excluded)!=112:
        raise ValueError("original and latest BIRD sample overlap; must not reuse")
    train=train_candidates(train_source)
    indexes=sorted(
        [(case_id(r,i),r) for i,r in enumerate(rows)
         if case_id(r,i) not in excluded],
        key=lambda pair:(sha256((SEED+":"+pair[0]).encode()).hexdigest(),pair[0]),
    )
    all_candidates=[]
    catalogs={}
    for cid,row in indexes:
        db_id=str(row["db_id"])
        question=str(row["question"])
        if not ops(question):
            continue
        examples=retrieve(question,db_id,train)
        if len(examples)!=2:
            continue
        if db_id not in catalogs:
            try:
                catalogs[db_id]=DatabaseCatalog.from_sqlite(_resolve_database(root,db_id))
            except (FileNotFoundError,OSError):
                continue
        schema,meta=select_schema_context(
            catalogs[db_id],root,db_id,question,tokenizer=legacy_tokens)
        if len(schema)>MAX_CONTEXT_CHARS:
            continue
        all_candidates.append({
            "case_id":cid,"database_id":db_id,"question":question,
            "evidence":str(row.get("evidence") or ""),
            "schema_sha256":meta["schema_context_sha256"],
            "training_example_sha256":[x["fingerprint"] for x in examples],
            "examples":examples,
        })
    picks,seen_ids,db_count=[],set(),Counter()
    for cap in (1,2):
        for row in all_candidates:
            if len(picks)>=N:
                break
            cid,db=row["case_id"],row["database_id"]
            if cid in seen_ids or db_count[db]>=cap:
                continue
            picks.append(row)
            seen_ids.add(cid)
            db_count[db]+=1
    if len(picks)!=N or len(db_count)<6:
        raise ValueError("not enough source-independent eligible BIRD12 from >=6 physical databases")
    identity=[(x["case_id"],x["database_id"],x["training_example_sha256"]) for x in picks]
    return {
        "protocol":PROTOCOL,"source_sha256":SOURCE_SHA,
        "model":MODEL,"model_digest":MODEL_DIGEST,
        "spider_train_parquet_sha256":train_sha,
        "training_rows_verified":len(train),
        "training_dbs_verified":len({x["db_id"] for x in train}),
        "BIRD_dev_previous_ids_excluded":112,
        "selection_seed":SEED,"count":N,"physical_dbs":len(db_count),
        "cohort_sha256":sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest(),
        "gold_used_for_selection":False,"gaia_or_bird_dev_supervision_used":False,
        "model_calls_before_freeze":0,"same_original_qwen9b":True,
        "no_sql_training":True,"public_corpora_pretraining_overlap_unknown":True,
        "cases":picks,
    }


def run_shard(source,root,manifest,shard):
    if shard not in range(SHARDS):
        raise ValueError("invalid shard")
    data=checked_source(source)
    lookup={case_id(r,i):r for i,r in enumerate(data)}
    cfg=json.loads(manifest.read_text())
    if (cfg.get("protocol")!=PROTOCOL or cfg.get("source_sha256")!=SOURCE_SHA
        or cfg.get("model_digest")!=MODEL_DIGEST or cfg.get("count")!=N
        or cfg.get("gold_used_for_selection") is not False
        or len(cfg.get("cases",[]))!=N):
        raise ValueError("frozen prospectively selected cohort changed")
    if _ollama_model_digest("http://127.0.0.1:11434",MODEL)!=MODEL_DIGEST:
        raise ValueError("original model digest drift")
    batch=cfg["cases"][shard*PER_SHARD:(shard+1)*PER_SHARD]
    base=PinnedQwen()
    pending=[]
    for sample in batch:
        cid,db=sample["case_id"],sample["database_id"]
        row=lookup[cid]
        if str(row["db_id"])!=db or row["question"]!=sample["question"]:
            raise ValueError("source sample differs from frozen source")
        database=_resolve_database(root,db)
        catalog=DatabaseCatalog.from_sqlite(database)
        schema,meta=select_schema_context(
            catalog,root,db,sample["question"],tokenizer=legacy_tokens)
        if meta["schema_context_sha256"]!=sample["schema_sha256"]:
            raise ValueError("original schema context drift")
        examples=sample["examples"]
        if (len(examples)!=2 or len({e["db_id"] for e in examples})!=2
            or any(x["db_id"]==db for x in examples)
            or [x["fingerprint"] for x in examples]!=sample["training_example_sha256"]):
            raise ValueError("training examples drift / source leakage")
        case={"case_id":cid,"database_id":db,"arms":{}}
        for arm,provider in (
            ("original",base),
            ("two_train_examples",TwoExampleFirstCall(base,examples)),
        ):
            start=monotonic()
            got=run_guarded(database=database,provider=provider,
                question=sample["question"],schema_context=schema,
                evidence=sample["evidence"].strip() or None,
                max_repairs=2,max_rows=50000)
            good=got.status=="ok" and not got.truncated
            case["arms"][arm]={
                "sql":got.candidate.sql if got.candidate else "",
                "status":got.status,
                "execution_success":good,
                "repair_attempt":got.candidate.attempt if got.candidate else None,
                "seconds":round(monotonic()-start,2),
                "_rows":list(got.rows) if good else None,
            }
        pending.append(case)
        print("QWEN9B_SPIDER_FEWSHOT_GENERATED",shard,cid,flush=True)
    for case in pending:
        row=lookup[case["case_id"]]
        try:
            gold=execute_readonly(_resolve_database(root,case["database_id"]),
                                  str(row["SQL"]),max_rows=50000)
            if gold.truncated:raise ValueError("truncated gold")
            case["reference_scorable"]=True
            for got in case["arms"].values():
                actual=got.pop("_rows")
                ok=got["execution_success"] and actual is not None
                got["set_ex"]=bool(ok and set(map(tuple,actual))==set(map(tuple,gold.rows)))
                got["bag_ex"]=bool(ok and Counter(map(tuple,actual))==Counter(map(tuple,gold.rows)))
        except (SQLExecutionError,ValueError) as exc:
            case["reference_scorable"]=False
            case["reference_error"]=type(exc).__name__
            for got in case["arms"].values():
                got.pop("_rows",None)
                got["set_ex"]=None
                got["bag_ex"]=None
    return {"protocol":PROTOCOL,"shard":shard,
        "model_digest":MODEL_DIGEST,"cohort_sha256":cfg["cohort_sha256"],
        "cases":pending}


def aggregate(shards):
    if len(shards)!=SHARDS or sorted(x["shard"] for x in shards)!=list(range(SHARDS)):
        raise ValueError("need all 4 paired Qwen shards")
    cohort=shards[0]["cohort_sha256"]
    if any(x.get("protocol")!=PROTOCOL or x.get("model_digest")!=MODEL_DIGEST
           or x.get("cohort_sha256")!=cohort or len(x["cases"])!=PER_SHARD
           for x in shards):
        raise ValueError("different source/model/cases in shard")
    rows=[r for x in shards for r in x["cases"]]
    if len({r["case_id"] for r in rows})!=N:
        raise ValueError("duplicate BIRD case")
    valid=[r for r in rows if r["reference_scorable"]]
    score=lambda arm,key:sum(bool(r["arms"][arm][key]) for r in valid)
    rescue=[r["case_id"] for r in valid if not r["arms"]["original"]["set_ex"]
            and r["arms"]["two_train_examples"]["set_ex"]]
    regress=[r["case_id"] for r in valid if r["arms"]["original"]["set_ex"]
             and not r["arms"]["two_train_examples"]["set_ex"]]
    resc_dbs={r["database_id"] for r in valid if r["case_id"] in rescue}
    return {
        "protocol":PROTOCOL,"model":MODEL,"model_digest":MODEL_DIGEST,
        "cohort_sha256":cohort,"n":N,"scorable":len(valid),
        "original_set_ex":score("original","set_ex"),
        "two_train_examples_set_ex":score("two_train_examples","set_ex"),
        "original_bag_ex":score("original","bag_ex"),
        "two_train_examples_bag_ex":score("two_train_examples","bag_ex"),
        "rescue_ids":rescue,"regression_ids":regress,
        "distinct_rescue_dbs":len(resc_dbs),"net_rescues":len(rescue)-len(regress),
        "positive_screen_only":bool(len(valid)>=11 and len(rescue)-len(regress)>=3
             and len(resc_dbs)>=3 and not regress
             and score("two_train_examples","bag_ex")>=score("original","bag_ex")),
        "promoted":False,"no_sql_training":True,"no_model_swap":True,
        "BIRD100_original_EX_unchanged":39,
        "public_dev_not_hidden_test":True,
        "cases":rows,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path)
    p.add_argument("--root",type=Path)
    p.add_argument("--reference",type=Path,default=Path("runs/bird-paired-dev-v5/paired.json"))
    p.add_argument("--previous-cohort",type=Path)
    p.add_argument("--spider-parquet",type=Path)
    p.add_argument("--manifest",type=Path)
    p.add_argument("--shard",type=int)
    p.add_argument("--inputs",nargs="+",type=Path)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError("refuse to overwrite frozen experiment")
    if args.inputs:
        result=aggregate([json.loads(f.read_text()) for f in args.inputs])
        print("QWEN9B_SPIDER_TRAIN_FEWSHOT_STRICT_EX_FINAL",
              json.dumps({k:v for k,v in result.items() if k!="cases"},sort_keys=True),flush=True)
    elif args.shard is not None:
        if not args.source or not args.root or not args.manifest:
            p.error("inference requires --source --root --manifest")
        result=run_shard(args.source,args.root,args.manifest,args.shard)
    else:
        if not args.source or not args.root or not args.previous_cohort or not args.spider_parquet:
            p.error("preregister requires BIRD physical source, prior cohort and Spider train parquet")
        import pandas as pd
        data=pd.read_parquet(args.spider_parquet)
        required={"db_id","question","query"}
        if not required.issubset(data.columns):
            raise ValueError(f"not actual Spider question/SQL data: {data.columns}")
        result=source_manifest(args.source,args.root,args.reference,args.previous_cohort,
            data.to_dict("records"),sha256(args.spider_parquet.read_bytes()).hexdigest())
        print("QWEN9B_SPIDER_FEWSHOT_GOLD_BLIND_PREREGISTERED",
              json.dumps({k:v for k,v in result.items() if k!="cases"},sort_keys=True),flush=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False)+"\n")


if __name__=="__main__":
    main()
