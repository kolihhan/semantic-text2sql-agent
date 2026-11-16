"""BIRD TRAIN only: source-verified database-disjoint Qwen3.5-9B SQL SFT candidates.

This is a CPU source/data quality screen, NOT an adapter, not a SQL EX
improvement, and does not touch the reserved BIRD DEV300 holdout.
No stronger teacher. The label is original publisher reference SQL.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha1, sha256
import json
from pathlib import Path
import re
import sqlite3
from urllib.parse import quote
from urllib.request import urlopen

from evaluation.schema_context import select_schema_context
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.direct import _SYSTEM
from semantic_sql.verifier import verify_sql_preflight

MIRROR = "https://huggingface.co"
DATASET = "Emulated-Inc/text-to-sql-with-evidence-training-pool"
REVISION = "cd2c4ea321075aa53ce7dfe52deaaffa505d046a"
TRAIN_SHA = "cd5884489dffd7c76d73509b396bd919fbe8b486efbc4ba2f48f89460de7d69f"
DEV_SHA = "ffd8018378ddb1a8794753e0a31cfc81862ff7318a5184c22f3dc4ce03a03feb"
SEED = "birdtrain-native-qwen9b-sft-db-separated-v1-20261116"
N_TRAIN_DBS = 16
N_VALID_DBS = 4
QUESTIONS_PER_DB = 16
DB_MAX_BYTES = 96 * 1024 * 1024

def normalized(question: str) -> str:
    return " ".join(str(question).casefold().split())


def load_checked(path: Path, digest: str) -> list[dict]:
    data=path.read_bytes()
    if sha256(data).hexdigest() != digest:
        raise ValueError(f"Source SHA mismatch: {path.name}")
    rows=json.loads(data)
    if not isinstance(rows,list) or not rows:
        raise ValueError("Source does not contain nonempty list")
    return rows


def fetch_manifest() -> dict:
    url=(f"{MIRROR}/api/datasets/{DATASET}/tree/{REVISION}/raw/bird/train_databases"
         "?recursive=true&limit=1000")
    with urlopen(url,timeout=60) as stream:
        payload=json.load(stream)
    result={Path(row["path"]).stem: row for row in payload if str(row.get("path","")).endswith(".sqlite")}
    if len(result)!=51:
        raise ValueError(f"Expected complete pinned 51 TRAIN DBs, got {len(result)}")
    return result


def plan_cohorts(train: list[dict], dev: list[dict], manifest: dict) -> dict[str,list[dict]]:
    dev_dbs={r["db_id"] for r in dev}
    dev_questions={normalized(r["question"]) for r in dev}
    by_db: dict[str,list[dict]] = defaultdict(list)
    seen=set(dev_questions)
    for index,row in enumerate(train):
        db=row.get("db_id")
        question=normalized(row.get("question",""))
        if (db not in manifest or db in dev_dbs or not question or
            question in seen or manifest[db]["size"] > DB_MAX_BYTES):
            continue
        seen.add(question)
        by_db[db].append({"source_index":index,"db_id":db,
                          "question":row["question"],"evidence":row.get("evidence") or ""})
    eligible=[db for db,rows in by_db.items() if len(rows)>=QUESTIONS_PER_DB]
    ordered=sorted(eligible,key=lambda db:sha256(f"{SEED}:db:{db}".encode()).hexdigest())
    required=N_TRAIN_DBS+N_VALID_DBS
    if len(ordered)<required:
        raise ValueError(f"Only {len(ordered)} eligible train DBs, require {required}")
    partition_dbs={"train":ordered[:N_TRAIN_DBS],
                   "validation":ordered[N_TRAIN_DBS:required]}
    result={}
    for partition,dbs in partition_dbs.items():
        result[partition]=[]
        for db in dbs:
            candidates=sorted(by_db[db],key=lambda row:sha256(
                f"{SEED}:question:{row['source_index']}".encode()).hexdigest())
            result[partition].extend(candidates[:QUESTIONS_PER_DB])
    if ({r["db_id"] for r in result["train"]} &
        {r["db_id"] for r in result["validation"]}):
        raise ValueError("TRAIN/validation databases overlap")
    if set(partition_dbs["train"]+partition_dbs["validation"]) & dev_dbs:
        raise ValueError("TRAIN and BIRD DEV databases overlap")
    ids=[r["source_index"] for part in result.values() for r in part]
    if len(ids)!=len(set(ids)):
        raise ValueError("Duplicate source row IDs across cohorts")
    return result


def _valid_binary(data:bytes,meta:dict)->bool:
    if len(data)!=meta["size"]:
        return False
    if "lfs" in meta:
        return sha256(data).hexdigest()==meta["lfs"]["oid"]
    return sha1(f"blob {len(data)}\0".encode()+data).hexdigest()==meta["oid"]


def fetch_database(db:str,meta:dict,work:Path)->Path:
    url=(f"{MIRROR}/datasets/{DATASET}/resolve/{REVISION}/raw/bird/"
         f"train_databases/{quote(db)}/{quote(db)}.sqlite")
    with urlopen(url,timeout=180) as stream:
        data=stream.read(meta["size"]+1)
    if not _valid_binary(data,meta):
        raise ValueError(f"{db}: database content SHA mismatch")
    out=work/db/f"{db}.sqlite"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_bytes(data)
    return out


def validate_gold_sql(db:Path,sql:str) -> str|None:
    if not isinstance(sql,str) or not re.match(r"^\s*(select|with)\b",sql,re.I):
        return "NON_READONLY_REFERENCE"
    if sql.count(";")>1 or (";" in sql.rstrip("; \n\t")):
        return "MULTI_STATEMENT_REFERENCE"
    verdict=verify_sql_preflight(db,sql)
    if not verdict.ok:
        return "PREFLIGHT_REJECTED"
    try:
        conn=sqlite3.connect(f"file:{db}?mode=ro",uri=True)
        try:
            conn.execute("EXPLAIN QUERY PLAN "+sql).fetchall()
        finally:
            conn.close()
    except (sqlite3.Error,ValueError):
        return "SQLITE_COMPILE_FAILURE"
    return None


def make_example(source:dict,catalog:DatabaseCatalog,root:Path) -> dict|None:
    question=str(source["question"]).strip()
    schema,_=select_schema_context(catalog,root,source["db_id"],question)
    evidence=str(source["evidence"]).strip()
    user=f"Question: {question}\nSchema context:\n{schema}"
    if evidence:
        user+=f"\nExternal evidence:\n{evidence}"
    if len(user)>16000:
        return None
    return {
        "source_index":source["source_index"],
        "source_db_id":source["db_id"],
        "input_sha256":sha256(user.encode()).hexdigest(),
        "messages":[
            {"role":"system","content":_SYSTEM},
            {"role":"user","content":user},
            {"role":"assistant","content":str(source["SQL"]).strip()},
        ]
    }


def collect(train_path:Path,dev_path:Path,work:Path,out:Path) -> dict:
    train=load_checked(train_path,TRAIN_SHA)
    dev=load_checked(dev_path,DEV_SHA)
    manifest=fetch_manifest()
    plan=plan_cohorts(train,dev,manifest)
    counts=Counter()
    corpus={"train":[],"validation":[]}
    for part,rows in plan.items():
        by_db=defaultdict(list)
        for row in rows:
            by_db[row["db_id"]].append(row)
        for dbid in sorted(by_db):
            database=fetch_database(dbid,manifest[dbid],work)
            catalog=DatabaseCatalog.from_sqlite(database)
            for row in by_db[dbid]:
                original=train[row["source_index"]]
                reason=validate_gold_sql(database,original.get("SQL",""))
                if reason:
                    counts[f"{part}_{reason}"]+=1
                    continue
                source={**row,"SQL":original["SQL"]}
                record=make_example(source,catalog,work)
                if record is None:
                    counts[f"{part}_LONG_CONTEXT"]+=1
                    continue
                corpus[part].append(record)
    if len(corpus["train"])<180 or len(corpus["validation"])<40:
        raise RuntimeError("Insufficient source-valid clean TRAIN-only reference SQL examples")
    train_dbs={r["source_db_id"] for r in corpus["train"]}
    val_dbs={r["source_db_id"] for r in corpus["validation"]}
    if train_dbs&val_dbs or train_dbs&{r["db_id"] for r in dev} or val_dbs&{r["db_id"] for r in dev}:
        raise ValueError("Disjoint database gate broken")
    out.mkdir(parents=True,exist_ok=True)
    for part,rows in corpus.items():
        (out/f"{part}.jsonl").write_text(
            "".join(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n" for r in rows),
            encoding="utf-8"
        )
    report={
        "status":"SOURCE_VERIFIED_NO_MODEL_TRAINING",
        "train_source_sha256":TRAIN_SHA,"dev_source_sha256":DEV_SHA,
        "selection_seed":SEED,
        "train_n":len(corpus["train"]),"validation_n":len(corpus["validation"]),
        "train_db_n":len(train_dbs),"validation_db_n":len(val_dbs),
        "dev_db_overlap":0,"train_valid_db_overlap":0,
        "drop_reasons":dict(sorted(counts.items())),
        "train_sha256":sha256((out/"train.jsonl").read_bytes()).hexdigest(),
        "validation_sha256":sha256((out/"validation.jsonl").read_bytes()).hexdigest(),
        "training_authorization":"BIRD TRAIN-only QLoRA on unchanged Qwen3.5-9B, not launched here",
        "heldout_bird_dev_v2_accessed":False,
        "gold_sql_visible_to_any_model":False,
        "actual_model_training_steps":0,
        "actual_accuracy_improvement":None,
    }
    (out/"source-verification.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--dev",type=Path,required=True)
    p.add_argument("--work",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(collect(a.train,a.dev,a.work,a.out),indent=2,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
