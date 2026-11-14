"""Qwen-only, no-SQL-training relational-metadata experiment on fresh BIRD TRAIN DBs.

Baseline: existing Qwen3.5-9B Guarded, original lexical schema.
Treatment: IDENTICAL model, decoding, question, repair and scoring, plus only
declared SQL key/FK metadata from SQLite PRAGMA (no inferred intent).

Cohort: third set of 10 BIRD TRAIN databases; excludes entire DEV and BOTH
previous 20-DB TRAIN source-shift experiment cohorts. Two questions/DB picked
by deterministic hash independent of the physical constraints and SQL gold.
The dataset itself is public and not a hidden final test.

No SQL reference is read in any generation or experimental selection step.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha1, sha256
import json
from pathlib import Path
import time
from urllib.parse import quote
from urllib.request import urlopen

from evaluation.relational_contract import inspect_constraints
from evaluation.run_bird import _ollama_model_digest
from evaluation.schema_context import select_schema_context
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.execution import SQLExecutionError, execute_readonly
from semantic_sql.inference import run_guarded
from semantic_sql.providers import OllamaProvider

MIRROR="https://huggingface.co"
DATASET="Emulated-Inc/text-to-sql-with-evidence-training-pool"
REVISION="cd2c4ea321075aa53ce7dfe52deaaffa505d046a"
TRAIN_SHA="cd5884489dffd7c76d73509b396bd919fbe8b486efbc4ba2f48f89460de7d69f"
DEV_SHA="630272f2b1c44d8cef2c3b246f623355cf0bbc1e832c81061df895530dfc2f06"
MODEL="qwen3.5:9b-q4_K_M"
MODEL_DIGEST="56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
FIRST_TWO_COHORTS_SEED="bird-generator-transfer-20261114"
THIS_SEED="qwen-relational-schema-db-shift-20261115-v1"
N_SHARDS=5
ARMS=("original_guarded","declared_relations_guarded")


def _checked(path: Path, expected: str) -> list[dict]:
    if sha256(path.read_bytes()).hexdigest()!=expected:
        raise ValueError("pinned source SHA256 mismatch")
    rows=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows,list) or not rows:
        raise ValueError("invalid source row structure")
    return rows


def _nq(q: str) -> str:
    return " ".join(str(q).casefold().split())


def select_manifest(train: list[dict], dev: list[dict], remote: dict) -> dict:
    """Fixed third 10-DB cohort WITHOUT inspecting labels, keys or row values."""
    used_db={str(row["db_id"]) for row in dev}
    used_q={_nq(row["question"]) for row in dev}
    groups=defaultdict(list)
    for index,row in enumerate(train):
        db=str(row["db_id"])
        if db in used_db or db not in remote or remote[db]["size"]>128*1024**2:
            continue
        question=str(row.get("question") or "")
        key=_nq(question)
        if not key or key in used_q:
            continue
        used_q.add(key)
        groups[db].append({
            "case_id":index,"database_id":db,
            "question":question,"evidence":str(row.get("evidence") or ""),
        })
    eligible=sorted(
        (db for db,items in groups.items() if len(items)>=2),
        key=lambda db:sha256(f"{FIRST_TWO_COHORTS_SEED}:{db}".encode()).digest(),
    )
    if len(eligible)<50:
        raise ValueError("need at least 50 distinct eligible BIRD train-only databases")
    previously_explored=set(eligible[:40])
    fresh=sorted(eligible[40:],key=lambda db:sha256(
        f"{THIS_SEED}:{db}".encode()).digest())[:10]
    if len(fresh)!=10 or set(fresh)&previously_explored:
        raise AssertionError("no ten never-previous-train DBs")
    chosen=[]
    for db in fresh:
        examples=sorted(
            groups[db],key=lambda r:sha256(
                f"{THIS_SEED}:{r['case_id']}".encode()).digest()
        )[:2]
        chosen.extend(examples)
    if (len(chosen)!=20 or len({row["case_id"] for row in chosen})!=20
        or len({row["database_id"] for row in chosen})!=10):
        raise ValueError("cohort identity/DB coverage changed")
    ids=[(r["case_id"],r["database_id"]) for r in chosen]
    return {
        "protocol":"qwen9b-declared-relational-contract-dbshift20-v1",
        "model":MODEL,"model_digest":MODEL_DIGEST,
        "train_sha256":TRAIN_SHA,"dev_sha256":DEV_SHA,
        "previous_cohort_db_count":len(previously_explored),
        "new_db_count":10,"cases_per_db":2,
        "question_selection_uses_only_question_and_db_id":True,
        "case_identity_sha256":sha256(json.dumps(
            ids,separators=(",",":")).encode()).hexdigest(),
        "selected":chosen,
    }


def remote_manifest() -> dict:
    url=f"{MIRROR}/api/datasets/{DATASET}/tree/{REVISION}/raw/bird/train_databases?recursive=true&limit=1000"
    with urlopen(url,timeout=90) as resp:
        rows=json.load(resp)
    by_db={Path(row["path"]).stem:row for row in rows if row["path"].endswith(".sqlite")}
    if len(by_db)!=51:
        raise ValueError("BIRD database mirror changed")
    return by_db


def _checked_sqlite(data: bytes, record: dict) -> bool:
    if len(data)!=record["size"]:
        return False
    if "lfs" in record:
        return sha256(data).hexdigest()==record["lfs"]["oid"]
    return sha1(f"blob {len(data)}\0".encode()+data).hexdigest()==record["oid"]


def _download_db(work: Path, dbid: str, record: dict) -> Path:
    path=work/dbid/(dbid+".sqlite")
    path.parent.mkdir(parents=True,exist_ok=True)
    url=f"{MIRROR}/datasets/{DATASET}/resolve/{REVISION}/raw/bird/train_databases/{quote(dbid)}/{quote(dbid)}.sqlite"
    with urlopen(url,timeout=240) as response:
        data=response.read(record["size"]+1)
    if not _checked_sqlite(data,record):
        raise ValueError(f"source physical SQLite checksum drift: {dbid}")
    path.write_bytes(data)
    return path


def _answer(res) -> dict:
    good=res.status=="ok" and not res.truncated
    return {
        "sql":res.candidate.sql if res.candidate else "",
        "status":res.status,
        "execution_success":bool(good),
        "rows":res.rows if good else (),
        "model_repairs":max(0,res.candidate.attempt) if res.candidate else 0,
    }


def run_shard(train_path: Path, dev_path: Path, shard: int, work: Path) -> dict:
    if not 0<=shard<N_SHARDS:
        raise ValueError("unknown shard")
    train=_checked(train_path,TRAIN_SHA)
    dev=_checked(dev_path,DEV_SHA)
    files=remote_manifest()
    manifest=select_manifest(train,dev,files)
    picked=manifest["selected"][4*shard:4*(shard+1)]
    if len(picked)!=4:
        raise ValueError("incomplete source-blind 4-case shard")
    digest=_ollama_model_digest("http://127.0.0.1:11434",MODEL)
    if digest!=MODEL_DIGEST:
        raise ValueError("different Qwen checkpoint")
    cache={}
    for dbid in sorted({x["database_id"] for x in picked}):
        path=_download_db(work,dbid,files[dbid])
        cache[dbid]=(path,DatabaseCatalog.from_sqlite(path))

    provider=OllamaProvider(
        model=MODEL,base_url="http://127.0.0.1:11434",
        timeout_s=900,temperature=0,seed=42,num_predict=1024,
    )
    records=[]
    for row in picked:
        dbid=row["database_id"]
        db,catalog=cache[dbid]
        schema,diagnostics=select_schema_context(catalog,work,dbid,row["question"])
        metadata=inspect_constraints(db,selected_tables=set(diagnostics["selected_tables"]))
        context=metadata.render()
        treatment=schema+("\n"+context if context else "")
        evidence=row["evidence"] or None
        arms={}
        for arm,chosen_schema in ((ARMS[0],schema),(ARMS[1],treatment)):
            started=time.monotonic()
            result=run_guarded(
                database=db,provider=provider,question=row["question"],
                schema_context=chosen_schema,evidence=evidence,
                max_repairs=2,max_rows=50_000,
            )
            details=_answer(result)
            details["latency_s"]=round(time.monotonic()-started,2)
            arms[arm]=details
        records.append({
            "case_id":row["case_id"],"db_id":dbid,
            "schema_changed":bool(context),
            "declared_key_count":len(metadata.keys),
            "fk_count":len(metadata.relations),
            "ignored_partial_or_expression_indexes":metadata.ignored_partial_or_expression_indexes,
            "base_context_sha256":sha256(schema.encode()).hexdigest(),
            "treatment_context_sha256":sha256(treatment.encode()).hexdigest(),
            "arms":arms,
        })
        print("QWEN_RELATIONAL_CONTRACT_GENERATED",shard,row["case_id"],
              "keys",len(metadata.keys),"fks",len(metadata.relations),flush=True)

    # GOLD is accessed ONLY after all original + treatment generation.
    for row in records:
        db=cache[row["db_id"]][0]
        try:
            gold=execute_readonly(db,str(train[row["case_id"]]["SQL"]),max_rows=50_000)
            if gold.truncated:
                raise ValueError("gold query truncated")
            gold_rows=set(gold.rows)
            gold_bag=Counter(gold.rows)
            scorable=True
        except (SQLExecutionError, ValueError) as exc:
            scorable=False
            row["gold_error_kind"]=type(exc).__name__
        row["reference_scorable"]=scorable
        for arm in row["arms"].values():
            output=arm.pop("rows")
            arm["strict_ex"]=bool(scorable and arm["execution_success"] and set(output)==gold_rows)
            arm["bag"]=bool(scorable and arm["execution_success"] and Counter(output)==gold_bag)
    return {
        "protocol":manifest["protocol"],"shard":shard,
        "cohort_sha256":manifest["case_identity_sha256"],
        "model_digest":digest,
        "cases":records,
        "constraints_only_from_ddl":True,
        "no_training":True,
        "question_selection_uses_only_question_and_db_id":True,
    }


def aggregate(shards: list[dict]) -> dict:
    if len(shards)!=N_SHARDS or sorted(s["shard"] for s in shards)!=list(range(N_SHARDS)):
        raise ValueError("need all 5 disjoint shards")
    first=shards[0]
    if any(
        s["protocol"]!=first["protocol"]
        or s["cohort_sha256"]!=first["cohort_sha256"]
        or s["model_digest"]!=first["model_digest"]
        or not s["no_training"] or not s["constraints_only_from_ddl"]
        for s in shards
    ):
        raise ValueError("model, cohort or mechanism drift")
    rows=[r for s in sorted(shards,key=lambda s:s["shard"]) for r in s["cases"]]
    if len(rows)!=20 or len({r["case_id"] for r in rows})!=20 or len({r["db_id"] for r in rows})!=10:
        raise ValueError("incomplete independent-db paired cohort")
    old,new=ARMS
    answer={
        "protocol":first["protocol"],"model_digest":first["model_digest"],
        "total_cases":20,"different_physical_databases":10,
        "reference_scorable":sum(r["reference_scorable"] for r in rows),
        "metadata_changed":sum(r["schema_changed"] for r in rows),
        "metadata_keys":sum(r["declared_key_count"] for r in rows),
        "metadata_relations":sum(r["fk_count"] for r in rows),
        "arms":{arm:{
            "strict_ex":sum(r["arms"][arm]["strict_ex"] for r in rows),
            "bag":sum(r["arms"][arm]["bag"] for r in rows),
            "execution_success":sum(r["arms"][arm]["execution_success"] for r in rows),
            "repair_count":sum(r["arms"][arm]["model_repairs"] for r in rows),
            "sum_latency_s":round(sum(r["arms"][arm]["latency_s"] for r in rows),1),
        } for arm in ARMS},
        "rescues":sum(not r["arms"][old]["strict_ex"] and r["arms"][new]["strict_ex"] for r in rows),
        "regressions":sum(r["arms"][old]["strict_ex"] and not r["arms"][new]["strict_ex"] for r in rows),
        "unchanged_schema_sql_divergences":sum(
            not r["schema_changed"] and r["arms"][old]["sql"]!=r["arms"][new]["sql"] for r in rows
        ),
        "improved_distinct_databases":len({
            r["db_id"] for r in rows
            if not r["arms"][old]["strict_ex"] and r["arms"][new]["strict_ex"]
        }),
        "cases":rows,
        "promotion_allowed":False,
        "boundary":(
            "Public BIRD TRAIN pool; ten DBs excluded from DEV and the two prior "
            "20-DB BIRD train cohorts, but public data and small N=20 are not a "
            "hidden independent benchmark. DDL metadata is source-only, no model "
            "training or category-specific handling. Do not promote based on a "
            "single paired set."
        )
    }
    return answer


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train",type=Path)
    p.add_argument("--dev",type=Path)
    p.add_argument("--shard",type=int)
    p.add_argument("--work",type=Path,default=Path("relational-trial-db"))
    p.add_argument("--inputs",type=Path,nargs="+")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError("refusing to overwrite an existing trial result")
    if a.inputs:
        result=aggregate([json.loads(x.read_text(encoding="utf-8")) for x in a.inputs])
        print("QWEN_RELATIONAL_CONTRACT_FINAL",
              json.dumps({k:v for k,v in result.items() if k!="cases"},sort_keys=True),flush=True)
    else:
        if a.train is None or a.dev is None or a.shard is None:
            p.error("one shard needs --train --dev --shard")
        result=run_shard(a.train,a.dev,a.shard,a.work)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
