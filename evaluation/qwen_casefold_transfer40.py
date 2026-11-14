"""Independent-of-DEV100 casefold transfer on frozen Qwen9B schema16 + physical24.

The hypothesis and deterministic rewrite were fixed after DEV100 inspection;
this source-shift audit is exploratory replication, NOT a hidden benchmark.
No fresh Qwen inference, no SQL training, gold read only after candidate freeze.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from urllib.parse import quote

from evaluation.qwen_literal_grounding_audit import (
    extract_simple_equalities, check_literal, _ident,
)
from evaluation.qwen_casefold_counterfactual import rewrite_verified_casefold
from evaluation.run_bird import _resolve_database
from semantic_sql.execution import execute_readonly, SQLExecutionError
from semantic_sql.verifier import verify_sql_preflight

MODEL_DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
SCHEMA_PROTOCOL = "qwen-schema-tokenizer-paired16-v1"
PHYSICAL_PROTOCOL = "qwen-physical-db-facts24-v1"
SOURCE_SHA = "ffd8018378ddb1a8794753e0a31cfc81862ff7318a5184c22f3dc4ce03a03feb"
COHORT_GROUPS={"schema16":(4,4,SCHEMA_PROTOCOL,"legacy"),"physical24":(6,4,PHYSICAL_PROTOCOL,"baseline")}


def _ensure_cohort(parts: list[dict], label: str) -> list[dict]:
    count, size, protocol, arm=COHORT_GROUPS[label]
    if len(parts)!=count or sorted(p["shard"] for p in parts)!=list(range(count)):
        raise ValueError("missing Qwen transfer shards")
    cohort_sha=parts[0].get("cohort_sha256")
    if (not cohort_sha or any(p["cohort_sha256"]!=cohort_sha
        or p.get("protocol")!=protocol or p.get("model_digest")!=MODEL_DIGEST
        or len(p["cases"])!=size for p in parts)):
        raise ValueError("model/protocol/cohort drift")
    rows=[]
    for part in sorted(parts,key=lambda v:v["shard"]):
        for c in part["cases"]:
            if not c.get("reference_scorable",False):
                raise ValueError("gold invalid in original pinned cohort")
            rows.append({
                "group":label,
                "case_id":str(c["case_id"]),
                "database_id":str(c.get("database_id") or c.get("db_id")),
                "question":str(c.get("question") or ""),
                "sql":str(c["arms"][arm]["sql"]),
                "original_archived_ex":bool(c["arms"][arm]["ex"]),
                "original_archived_bag":bool(c["arms"][arm]["bag"]),
            })
    return rows


def assemble(schema: list[dict], physical: list[dict], excluded100: set[str]) -> list[dict]:
    rows=_ensure_cohort(schema,"schema16")+_ensure_cohort(physical,"physical24")
    ids=[x["case_id"] for x in rows]
    if len(rows)!=40 or len(set(ids))!=40:
        raise ValueError("case overlaps between new Qwen transfer cohorts")
    if set(ids)&excluded100:
        raise ValueError("frozen100 case leaked into independent cohort")
    return rows


def summarize(rows: list[dict]) -> dict:
    return {
        "n":len(rows),"changed":sum(x["revised"] for x in rows),
        "eligible":sum(x["eligible"] for x in rows),
        "original_ex":sum(x["original_ex"] for x in rows),
        "new_ex":sum(x["new_ex"] for x in rows),
        "original_bag":sum(x["original_bag"] for x in rows),
        "new_bag":sum(x["new_bag"] for x in rows),
        "rescues":sum(not x["original_ex"] and x["new_ex"] for x in rows),
        "regressions":sum(x["original_ex"] and not x["new_ex"] for x in rows),
        "groups": {
            name: {
                "n":sum(x["group"]==name for x in rows),
                "changed":sum(x["group"]==name and x["revised"] for x in rows),
                "original_ex":sum(x["group"]==name and x["original_ex"] for x in rows),
                "new_ex":sum(x["group"]==name and x["new_ex"] for x in rows),
            } for name in COHORT_GROUPS
        }
    }


def _all_tables(path: Path) -> dict[str, tuple[str,...]]:
    uri="file:"+quote(str(path.resolve()))+"?mode=ro"
    with sqlite3.connect(uri,uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        return {name:tuple(col[1] for col in db.execute(f"PRAGMA table_info({_ident(name)})"))
                for (name,) in db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}


def transfer(schema: list[dict], physical: list[dict], frozen100: Path,
             source: Path, dbroot: Path) -> dict:
    old=json.loads(frozen100.read_text(encoding="utf-8"))
    if old.get("status")!="complete" or old.get("sample_size")!=100:
        raise ValueError("historical 100 protection invalid")
    rows=assemble(schema,physical,{str(r["case_id"]) for r in old["cases"]})
    cache={}
    # Only the frozen Qwen SQL, physical SQLite DB, and case-blind rule below
    # contribute to proposed counterfactuals. Gold not yet loaded here.
    decisions=[]
    for item in rows:
        dbid=item["database_id"]
        if dbid not in cache:
            p=_resolve_database(dbroot,dbid)
            cache[dbid]=(p,_all_tables(p))
        db,tables=cache[dbid]
        probes=[]
        for pred in extract_simple_equalities(item["sql"],tables):
            r=check_literal(db,pred)
            probes.append({"table":pred.table,"column":pred.column,"value":pred.value,
                           "status":r.status,"replacement":r.replacement,
                           "operator":r.operator,"interrupted":r.interrupted})
        candidate=rewrite_verified_casefold(item["sql"],probes)
        decisions.append({
            **item,
            "eligible":any(p["status"]=="casefold_match" for p in probes),
            "revised":candidate is not None,
            "sql_sha256":sha256(item["sql"].encode()).hexdigest(),
            "counterfactual_sql":candidate,
            "probes":probes,
        })
    revision_sha=sha256(json.dumps(
        [(x["case_id"],x["counterfactual_sql"]) for x in decisions],
        separators=(",",":"),ensure_ascii=False,
    ).encode()).hexdigest()

    # All question/candidate decisions frozen. Read benchmark gold only now.
    if sha256(source.read_bytes()).hexdigest()!=SOURCE_SHA:
        raise ValueError("new-cohort BIRD data source drift")
    raw=json.loads(source.read_text(encoding="utf-8"))
    by_id={str(r.get("question_id") or r.get("id") or i):r
           for i,r in enumerate(raw)}
    results=[]
    for row in decisions:
        raw_case=by_id[row["case_id"]]
        if str(raw_case["db_id"])!=row["database_id"]:
            raise ValueError("question source drift")
        if row["question"] and row["question"]!=str(raw_case["question"]):
            raise ValueError("question source drift")
        db=cache[row["database_id"]][0]
        try:
            gold=execute_readonly(db,raw_case["SQL"],max_rows=50_000)
            original=execute_readonly(db,row["sql"],max_rows=50_000)
            if gold.truncated or original.truncated:
                raise ValueError("reference/baseline query truncated")
        except (SQLExecutionError,ValueError) as exc:
            raise ValueError("original physically scored Qwen case drift "+row["case_id"]) from exc
        original_ex=set(original.rows)==set(gold.rows)
        original_bag=Counter(original.rows)==Counter(gold.rows)
        if original_ex!=row["original_archived_ex"] or original_bag!=row["original_archived_bag"]:
            raise ValueError("old Qwen score drift "+row["case_id"])
        new_ex,new_bag=original_ex,original_bag
        changed=False
        if row["counterfactual_sql"] is not None:
            statement=row["counterfactual_sql"]
            if verify_sql_preflight(db,statement).ok:
                try:
                    new=execute_readonly(db,statement,max_rows=50_000)
                    if not new.truncated:
                        new_ex=set(new.rows)==set(gold.rows)
                        new_bag=Counter(new.rows)==Counter(gold.rows)
                        changed=True
                except SQLExecutionError:
                    pass
        results.append({
            "group":row["group"],"case_id":row["case_id"],
            "db_id":row["database_id"],"eligible":row["eligible"],
            "revised":changed,
            "probes":row["probes"],"original_ex":original_ex,"new_ex":new_ex,
            "original_bag":original_bag,"new_bag":new_bag,
        })
    stats=summarize(results)
    return {
        "protocol":"qwen9b-frozen-casefold-transfer40-v1",
        "model_digest":MODEL_DIGEST,"source_sha256":SOURCE_SHA,
        "candidate_set_sha256_pre_gold":revision_sha,
        "no_new_model_inference":True,"no_sql_training":True,
        "total":stats,
        "cases":results,
        "claim_boundary":"40 case IDs disjoint from historic frozen100 but public BIRD DEV and two earlier inspected exploratory cohorts; not DB-disjoint or hidden benchmark, no promotion without new-source holdout.",
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--schema",type=Path,nargs="+",required=True)
    ap.add_argument("--physical",type=Path,nargs="+",required=True)
    ap.add_argument("--frozen",type=Path,default=Path("runs/bird-paired-dev-v5/paired.json"))
    ap.add_argument("--source",type=Path,required=True)
    ap.add_argument("--database-root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():
        raise FileExistsError("refuse overwrite")
    sch=[json.loads(p.read_text(encoding="utf-8")) for p in args.schema]
    phy=[json.loads(p.read_text(encoding="utf-8")) for p in args.physical]
    out=transfer(sch,phy,args.frozen,args.source,args.database_root)
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
    print("QWEN9B_CASEFOLD_TRANSFER40",json.dumps(out["total"],sort_keys=True),flush=True)


if __name__=="__main__":
    main()
