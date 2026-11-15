"""Exact frozen Spider grain-first Qwen prompt transferred to BIRD DEV20.

No prompt tuning after Spider18 (9/18 -> 11/18, 3 rescues, 1 regression).
Only the INITIAL Qwen SQL generation system prefix receives the exact same
additional grain text; repairs use original unchanged Qwen9B Guarded.

This cohort is question-disjoint from original BIRD100 and prior named
BIRD DEV cohorts, but all public databases may overlap earlier experiments
and model pretraining. Score strict set + bag physical SQLite EX; no SQL
training; no inferred gain on the original BIRD100 until separately measured.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from hashlib import sha256,file_digest
import json
from pathlib import Path
from time import monotonic

from evaluation.bird_loader import load_bird_json
from evaluation.qwen_grain_direct_spider import (
    _GRAIN_HINT, _generate_and_run, is_group_question,
    DIGEST,MODEL,GrainDirectProvider
)
from evaluation.run_bird import _resolve_database,_ollama_model_digest
from evaluation.schema_context import select_schema_context
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.execution import execute_readonly,SQLExecutionError

PROTOCOL="qwen9b-original-spider-grainhint-untouched-bird-dev20-v1"
BIRD_SOURCE_SHA="ffd8018378ddb1a8794753e0a31cfc81862ff7318a5184c22f3dc4ce03a03feb"
PRIOR_BASELINE_SHA="0310b7ca2dfb6247234e2073f93e7a75106ee41ba55231f75d06d2ef68efe2ec"
N=20
DBS=10
SHARDS=4
SEED="bird-source-shift-exact-original-spider-grain-hint-20261115-v1"
# Public prior BIRD16/24 task identities. Completely independent
# verification of case novelty beyond explicit exclusions is not claimed.
PRIOR16={
    "632","706","1477","629","635","1140","1097","1112",
    "1156","195","721","1422","391","155","821","851"
}
PRIOR24={
    "1465","576","1348","539","156","525","1469","1413",
    "113","628","499","1324","1000","981","307","769",
    "244","1174","489","141","1186","748","529","954"
}


def _qid(row:dict,i:int)->str:
    return str(row.get("question_id") or row.get("id") or i)


def freeze(source:Path,original:Path)->dict:
    if sha256(source.read_bytes()).hexdigest()!=BIRD_SOURCE_SHA:
        raise ValueError("publisher BIRD DEV source checksum drift")
    original_json=json.loads(original.read_text(encoding="utf-8"))
    if (original_json.get("status")!="complete"
        or original_json.get("sample_size")!=100
        or len(original_json.get("cases",[]))!=100):
        raise ValueError("original historical 100 test identities unavailable")
    original_ids={str(x["case_id"]) for x in original_json["cases"]}
    if len(original_ids)!=100:
        raise ValueError("historical source identities repeated")
    exclude=original_ids|PRIOR16|PRIOR24
    data=json.loads(source.read_text(encoding="utf-8"))
    if len(data)!=1534:
        raise ValueError("unexpected original BIRD DEV size")
    groups=defaultdict(dict)
    for index,row in enumerate(data):
        ident=_qid(row,index)
        question=" ".join(str(row.get("question") or "").split())
        dbid=str(row.get("db_id") or "")
        if ident in exclude or not dbid or not is_group_question(question):
            continue
        groups[dbid].setdefault(question.casefold(),{
            "case_id":ident,"database_id":dbid,
            "question":question,"evidence":str(row.get("evidence") or "")
        })
    ordering=lambda text:sha256((SEED+":"+text).encode()).hexdigest()
    bydb=sorted(
        (db for db,questions in groups.items() if len(questions)>=2),
        key=lambda db:ordering("db:"+db)
    )
    selected=[]
    for db in bydb[:DBS]:
        selected.extend(sorted(groups[db].values(),key=lambda r:ordering(r["case_id"]))[:2])
    if len(selected)!=N or len({r["database_id"] for r in selected})!=DBS:
        raise ValueError(
            f"BIRD target source has only {len(selected)} questions and "
            f"{len({r['database_id'] for r in selected})} DB; do not relax frozen screen"
        )
    if set(r["case_id"] for r in selected)&exclude:
        raise ValueError("previously exposed official BIRD question included")
    sha=sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest()
    return {
        "protocol":PROTOCOL,
        "original_model":MODEL,"original_model_digest":DIGEST,
        "bird_source_sha":BIRD_SOURCE_SHA,
        "original_case_ids_source":PRIOR_BASELINE_SHA,
        "spider_grain_hint_sha256":sha256(_GRAIN_HINT.encode()).hexdigest(),
        "selected_cohort_sha256":sha,
        "question_disjoint_original_BIRD100":True,
        "question_disjoint_named_previous_BIRD16_24":True,
        "source_gold_used_during_freeze":False,
        "physical_database_independence_from_prior_BIRD":False,
        "possible_public_pretraining_overlap":True,
        "n":N,"databases":DBS,"shards":SHARDS,
        "cases":selected,
    }


def generate_shard(source:Path, original:Path, frozen:Path, dbroot:Path, shard:int)->dict:
    if not 0<=shard<SHARDS:
        raise ValueError("invalid shard")
    chosen=freeze(source,original)
    manifest=json.loads(frozen.read_text())
    if manifest!=chosen:
        raise ValueError("post-freeze original Spider prompt or source cohort modified")
    if _ollama_model_digest("http://127.0.0.1:11434",MODEL)!=DIGEST:
        raise ValueError("not the pinned original Qwen3.5-9B model")
    allrows=json.loads(source.read_text(encoding="utf-8"))
    by_id={_qid(r,i):r for i,r in enumerate(allrows)}
    batch=manifest["cases"][5*shard:5*(shard+1)]
    if len(batch)!=5:
        raise ValueError("BIRD transfer shard incomplete")
    cache={}
    records=[]
    for item in batch:
        cid=item["case_id"]
        row=by_id[cid]
        dbid=item["database_id"]
        if row["db_id"]!=dbid or " ".join(row["question"].split())!=item["question"]:
            raise ValueError("original BIRD physical case identity drift")
        if dbid not in cache:
            db=_resolve_database(dbroot,dbid)
            cache[dbid]=(db,DatabaseCatalog.from_sqlite(db))
        db,cat=cache[dbid]
        schema,_=select_schema_context(cat,dbroot,dbid,item["question"])
        outputs={}
        for label,treatment in (("original",False),("grain_direct",True)):
            # Same unchanged provider and exact frozen Spider treatment;
            # always read answers only after both arms generated.
            result=_generate_and_run(
                db,schema,item["question"],treated=treatment,
                external_evidence=item["evidence"] or None,
            )
            outputs[label]=result
            print("QWEN_GRAIN_BIRD20_GENERATED",shard,cid,label,
                  "calls",result["model_calls"],flush=True)
        records.append({
            **item,"schema_sha256":sha256(schema.encode()).hexdigest(),
            "database_sha256":file_digest(db.open("rb"),"sha256").hexdigest(),
            "arms":outputs
        })
    # Gold SQL was not consulted for sampling, generation or gating.
    for record in records:
        raw=by_id[record["case_id"]]
        db=cache[record["database_id"]][0]
        try:
            gold=execute_readonly(db,str(raw["SQL"]),max_rows=50_000)
            valid=not gold.truncated
        except (SQLExecutionError,ValueError):
            gold,valid=None,False
        record["reference_scorable"]=valid
        for arm in ("original","grain_direct"):
            predicted=record["arms"][arm]
            rows=predicted.pop("rows")
            good=valid and predicted["status"]=="ok" and not predicted["truncated"]
            predicted["set_ex"]=bool(good and set(map(tuple,rows))==set(map(tuple,gold.rows))) if valid else None
            predicted["bag_ex"]=bool(good and Counter(map(tuple,rows))==Counter(map(tuple,gold.rows))) if valid else None
    return {
        "protocol":PROTOCOL,"source_sha":BIRD_SOURCE_SHA,
        "cohort_sha":manifest["selected_cohort_sha256"],
        "hint_sha":manifest["spider_grain_hint_sha256"],
        "digest":DIGEST,"shard":shard,"cases":records,
        "sql_training":False,"source_gold_blind_generation":True,
    }


def aggregate(shards:list[dict])->dict:
    if (len(shards)!=SHARDS
        or sorted(s["shard"] for s in shards)!=list(range(SHARDS))
        or any(s.get("protocol")!=PROTOCOL
               or s.get("source_sha")!=BIRD_SOURCE_SHA
               or s.get("digest")!=DIGEST
               or len(s.get("cases",[]))!=N//SHARDS
               or not s.get("source_gold_blind_generation")
               for s in shards)):
        raise ValueError("invalid incomplete BIRD transfer experiment")
    if len({s["cohort_sha"] for s in shards})!=1 or len({s["hint_sha"] for s in shards})!=1:
        raise ValueError("model cohort or treatment prompt drift")
    rows=[r for s in sorted(shards,key=lambda s:s["shard"]) for r in s["cases"]]
    if len(rows)!=N or len({r["case_id"] for r in rows})!=N:
        raise ValueError("duplicate/missing independent BIRD questions")
    complete=[r for r in rows if r["reference_scorable"]]
    def scores(arm):
        return {
            "set_ex":sum(r["arms"][arm]["set_ex"] for r in complete),
            "bag_ex":sum(r["arms"][arm]["bag_ex"] for r in complete),
            "executed":sum(r["arms"][arm]["status"]=="ok" and not r["arms"][arm]["truncated"] for r in rows),
            "model_calls":sum(r["arms"][arm]["model_calls"] for r in rows),
            "latency_s":round(sum(r["arms"][arm]["elapsed_s"] for r in rows),1),
        }
    rescued=[r for r in complete if not r["arms"]["original"]["set_ex"] and r["arms"]["grain_direct"]["set_ex"]]
    regressed=[r for r in complete if r["arms"]["original"]["set_ex"] and not r["arms"]["grain_direct"]["set_ex"]]
    prior,treatment=scores("original"),scores("grain_direct")
    positive=(
        len(complete)>=18
        and len(rescued)-len(regressed)>=3
        and not regressed
        and len({r["database_id"] for r in rescued})>=3
        and treatment["bag_ex"]>=prior["bag_ex"]
    )
    return {
        "protocol":PROTOCOL,"source":"BIRD DEV 20 question-disjoint from original100 and named 16/24 cohorts",
        "scorable":len(complete),"n":N,
        "database_count":len({r["database_id"] for r in rows}),
        "arms":{"original":prior,"grain_direct":treatment},
        "rescues":[r["case_id"] for r in rescued],
        "regressions":[r["case_id"] for r in regressed],
        "rescue_database_count":len({r["database_id"] for r in rescued}),
        "positive_screen_only":positive,
        "bird100_original_39_unchanged":True,
        "promotion_allowed":False,
        "training":False,
        "independence_limits":"same BIRD DEV database families; public source pretraining unknown, not hidden-test",
        "cases":rows,
    }


def main():
    p=argparse.ArgumentParser()
    for key in ("source","original","frozen","database-root","output"):
        p.add_argument("--"+key,type=Path)
    p.add_argument("--shard",type=int)
    p.add_argument("--inputs",type=Path,nargs="+")
    a=p.parse_args()
    if a.output is None or a.output.exists():
        raise FileExistsError("refuse overwrite")
    if a.inputs:
        result=aggregate([json.loads(f.read_text()) for f in a.inputs])
        print("QWEN_GRAIN_UNCHANGED_BIRD20_FINAL",json.dumps({
            key:value for key,value in result.items() if key!="cases"
        },sort_keys=True),flush=True)
    elif a.frozen:
        if any(x is None for x in (a.source,a.original,a.database_root,a.shard)):
            p.error("paired run needs source/original/frozen/database-root/shard")
        result=generate_shard(a.source,a.original,a.frozen,a.database_root,a.shard)
    else:
        if a.source is None or a.original is None:
            p.error("freeze needs source/original")
        result=freeze(a.source,a.original)
        print("QWEN_GRAIN_BIRD20_FROZEN",result["selected_cohort_sha256"],flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")


if __name__=="__main__":
    main()
