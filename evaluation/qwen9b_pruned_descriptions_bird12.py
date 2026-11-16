"""Preregister original Qwen3.5-9B vs question-relevant description pruning.

The original physical table, column and FK schema bytes are EXACTLY preserved.
Gold BIRD SQL is not read until both Qwen9B candidates have been generated.
Excludes original100 and both public BIRD12 intervention cohorts (124 IDs).
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic

from evaluation.qwen_schema_grounding_pair import (
    MODEL, MODEL_DIGEST, SOURCE_SHA, PinnedQwen, checked_source, case_id, legacy_tokens,
)
from evaluation.qwen9b_description_noise_gate import prune_descriptions
from evaluation.qwen9b_full_schema_bird12 import evaluated_rows
from evaluation.run_bird import _ollama_model_digest, _resolve_database
from evaluation.schema_context import select_schema_context
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.inference import run_guarded

PROTOCOL="original-qwen9b-relevance-prune-descriptions-bird12-v1"
SEED="original-qwen9b-goldblind-description-noise-reduction-20261116"
N,SHARDS,PER_SHARD=12,4,3
MAX_CONTEXT_CHARS=12000
MIN_AFFECTED_DBS=6
MIN_SAVED_CHARS=250


def contexts(catalog,root,dbid,question):
    original,original_meta=select_schema_context(
        catalog,root,dbid,question,tokenizer=legacy_tokens)
    pruned,pruned_meta=prune_descriptions(original,question)
    assert original.split("\nColumn descriptions:")[0]==pruned.split("\nColumn descriptions:")[0]
    assert len(pruned)<=len(original)
    return (original,original_meta),(pruned,pruned_meta)


def exclusion_ids(original_reference,full_schema_cohort,spider_fewshot_cohort):
    ledger=json.loads(original_reference.read_text())
    if (ledger.get("status")!="complete" or ledger.get("sample_size")!=100
        or len(ledger.get("cases",[]))!=100):
        raise ValueError("original BIRD100 frozen identities missing")
    groups=[[str(x["case_id"]) for x in ledger["cases"]]]
    for path,expected_protocol in [
        (full_schema_cohort,"original-qwen9b-full-physical-schema-vs-lexical-bird12-v1"),
        (spider_fewshot_cohort,"same-qwen9b-spider-train-structure-two-examples-bird12-v1"),
    ]:
        j=json.loads(path.read_text())
        if j.get("protocol")!=expected_protocol or len(j.get("cases",[]))!=12:
            raise ValueError("prior BIRD12 exposure artifact mismatched")
        groups.append([str(x["case_id"]) for x in j["cases"]])
    if [len(g) for g in groups]!=[100,12,12] or len(set(x for g in groups for x in g))!=124:
        raise ValueError("previous exposures not precisely 124 independent BIRD identities")
    return {x for group in groups for x in group}


def preregister(source,root,reference,full_schema_cohort,spider_fewshot_cohort):
    source_rows=checked_source(source)
    excluded=exclusion_ids(reference,full_schema_cohort,spider_fewshot_cohort)
    candidates=sorted(
        ((case_id(x,i),x) for i,x in enumerate(source_rows) if case_id(x,i) not in excluded),
        key=lambda item:(sha256((SEED+":"+item[0]).encode()).hexdigest(),item[0]),
    )
    db_cache={}
    ranked=[]
    count=Counter()
    eligible=Counter()
    for cid,row in candidates:
        dbid=str(row["db_id"])
        if dbid not in db_cache:
            try:
                db_cache[dbid]=DatabaseCatalog.from_sqlite(_resolve_database(root,dbid))
            except (FileNotFoundError,OSError):
                continue
        (original,original_meta),(pruned,pruned_meta)=contexts(
            db_cache[dbid],root,dbid,str(row["question"]))
        saved=len(original)-len(pruned)
        if saved<MIN_SAVED_CHARS or len(original)>MAX_CONTEXT_CHARS:
            continue
        eligible[dbid]+=1
        if count[dbid]>=2:
            continue
        count[dbid]+=1
        ranked.append({
            "case_id":cid,"database_id":dbid,"question":str(row["question"]),
            "evidence":str(row.get("evidence") or ""),
            "original_schema_sha256":original_meta["schema_context_sha256"],
            "pruned_schema_sha256":sha256(pruned.encode()).hexdigest(),
            "original_context_chars":len(original),"pruned_context_chars":len(pruned),
            "dropped_description_rows":
                pruned_meta["original_descriptions"]-pruned_meta["retained_descriptions"],
            "original_selected_tables":original_meta["selected_tables"],
        })
        if len(ranked)==N:
            break
    if len(ranked)!=N or len({r["database_id"] for r in ranked})<MIN_AFFECTED_DBS:
        raise ValueError("Source-only prospective gate lacks 12 affected cases across >=6 databases")
    identities=[(r["case_id"],r["database_id"]) for r in ranked]
    return {
        "protocol":PROTOCOL,"source_sha256":SOURCE_SHA,
        "model":MODEL,"model_digest":MODEL_DIGEST,
        "previous_public_dev_exposures_excluded":124,
        "no_gold_used_for_selection":True,"no_model_calls_at_freeze":True,
        "selection_seed":SEED,"count":N,
        "affected_dbs":len({r["database_id"] for r in ranked}),
        "cohort_sha256":sha256(json.dumps(identities,separators=(",",":")).encode()).hexdigest(),
        "no_SQL_training":True,"same_original_Qwen9B":True,
        "potential_pretraining_contamination_unmeasured":True,
        "positive_screen_requirement":"net rescues>=3 on >=3 DB, zero regressions, no bag EX decrease",
        "cases":ranked,
    }


def generate_pair(source,root,manifest,shard):
    if shard not in range(SHARDS):raise ValueError("invalid shard")
    rows=checked_source(source)
    by_id={case_id(x,i):x for i,x in enumerate(rows)}
    frozen=json.loads(manifest.read_text())
    if (frozen.get("protocol")!=PROTOCOL or frozen.get("source_sha256")!=SOURCE_SHA
        or frozen.get("model_digest")!=MODEL_DIGEST or frozen.get("count")!=N
        or frozen.get("no_gold_used_for_selection") is not True
        or frozen.get("previous_public_dev_exposures_excluded")!=124
        or len(frozen.get("cases",[]))!=N):
        raise ValueError("Qwen9B frozen prospective experiment source changed")
    if _ollama_model_digest("http://127.0.0.1:11434",MODEL)!=MODEL_DIGEST:
        raise ValueError("original Qwen9B model digest drifted")
    batch=frozen["cases"][shard*PER_SHARD:(shard+1)*PER_SHARD]
    model=PinnedQwen()
    staged=[]
    cache={}
    for item in batch:
        cid,dbid=item["case_id"],item["database_id"]
        row=by_id[cid]
        if str(row["question"])!=item["question"] or str(row["db_id"])!=dbid:
            raise ValueError("frozen case identity drift")
        if dbid not in cache:
            database=_resolve_database(root,dbid)
            cache[dbid]=(database,DatabaseCatalog.from_sqlite(database))
        database,catalog=cache[dbid]
        (original,o),(pruned,p)=contexts(catalog,root,dbid,item["question"])
        if (o["schema_context_sha256"]!=item["original_schema_sha256"]
            or sha256(pruned.encode()).hexdigest()!=item["pruned_schema_sha256"]
            or len(original)!=item["original_context_chars"]
            or len(pruned)!=item["pruned_context_chars"]
            or o["selected_tables"]!=item["original_selected_tables"]):
            raise ValueError("source-schema bytes changed from gold-blind freeze")
        record={"case_id":cid,"database_id":dbid,
                "description_char_reduction":len(original)-len(pruned),"arms":{}}
        for arm,schema in (("original",original),("pruned_descriptions",pruned)):
            start=monotonic()
            result=run_guarded(database=database,provider=model,
                question=item["question"],schema_context=schema,
                evidence=item["evidence"].strip() or None,max_repairs=2,max_rows=50000)
            valid=result.status=="ok" and not result.truncated
            record["arms"][arm]={
                "sql":result.candidate.sql if result.candidate else "",
                "status":result.status,"execution_success":bool(valid),
                "repair_count":result.candidate.attempt if result.candidate else None,
                "seconds":round(monotonic()-start,2),
                "_result":list(result.rows) if valid else None,
            }
        staged.append(record)
        print("QWEN9B_DESCRIPTION_PRUNE_GENERATED",shard,cid,flush=True)
    # Gold is accessed ONLY after ALL original and treatment answers for this
    # preregistered shard have been frozen in memory.
    for record in staged:
        row=by_id[record["case_id"]]
        database,_=cache[record["database_id"]]
        ok,why=evaluated_rows(database,str(row["SQL"]),record["arms"])
        record["reference_scorable"]=ok
        record["reference_error"]=why
    return {
        "protocol":PROTOCOL,"shard":shard,
        "model_digest":MODEL_DIGEST,"cohort_sha256":frozen["cohort_sha256"],
        "cases":staged,
    }


def aggregate(shards):
    if len(shards)!=SHARDS or sorted(x["shard"] for x in shards)!=list(range(SHARDS)):
        raise ValueError("need four independent Qwen9B paired shards")
    checksum=shards[0]["cohort_sha256"]
    if any(x["cohort_sha256"]!=checksum or x["protocol"]!=PROTOCOL
            or x["model_digest"]!=MODEL_DIGEST or len(x["cases"])!=PER_SHARD for x in shards):
        raise ValueError("shard cohort/model integrity failed")
    rows=[record for shard in shards for record in shard["cases"]]
    if len({r["case_id"] for r in rows})!=N:raise ValueError("duplicate case in paired Qwen output")
    scored=[r for r in rows if r["reference_scorable"]]
    total=lambda arm,field:sum(bool(r["arms"][arm][field]) for r in scored)
    rescues=[r["case_id"] for r in scored if not r["arms"]["original"]["strict_ex"]
             and r["arms"]["pruned_descriptions"]["strict_ex"]]
    regressions=[r["case_id"] for r in scored if r["arms"]["original"]["strict_ex"]
                 and not r["arms"]["pruned_descriptions"]["strict_ex"]]
    positive=bool(
        len(scored)>=11 and len(rescues)-len(regressions)>=3
        and len({r["database_id"] for r in scored if r["case_id"] in rescues})>=3
        and not regressions
        and total("pruned_descriptions","strict_bag")>=total("original","strict_bag")
    )
    return {
        "protocol":PROTOCOL,"model":MODEL,"model_digest":MODEL_DIGEST,
        "cohort_sha256":checksum,"n":N,"scorable":len(scored),
        "original_set_EX":total("original","strict_ex"),
        "pruned_set_EX":total("pruned_descriptions","strict_ex"),
        "original_bag_EX":total("original","strict_bag"),
        "pruned_bag_EX":total("pruned_descriptions","strict_bag"),
        "rescues":rescues,"regressions":regressions,
        "net_rescues":len(rescues)-len(regressions),
        "distinct_rescue_dbs":len({r["database_id"] for r in scored if r["case_id"] in rescues}),
        "screen_positive_only":positive,"promoted":False,
        "no_model_swap":True,"no_SQL_training":True,
        "BIRD100_Guarded_original_still":39,
        "BIRD_DEV_public_not_hidden":True,
        "cases":rows,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path)
    p.add_argument("--root",type=Path)
    p.add_argument("--reference",type=Path,default=Path("runs/bird-paired-dev-v5/paired.json"))
    p.add_argument("--full-schema-prior",type=Path)
    p.add_argument("--spider-prior",type=Path)
    p.add_argument("--manifest",type=Path)
    p.add_argument("--shard",type=int)
    p.add_argument("--inputs",nargs="+",type=Path)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError("refuse to overwrite result")
    if args.inputs:
        result=aggregate([json.loads(x.read_text()) for x in args.inputs])
        print("QWEN9B_DESCRIPTION_PRUNE_STRICT_EX_FINAL",
              json.dumps({k:v for k,v in result.items() if k!="cases"},sort_keys=True),flush=True)
    elif args.shard is not None:
        if not args.source or not args.root or not args.manifest:
            p.error("shard requires source, root and frozen manifest")
        result=generate_pair(args.source,args.root,args.manifest,args.shard)
    else:
        if not all((args.source,args.root,args.full_schema_prior,args.spider_prior)):
            p.error("freeze requires source,root and both prior exposed cohorts")
        result=preregister(args.source,args.root,args.reference,args.full_schema_prior,args.spider_prior)
        print("QWEN9B_DESCRIPTION_PRUNE_PROSPECTIVE_FROZEN",
              json.dumps({k:v for k,v in result.items() if k!="cases"},sort_keys=True),flush=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False)+"\n")


if __name__=="__main__":
    main()
