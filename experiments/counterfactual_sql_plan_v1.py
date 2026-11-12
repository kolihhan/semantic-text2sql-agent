"""A0 vs A5 independent *planning-first* SQL candidate experiment.

A0 and A5 see the same novel NL question and SQLite M-Schema. A5 additionally
asks the same fixed-seed XiYan model to independently spell out answer grain,
semantic roles, aggregation and filters *before* SQL; only valid bounded JSON
plans enter its second-generation prompt. No reference SQL enters any prompt,
planner acceptance check or chooser. Judge via strict bag-equality post-hoc.

A5 is an experimental generative candidate, not an automatic production repair.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from experiments.sql_roles_synthetic import create_domain,mschema
from experiments.spider_grounding_shift import FixedSeedOllamaProvider,MODEL
from experiments.xiyan_dgsl_frozen100 import generate_xiyan
from experiments.audit_spider_exec_strict import score_readonly_pair

_PLAN_KEYS=("answer_entity","relation_role","aggregation_grain","filters")


def parse_plan(s:str)->dict | None:
    text=s.strip()
    if len(text)>4000:return None
    try:
        obj=json.loads(text)
    except (TypeError,ValueError):
        return None
    if (not isinstance(obj,dict) or set(obj)!=set(_PLAN_KEYS)
        or any(not isinstance(obj[k],str) or not obj[k].strip() or
               len(obj[k])>180 for k in _PLAN_KEYS)):
        return None
    return obj


def prompt(question:str,schema:str)->str:
    return (
        "Create a short semantic query PLAN, not an SQL query. "
        "Use EXACTLY a single JSON object with four string fields: "
        "answer_entity, relation_role, aggregation_grain, filters. "
        "For any role or value not determined by the question and schema, "
        "write UNKNOWN. Distinguish who INITIATED a relation from who RECEIVED it; "
        "count entities with DISTINCT when a join can duplicate them; "
        "do not replace AND by OR; do not change explicit numeric thresholds. "
        "Never write SQL, never invent table names, and never claim to know gold SQL.\n"
        f"USER QUESTION:\n{question}\n\nSQLITE DATABASE SCHEMA:\n{schema}\n\n"
        "RETURN JSON ONLY:"
    )


def run_shard(root:Path,shard:int,model:str=MODEL)->dict:
    if shard not in range(4):raise ValueError("exactly four disjoint domains")
    db=root/f"novel_domain_{shard}.sqlite"
    cases=create_domain(db,shard)
    schema=mschema(db,shard)
    provider=FixedSeedOllamaProvider(model=model,timeout_s=900)
    proposals=[]
    for idx,case in enumerate(cases,1):
        started=perf_counter()
        base=generate_xiyan(question=case["question"],schema_context=schema,
                            evidence=None,provider=provider)
        raw_plan=provider.complete_text(system="Only JSON plans, no SQL.",
                                        user=prompt(case["question"],schema))
        plan=parse_plan(raw_plan)
        if plan is None:
            candidate=base
            reason="invalid_plan_original_sql_kept"
        else:
            grounded_context=(
                schema+"\nProposed question-semantic plan (NOT a fact, "
                "verify every assumption against the user's question):\n"
                +json.dumps(plan,sort_keys=True,ensure_ascii=False))
            candidate=generate_xiyan(question=case["question"],
                                     schema_context=grounded_context,
                                     evidence=None,provider=provider)
            reason="plan_accepted_new_sql_generated"
        proposals.append({
            "case_id":case["case_id"],"question":case["question"],
            "database_id":f"novel_domain_{shard}",
            "sql":{"a0":base,"a5":candidate},
            "plan_valid":plan is not None,
            "reason":reason,
            "plan_sha256":sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest() if plan else None,
            "elapsed_s":round(perf_counter()-started,2),
            "gold_used_in_generation":False,
        })
        print(f"A5_FROZEN_GENERATED shard={shard} {idx}/{len(cases)} plan={reason}",flush=True)
    # Critical separation: NO model call after this point. Candidate SQL frozen.
    golds={x["case_id"]:x["reference_sql"] for x in cases}
    for item in proposals:
        item["strict"]={
            key:score_readonly_pair(db,sql,golds[item["case_id"]])
            for key,sql in item["sql"].items()
        }
    return {
        "schema":"counterfactual-sql-plan-shard-v1","model":model,
        "domain":shard,"count":len(cases),"gold_used_for_generation":False,
        "cases":proposals,
    }


def aggregate(shards:list[dict])->dict:
    if len(shards)!=4 or sorted(s["domain"] for s in shards)!=[0,1,2,3]:
        raise ValueError("need exactly four disjoint synthetic schema domains")
    if (len({s["model"] for s in shards})!=1 or
        any(s["schema"]!="counterfactual-sql-plan-shard-v1" or
            s["count"]!=12 or s["gold_used_for_generation"] is not False for s in shards)):
        raise ValueError("shard model/provenance mismatch")
    cases=[c for s in shards for c in s["cases"]]
    if len(cases)!=48 or len({c["case_id"] for c in cases})!=48:
        raise ValueError("expected 48 unique, paired questions")
    base=sum(bool(c["strict"]["a0"]["bag_equal"]) for c in cases)
    proposed=sum(bool(c["strict"]["a5"]["bag_equal"]) for c in cases)
    wins=sum(not c["strict"]["a0"]["bag_equal"]
             and c["strict"]["a5"]["bag_equal"] for c in cases)
    losses=sum(c["strict"]["a0"]["bag_equal"]
               and not c["strict"]["a5"]["bag_equal"] for c in cases)
    if proposed-base!=wins-losses:raise AssertionError("paired math mismatch")
    return {
        "schema":"counterfactual-sql-plan-aggregate-v1",
        "sample_size":48,"database_count":4,
        "model":shards[0]["model"],
        "a0_bag_correct":base,"a5_bag_correct":proposed,
        "wrong_to_correct":wins,"correct_to_wrong":losses,
        "net_gain":wins-losses,
        "plan_valid_count":sum(c["plan_valid"] for c in cases),
        "candidate_sql_changed":sum(c["sql"]["a0"].strip()!=c["sql"]["a5"].strip() for c in cases),
        "mean_seconds_per_pair":round(sum(c["elapsed_s"] for c in cases)/48,2),
        "limitations":[
            "Novel but hand-authored small synthetic tasks; no official benchmark score",
            "Four DB domains share a common relational data topology; not an OOD proof",
            "Strict row-multiset equality is an imperfect semantic correctness proxy",
            "Planner model is identical to SQL generator; non-independence is intentional",
            "Never optimize prompts on this revealed 48-case result and call it a holdout",
        ],
        "cases":sorted(cases,key=lambda c:c["case_id"]),
    }


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--shard-index",type=int)
    p.add_argument("--work-root",type=Path)
    p.add_argument("--model",default=MODEL)
    p.add_argument("--inputs",nargs="+",type=Path)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("refuse overwrite of frozen evidence")
    if a.inputs:
        report=aggregate([json.loads(f.read_text(encoding="utf-8")) for f in a.inputs])
        print("COUNTERFACTUAL_A5_FINAL",json.dumps({
            k:v for k,v in report.items() if k!="cases"
        },sort_keys=True),flush=True)
    else:
        if a.shard_index is None or a.work_root is None:
            p.error("need --shard-index and --work-root")
        report=run_shard(a.work_root,a.shard_index,model=a.model)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,sort_keys=True,ensure_ascii=False,indent=2)+"\n")


if __name__=="__main__":
    main()
