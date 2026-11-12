"""Confirm pre-frozen A5 plan-first prompt on a NEW relational topology.

No alteration to the original A5 prompt or planner validation. The prior
positive 48-case result is NOT used for selection or scoring. Models see
question + schema, never gold SQL, before *both* arms are frozen.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from experiments.sql_roles_holdout_v2 import make_cases,build_holdout_schema
from experiments.counterfactual_sql_plan_v1 import prompt,parse_plan
from experiments.spider_grounding_shift import FixedSeedOllamaProvider,MODEL
from experiments.xiyan_dgsl_frozen100 import generate_xiyan
from experiments.audit_spider_exec_strict import score_readonly_pair


def run_shard(root:Path,domain:int,model:str=MODEL)->dict:
    if domain not in range(4):
        raise ValueError("exactly four holdout domains")
    db=root/f"holdout-domain-{domain}.sqlite"
    cases=make_cases(db,domain)
    schema=build_holdout_schema(db,domain)
    provider=FixedSeedOllamaProvider(model=model,timeout_s=900)
    frozen=[]
    for pos,item in enumerate(cases,1):
        started=perf_counter()
        base=generate_xiyan(question=item["question"],schema_context=schema,
                            evidence=None,provider=provider)
        raw=provider.complete_text(system="Only JSON plans, no SQL.",
                                  user=prompt(item["question"],schema))
        plan=parse_plan(raw)
        if plan is None:
            proposed=base
            reason="invalid_plan_original_sql_kept"
        else:
            enhanced=(
                schema+"\nProposed question-semantic plan (NOT a fact, "
                "verify every assumption against the user's question):\n"
                +json.dumps(plan,sort_keys=True,ensure_ascii=False)
            )
            proposed=generate_xiyan(question=item["question"],
                                     schema_context=enhanced,
                                     evidence=None,provider=provider)
            reason="plan_accepted_new_sql_generated"
        frozen.append({
            "case_id":item["case_id"],
            "question":item["question"],
            "database_id":str(domain),
            "sql":{"a0":base,"a5":proposed},
            "plan_valid":plan is not None,
            "reason":reason,
            "elapsed_s":round(perf_counter()-started,2),
            "gold_used_in_generation":False,
        })
        print(f"A5_HOLDOUT_FROZEN {domain} {pos}/{len(cases)}",flush=True)
    # Gold SQL is read only here, after every model call has completed.
    refs={x["case_id"]:x["reference_sql"] for x in cases}
    for item in frozen:
        item["strict"]={
            k:score_readonly_pair(db,sql,refs[item["case_id"]])
            for k,sql in item["sql"].items()
        }
    return {
        "schema":"counterfactual-sql-plan-holdout-v2-shard",
        "domain":domain,"model":model,"count":len(cases),
        "uses_prior_48_gold":False,"gold_used_for_generation":False,
        "cases":frozen,
    }


def aggregate(shards:list[dict])->dict:
    if len(shards)!=4 or sorted(x.get("domain") for x in shards)!=[0,1,2,3]:
        raise ValueError("four disjoint holdout shards required")
    if any(x.get("schema")!="counterfactual-sql-plan-holdout-v2-shard"
           or x.get("count")!=12
           or x.get("uses_prior_48_gold") is not False
           or x.get("gold_used_for_generation") is not False for x in shards):
        raise ValueError("holdout integrity failed")
    if len({x["model"] for x in shards})!=1:
        raise ValueError("model differs across shards")
    cases=[c for x in shards for c in x["cases"]]
    if len(cases)!=48 or len({c["case_id"] for c in cases})!=48:
        raise ValueError("expected 48 unique questions across four schemas")
    a0=sum(bool(x["strict"]["a0"]["bag_equal"]) for x in cases)
    a5=sum(bool(x["strict"]["a5"]["bag_equal"]) for x in cases)
    wins=sum(not x["strict"]["a0"]["bag_equal"]
             and x["strict"]["a5"]["bag_equal"] for x in cases)
    losses=sum(x["strict"]["a0"]["bag_equal"]
               and not x["strict"]["a5"]["bag_equal"] for x in cases)
    if a5-a0!=wins-losses:
        raise ValueError("corrupt paired gain")
    return {
        "schema":"counterfactual-sql-plan-confirmation48-v1",
        "model":shards[0]["model"],
        "sample_size":48,"schema_domains":4,
        "a0_bag_correct":a0,"a5_bag_correct":a5,
        "wrong_to_correct":wins,"correct_to_wrong":losses,"net_gain":wins-losses,
        "valid_plans":sum(c["plan_valid"] for c in cases),
        "changed_candidates":sum(c["sql"]["a0"].strip()!=c["sql"]["a5"].strip() for c in cases),
        "caveats":[
            "Second synthetic topology is newly authored after viewing preliminary A5 gains; selection bias remains",
            "No model finetuning or test-set hidden evaluation",
            "A0/A5 inference remains correlated, all runs share the same QwenCoder model",
            "Do not treat these small synthetic suites as proof of BIRD semantic improvement",
        ],
        "cases":sorted(cases,key=lambda c:c["case_id"]),
    }


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path)
    p.add_argument("--domain",type=int)
    p.add_argument("--model",default=MODEL)
    p.add_argument("--inputs",nargs="+",type=Path)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError("never overwrite frozen experiment")
    if args.inputs:
        report=aggregate([json.loads(p.read_text(encoding="utf-8")) for p in args.inputs])
        print("A5_INDEPENDENT_HOLDOUT_FINAL",json.dumps(
            {k:v for k,v in report.items() if k!="cases"},sort_keys=True),flush=True)
    else:
        if args.root is None or args.domain is None:
            p.error("--root and --domain are required")
        report=run_shard(args.root,args.domain,model=args.model)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,sort_keys=True,indent=2,ensure_ascii=False)+"\n")


if __name__=="__main__":
    main()
