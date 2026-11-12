"""Read-only Spider A0/A4 post-hoc scoring robustness audit.

Replays already-frozen prediction candidates on the exact pinned Spider DEV
databases. No inference or tuning. Separates the original set-of-tuples proxy
from multiset-aware row equivalence, order sensitivity, and empty/NULL pitfalls.

Not the official Spider Test-Suite Accuracy. This is a stricter independent
scorer for the already-selected 100 questions, not a new holdout.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from sqlglot import parse_one, exp

from experiments.xiyan_dgsl_frozen100 import resolve_database
from semantic_sql.execution import execute_readonly,SQLExecutionError


def _sql_ast_normal(sql:str)->str | None:
    try:
        tree=parse_one(sql,read="sqlite")
        if tree is None or not isinstance(tree,(exp.Select,exp.Union,exp.Except,exp.Intersect)):
            return None
        return tree.sql(dialect="sqlite", normalize=True)
    except Exception:
        return None


def _has_order(sql:str)->bool:
    try:
        tree=parse_one(sql,read="sqlite")
        return tree is not None and tree.find(exp.Order) is not None
    except Exception:
        return False


def score_readonly_pair(database:Path,predicted_sql:str,gold_sql:str,
                        *,max_rows:int=50_000)->dict:
    if max_rows<1:
        raise ValueError("max_rows must be positive")
    try:
        pred=execute_readonly(database,predicted_sql,max_rows=max_rows)
        gold=execute_readonly(database,gold_sql,max_rows=max_rows)
    except (SQLExecutionError,ValueError,TypeError) as exc:
        return {"execution_success":False,"failure":type(exc).__name__,
                "set_equal":False,"bag_equal":False,"ordered_equal":False,
                "exact_ast_equal":False,"both_empty":False,
                "gold_has_order":_has_order(gold_sql)}
    if pred.truncated or gold.truncated:
        return {"execution_success":True,"truncated":True,
                "set_equal":False,"bag_equal":False,"ordered_equal":False,
                "exact_ast_equal":False,"both_empty":False,
                "gold_has_order":_has_order(gold_sql)}
    pr=list(pred.rows)
    gl=list(gold.rows)
    pred_ast=_sql_ast_normal(predicted_sql)
    gold_ast=_sql_ast_normal(gold_sql)
    return {
        "execution_success":True,"truncated":False,
        "set_equal":set(pr)==set(gl),
        "bag_equal":Counter(pr)==Counter(gl),
        "ordered_equal":pr==gl,
        "exact_ast_equal":pred_ast is not None and gold_ast is not None and pred_ast==gold_ast,
        "both_empty":len(pr)==0 and len(gl)==0,
        "gold_has_order":_has_order(gold_sql),
        "pred_rows":len(pr),"gold_rows":len(gl),
        "columns_equal":pred.columns==gold.columns,
    }


def audit(frozen:dict, raw:list[dict],db_root:Path)->dict:
    if frozen.get("schema")!="spider-source-shift-a0-a4-aggregate-v1":
        raise ValueError("only frozen Spider source-shift paired100 allowed")
    cases=frozen.get("cases")
    if not isinstance(cases,list) or len(cases)!=100 or len({c["case_id"] for c in cases})!=100:
        raise ValueError("need 100 unique paired candidates")
    results=[]
    for case in cases:
        index=int(case["row_index"])
        if not 0<=index<len(raw):
            raise ValueError("case index out of bounds")
        source=raw[index]
        if source["db_id"]!=case["database_id"] or source["question"]!=case["question"]:
            raise ValueError("pinned Spider question identity changed")
        database=resolve_database(db_root,case["database_id"])
        gold=source["query"]
        arm={}
        for name in ("a0","a4"):
            outcome=score_readonly_pair(database,case["sql"][name],gold)
            observed=case["scores"][name]["frozen_execution_match"]
            if bool(outcome["set_equal"])!=bool(observed):
                raise ValueError(f"immutable scoring drift at {case['case_id']} / {name}")
            arm[name]=outcome
        results.append({
            "case_id":case["case_id"],
            "database_id":case["database_id"],
            "original_scores":{n:bool(case["scores"][n]["frozen_execution_match"]) for n in ("a0","a4")},
            "arms":arm,
            "sql_changed":case["sql"]["a0"].strip()!=case["sql"]["a4"].strip(),
        })
    metrics={}
    for name in ("a0","a4"):
        metrics[name]={
            "set_correct":sum(r["arms"][name]["set_equal"] for r in results),
            "bag_correct":sum(r["arms"][name]["bag_equal"] for r in results),
            "exact_sql_ast_equal":sum(r["arms"][name]["exact_ast_equal"] for r in results),
            "both_empty":sum(r["arms"][name]["both_empty"] for r in results),
            "bag_correct_and_order_sensitive":sum(
                r["arms"][name]["bag_equal"] and
                (not r["arms"][name]["gold_has_order"] or r["arms"][name]["ordered_equal"])
                for r in results
            ),
            "set_false_positive_vs_bag":sum(
                r["arms"][name]["set_equal"] and not r["arms"][name]["bag_equal"]
                for r in results
            ),
        }
    transitions={}
    for key in ("set_equal","bag_equal","exact_ast_equal"):
        win=sum(not r["arms"]["a0"][key] and r["arms"]["a4"][key] for r in results)
        loss=sum(r["arms"]["a0"][key] and not r["arms"]["a4"][key] for r in results)
        transitions[key]={"wrong_to_correct":win,"correct_to_wrong":loss,"net":win-loss}
    if metrics["a0"]["set_correct"]!=frozen["a0_correct"] or metrics["a4"]["set_correct"]!=frozen["a4_correct"]:
        raise ValueError("original frozen scores did not reproduce")
    return {
        "schema":"spider-posthoc-frozen-paired-bag-robustness-v1",
        "source_run_id":43259940204,
        "cohort_sha256":frozen["cohort_sha256"],"sample_size":100,
        "changed_sql_count":sum(r["sql_changed"] for r in results),
        "metrics":metrics,"paired_transitions":transitions,
        "limitations":[
            "Only the already-explored 100 public Spider DEV questions; no independent holdout",
            "Bag equality repairs duplicate-collapsing bias but cannot prove semantic equivalence",
            "Order sensitivity is reported as supplemental, not official Spider test-suite accuracy",
            "Spider dev questions may overlap model pretraining or fine-tuning data; not verified",
        ],
        "cases":results,
    }


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frozen",type=Path,required=True)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--database-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError("never overwrite posthoc metric evidence")
    frozen=json.loads(args.frozen.read_text(encoding="utf-8"))
    raw=json.loads(args.source.read_text(encoding="utf-8"))
    report=audit(frozen,raw,args.database_root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False)+"\n",encoding="utf-8")
    print("SPIDER_STRICT_AUDIT",json.dumps({k:v for k,v in report.items() if k!="cases"},sort_keys=True,ensure_ascii=False))


if __name__=="__main__":
    main()
