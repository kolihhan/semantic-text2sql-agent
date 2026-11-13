"""Diagnostic-only SQL structural mismatch audit on original frozen Qwen9B 60.

No new SQL generation, no training, no selector or auto repair. Reference
SQL is used ONLY after the existing frozen original predictions are selected,
for measurement and hypothesis generation. An AST feature mismatch is NOT
proof of the semantic error; structurally different SQL can be equivalent.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import sqlglot
from sqlglot import exp

from evaluation.bird_loader import load_bird_json
from evaluation.semantic_v2_frozen_pilot import freeze_inputs


def feature(sql: str) -> dict:
    try:
        parsed = sqlglot.parse_one(sql or "", read="sqlite")
    except (sqlglot.errors.ParseError, ValueError, TypeError):
        return {"parse_error": True}
    if not isinstance(parsed, exp.Expression):
        return {"parse_error": True}
    selects = list(parsed.find_all(exp.Select))
    joins = list(parsed.find_all(exp.Join))
    aggregate_nodes = list(parsed.find_all(exp.AggFunc))
    operators = ["eq","neq","gt","gte","lt","lte","between","in","like","is"]
    opmap = {
        "eq":exp.EQ,"neq":exp.NEQ,"gt":exp.GT,"gte":exp.GTE,
        "lt":exp.LT,"lte":exp.LTE,"between":exp.Between,
        "in":exp.In,"like":exp.Like,"is":exp.Is,
    }
    aggs = ("Count","Sum","Avg","Min","Max")
    count_by_agg={k:sum(type(node).__name__==k for node in aggregate_nodes) for k in aggs}
    # Compact AST signature: no table names/Gold SQL/answer leakage into
    # any decision; the analysis stage has no decision or repair.
    return {
        "parse_error": False,
        "select_blocks":len(selects),
        "joins":len(joins),
        "tables":len(list(parsed.find_all(exp.Table))),
        "group_columns":sum(len(node.expressions) for node in parsed.find_all(exp.Group)),
        "order_columns":sum(len(node.expressions) for node in parsed.find_all(exp.Order)),
        "limits":len(list(parsed.find_all(exp.Limit))),
        "distinct":len(list(parsed.find_all(exp.Distinct))),
        "aggs":count_by_agg,
        "where_nodes":len(list(parsed.find_all(exp.Where))),
        "predicate_ops":{name:len(list(parsed.find_all(cls))) for name,cls in opmap.items()},
    }


def compare(a: dict, b: dict) -> list[str]:
    if a["parse_error"] or b["parse_error"]:
        return ["parse_failure"]
    return [key for key in
            ("select_blocks","joins","tables","group_columns","order_columns",
             "limits","distinct","aggs","where_nodes","predicate_ops")
            if a[key] != b[key]]


def analyze(literal40: dict, aa20: dict, *, source: Path) -> dict:
    rows = freeze_inputs(literal40, aa20)
    if len(rows)!=60 or len({r["case_id"] for r in rows})!=60:
        raise ValueError("must use frozen original Qwen9B 60")
    _questions, label_rows = load_bird_json(source)
    references = {str(q.case_id):q.sql for q in label_rows}
    by_group=defaultdict(lambda: {"total":0,"correct":0,"wrong":0,
                                 "wrong_mismatch":Counter(),"correct_mismatch":Counter(),
                                 "wrong_no_mismatch":0,"wrong_parse_failure":0})
    for row in rows:
        group=row["cohort"]
        stats=by_group[group]
        stats["total"]+=1
        original=feature(row["baseline_sql"])
        gold=feature(references[row["case_id"]])
        differences=compare(original,gold)
        is_correct=bool(row["pre_scored"])
        stats["correct" if is_correct else "wrong"]+=1
        if is_correct:
            stats["correct_mismatch"].update(differences)
        else:
            stats["wrong_mismatch"].update(differences)
            stats["wrong_no_mismatch"]+=int(not differences)
            stats["wrong_parse_failure"]+=int("parse_failure" in differences)

    output={}
    totals={"total":0,"correct":0,"wrong":0,
            "wrong_no_mismatch":0,"wrong_parse_failure":0}
    overall_wrong=Counter()
    overall_correct=Counter()
    for name,stat in sorted(by_group.items()):
        output[name]={
            "total":stat["total"],"correct":stat["correct"],"wrong":stat["wrong"],
            "wrong_mismatch":dict(stat["wrong_mismatch"]),
            "correct_mismatch":dict(stat["correct_mismatch"]),
            "wrong_no_mismatch":stat["wrong_no_mismatch"],
            "wrong_parse_failure":stat["wrong_parse_failure"],
        }
        for k in totals: totals[k]+=stat[k]
        overall_wrong.update(stat["wrong_mismatch"])
        overall_correct.update(stat["correct_mismatch"])
    if totals["total"]!=60 or totals["correct"]!=33 or totals["wrong"]!=27:
        raise AssertionError("frozen Qwen baseline identity drift")
    return {
        "schema":"qwen9b-semantic-structure-diagnostic60/v1",
        "overall":{**totals,"wrong_mismatch":dict(overall_wrong),
                   "correct_mismatch":dict(overall_correct)},
        "groups":output,
        "no_generation_or_sql_selector":True,
        "promote":False,
        "caveat":"Only 60 previously inspected public BIRD DEV questions. "
                 "AST differences are structural signals, NOT proof of why "
                 "SQL is semantically wrong. No selector trained/fitted/evaluated.",
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--literal40",type=Path,required=True)
    p.add_argument("--aa20",type=Path,required=True)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError("refuse overwrite frozen diagnostic")
    report=analyze(
        json.loads(a.literal40.read_text()),
        json.loads(a.aa20.read_text()),source=a.source)
    a.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("QWEN9B_SEMANTIC_STRUCTURE_DIAGNOSTIC",json.dumps(report,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
