"""Paired BIRD semantic-correctness scorecard; never compare different cohorts.

Works on frozen per-case post-gold physical SET/BAG results. Missing real tokens,
money or full E2E timing are UNKNOWN, not zero. No model, publisher gold SQL,
protected database or paid resource is accessed by this module.
"""
from __future__ import annotations

from collections import defaultdict
from math import comb
from statistics import median
from typing import Any


def _exact_p(rescues: int, regressions: int) -> float | None:
    """Two-sided exact paired sign / McNemar test, no asymptotic shortcut."""
    n=rescues+regressions
    if n==0:
        return None
    low=min(rescues,regressions)
    return min(1.0,2*sum(comb(n,k) for k in range(low+1))/(2**n))


def _sum_known(rows: list[dict], key: str) -> int | float | None:
    values=[row.get(key) for row in rows]
    if any(type(v) not in (int,float) or v<0 for v in values):
        return None
    return sum(values)


def _percentile(values: list[float], p: float) -> float | None:
    if not values:return None
    sorted_values=sorted(values)
    index=max(0,__import__("math").ceil(len(values)*p)-1)
    return float(sorted_values[index])


def _paired_outcomes(rows: list[dict], *, control_key: str, treatment_key: str) -> dict:
    c=sum(int(row[control_key]) for row in rows)
    t=sum(int(row[treatment_key]) for row in rows)
    rescue=sum(int(not row[control_key] and row[treatment_key]) for row in rows)
    regress=sum(int(row[control_key] and not row[treatment_key]) for row in rows)
    return {"control_correct":c,"treatment_correct":t,
            "net_correct":t-c,"rescue_n":rescue,"regression_n":regress,
            "mcnemar_exact_p":_exact_p(rescue,regress)}


def paired_bird_scorecard(cases: list[dict[str,Any]]) -> dict:
    if not cases:
        raise ValueError("no frozen paired cases")
    keys=set();good=[];per_db=defaultdict(list)
    for row in cases:
        if not isinstance(row,dict):
            raise ValueError("invalid per-case score")
        key=(row.get("db_id"),row.get("source_index"))
        if not isinstance(key[0],str) or not key[0] or type(key[1]) is not int or key in keys:
            raise ValueError("invalid or duplicate original BIRD identity")
        keys.add(key)
        if type(row.get("scorable")) is not bool:
            raise ValueError("scorable must be a ground-truth source-verified boolean")
        if row["scorable"]:
            for k in ("control_set","treatment_set","control_bag","treatment_bag"):
                if type(row.get(k)) is not bool:
                    raise ValueError("missing strict publisher SET/BAG outcome: "+k)
            good.append(row)
            per_db[key[0]].append(row)
    if not good:
        raise ValueError("no publisher-scorable original BIRD questions")

    latency=[row["model_wall_s_shared"] for row in cases
             if type(row.get("model_wall_s_shared")) in (int,float)
             and row["model_wall_s_shared"]>=0]
    latency_complete=len(latency)==len(cases)
    records={
        "questions_total":len(cases),
        "questions_scorable":len(good),
        "questions_unscorable":len(cases)-len(good),
        "distinct_database_families":len({row["db_id"] for row in cases}),
        "strict_set_ex":_paired_outcomes(good,control_key="control_set",treatment_key="treatment_set"),
        "strict_bag_ex":_paired_outcomes(good,control_key="control_bag",treatment_key="treatment_bag"),
        # Changed SQL is not detection precision: a different executable query
        # is NOT a certified semantic improvement absent publisher evaluation.
        "changed_sql_n":(
            sum(int(row["graph_changed"]) for row in cases)
            if all(type(row.get("graph_changed")) is bool for row in cases)
            else None
        ),
        "per_database_net_set":{
            db: _paired_outcomes(rows,control_key="control_set",treatment_key="treatment_set")["net_correct"]
            for db,rows in sorted(per_db.items())
        },
        "cost":{
            "shared_Qwen_calls":_sum_known(cases,"model_calls_shared"),
            "shared_Qwen_wall_s":_sum_known(cases,"model_wall_s_shared"),
            "incremental_graph_wall_s":_sum_known(cases,"additional_graph_wall_s"),
            "shared_Qwen_latency_p50_s":float(median(latency)) if latency_complete else None,
            "shared_Qwen_latency_p95_s":_percentile(latency,.95) if latency_complete else None,
            "prompt_tokens":_sum_known(cases,"prompt_eval_count"),
            "generated_tokens":_sum_known(cases,"eval_count"),
            # Ollama token counts do not equal money spent on a CPU Actions runner.
            "actual_USD_cost":_sum_known(cases,"actual_usd_cost"),
            "end_to_end_latency_p95_s":_percentile([
                float(row["end_to_end_wall_s"]) for row in cases
                if type(row.get("end_to_end_wall_s")) in (int,float)
                and row["end_to_end_wall_s"]>=0
            ],.95) if all(type(row.get("end_to_end_wall_s")) in (int,float)
                      and row["end_to_end_wall_s"]>=0 for row in cases) else None,
        },
        "note":"Strict SET/BAG requires same question, same database and postsealed gold; exact McNemar on discordant pairs; never substitute SQL executability. Missing observed cost/latency is null. No cross-cohort comparison.",
    }
    return records
