"""CPU-only matched comparison of original Guarded graph, adaptive graph, K4, and physical K4.

Requires fully sealed original BIRD TRAIN16 source and both completed, separately
scored run artifacts. Never runs new Qwen inference and never accesses gold SQL.
Individual first SQL generation latency is NOT recoverable from K4 aggregated
wall-clock and is deliberately reported as unknown, not estimated.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from experiments.bird_qwen9b_physical_answer_contract16 import (
    checked_source, DIGEST, N, N_DB, PER_DB, PROTOCOL as K4_PROTOCOL
)
from experiments.bird_langgraph_adaptive16 import PROTOCOL as GRAPH_PROTOCOL


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ids(items: list[dict]) -> dict[int, dict]:
    out = {}
    for row in items:
        i = int(row["source_index"])
        if i in out:
            raise ValueError("duplicate source index across BIRD16 shards")
        out[i] = row
    return out


def summarize(
    *, source: Path, manifest: Path,
    adaptive_generated: list[Path], adaptive_scored: list[Path],
    k4_generated: list[Path], k4_scored: list[Path],
    out: Path
) -> dict:
    questions, meta = checked_source(source, manifest)
    if any(len(items) != N_DB for items in
           (adaptive_generated, adaptive_scored, k4_generated, k4_scored)):
        raise ValueError("all four DB shards required for every sealed arm")
    source_by_id = _ids(questions)
    if len(source_by_id) != N:
        raise ValueError("frozen source must contain exactly 16 unique BIRD questions")
    tables = {}
    for family, paths, protocol, status in (
        ("adaptive_generated", adaptive_generated, GRAPH_PROTOCOL, "BIRD_LANGGRAPH_CANDIDATE_SEALED_PRE_GOLD"),
        ("adaptive_scored", adaptive_scored, GRAPH_PROTOCOL, "POST_SEAL_BIRD_STRICT_PHYSICAL_SCORE"),
        ("k4_generated", k4_generated, K4_PROTOCOL, "ORIGINAL_QWEN9B_CANDIDATES_AND_CHOICES_SEALED_PRE_GOLD"),
        ("k4_scored", k4_scored, K4_PROTOCOL, "PUBLISHER_STRICT_PHYSICAL_SET_BAG_EX_SCORED_POST_SEAL"),
    ):
        shards = {}
        for path in paths:
            d=load(path)
            sh=d.get("shard")
            if (not isinstance(sh,int) or isinstance(sh,bool) or sh not in range(N_DB)
                or sh in shards
                or d.get("protocol")!=protocol
                or d.get("status")!=status
                or d.get("model_digest")!=DIGEST
                or d.get("source_sha256")!=meta["source_sha256"]
                or d.get("db_id")!=meta["dbs"][sh]
                or d.get("n")!=PER_DB
                or len(d.get("cases",[]))!=PER_DB):
                raise ValueError(f"bad {family} shard {path}")
            shards[sh]=d
        if len(shards)!=N_DB:
            raise ValueError("missing original physical BIRD shard")
        cases = [row for sh in range(N_DB) for row in shards[sh]["cases"]]
        by_id=_ids(cases)
        if set(by_id)!=set(source_by_id):
            raise ValueError("different source cohorts; cannot compare EX")
        for i,row in by_id.items():
            source_row=source_by_id[i]
            if row["db_id"] != source_row["db_id"]:
                raise ValueError("physical DB identity mismatch")
            if family.endswith("generated") and row["question_sha256"]!=source_row["question_sha256"]:
                raise ValueError("question content identity mismatch")
        tables[family]=by_id
    if any(not tables["adaptive_scored"][i]["scorable"]
           or not tables["k4_scored"][i]["reference_scorable"] for i in source_by_id):
        raise ValueError("cannot compare censored/unscorable question as common N=16")

    sums=defaultdict(int)
    time_s=defaultdict(float)
    dbs=defaultdict(lambda: defaultdict(int))
    cases=[]
    for i in sorted(source_by_id):
        q=source_by_id[i]
        ag=tables["adaptive_generated"][i]
        asc=tables["adaptive_scored"][i]
        kg=tables["k4_generated"][i]
        ksc=tables["k4_scored"][i]
        # This *exact* SQL was pre-generated in the older sealed original Qwen
        # control K4 run; it is the same original input for both LangGraph arms.
        if ag["original_sql"]!=kg["control_sql"][0]:
            raise ValueError("LangGraph and K4 do not share the same original SQL")
        if (ag["control_calls"]!=asc["control_calls"]
            or ag["treatment_calls"]!=asc["treatment_calls"]
            or ag["control_wall_s"]!=asc["control_wall_s"]
            or ag["treatment_wall_s"]!=asc["treatment_wall_s"]):
            raise ValueError("scorer stage mutated measured inference costs")
        selections={
            "guarded":bool(asc["control_set"]),
            "adaptive":bool(asc["treatment_set"]),
            "k4":bool(ksc["control_selected_set_EX"]),
            "physical_k4":bool(ksc["treatment_selected_set_EX"]),
        }
        for k,v in selections.items():
            sums[k+"_set"]+=int(v)
            dbs[q["db_id"]][k+"_set"]+=int(v)
        for k,v in {
            "guarded":asc["control_bag"],"adaptive":asc["treatment_bag"],
            "k4":ksc["control_selected_bag_EX"],
            "physical_k4":ksc["treatment_selected_bag_EX"],
        }.items():
            sums[k+"_bag"]+=int(v)
        sums["adaptive_rescues"]+=int(selections["adaptive"] and not selections["guarded"])
        sums["adaptive_regressions"]+=int(selections["guarded"] and not selections["adaptive"])
        sums["k4_rescues_vs_guarded"]+=int(selections["k4"] and not selections["guarded"])
        sums["k4_regressions_vs_guarded"]+=int(selections["guarded"] and not selections["k4"])
        sums["physical_k4_rescues_vs_guarded"]+=int(selections["physical_k4"] and not selections["guarded"])
        sums["physical_k4_regressions_vs_guarded"]+=int(selections["guarded"] and not selections["physical_k4"])
        sums["probed_questions"]+=int(asc["adaptive_probed"])
        sums["fallback_questions"]+=int(asc["adaptive_fallback"])
        sums["guarded_extra_calls"]+=int(asc["control_calls"])
        sums["adaptive_extra_calls"]+=int(asc["treatment_calls"])
        time_s["guarded_extra_model_wall_s"]+=float(asc["control_wall_s"])
        time_s["adaptive_extra_model_wall_s"]+=float(asc["treatment_wall_s"])
        time_s["k4_total_original_generation_wall_s"]+=float(ksc["control_wall_s"])
        time_s["physical_k4_total_original_generation_wall_s"]+=float(ksc["treatment_wall_s"])
        cases.append({
            "source_index":i,"db_id":q["db_id"],
            "guarded_set":selections["guarded"],
            "adaptive_set":selections["adaptive"],
            "k4_set":selections["k4"],
            "physical_k4_set":selections["physical_k4"],
            "adaptive_probed":bool(asc["adaptive_probed"]),
            "adaptive_fallback":bool(asc["adaptive_fallback"]),
            "guarded_extra_calls":int(asc["control_calls"]),
            "adaptive_extra_calls":int(asc["treatment_calls"]),
        })
    calls={
        # Cached first Qwen candidate was actually generated in predecessor run.
        "guarded_initial_sql_generation_calls":N,
        "guarded_post_generation_calls":sums["guarded_extra_calls"],
        "guarded_total_logical_calls":N+sums["guarded_extra_calls"],
        "adaptive_initial_sql_generation_calls":N,
        "adaptive_post_generation_calls":sums["adaptive_extra_calls"],
        "adaptive_total_logical_calls":N+sums["adaptive_extra_calls"],
        "k4_total_logical_calls":4*N,
        "physical_k4_total_logical_calls":4*N,
        "incremental_adaptive_calls_vs_guarded":sums["adaptive_extra_calls"]-sums["guarded_extra_calls"],
        "model_call_cost_currency":None,
    }
    runtime={
        k:round(v,2) for k,v in time_s.items()
    }
    runtime["adaptive_minus_guarded_post_generation_model_wall_s"]=round(
        time_s["adaptive_extra_model_wall_s"]-time_s["guarded_extra_model_wall_s"],2)
    runtime["physical_k4_over_k4_model_wall_ratio"]=round(
        time_s["physical_k4_total_original_generation_wall_s"]
        / max(time_s["k4_total_original_generation_wall_s"],1e-9),3)
    runtime["guarded_incremental_model_wall_s_per_question"]=round(
        time_s["guarded_extra_model_wall_s"]/N,2)
    runtime["adaptive_incremental_model_wall_s_per_question"]=round(
        time_s["adaptive_extra_model_wall_s"]/N,2)
    runtime["initial_first_candidate_generation_wall_s"]=None
    runtime["sqlite_probe_wall_s"]=None
    runtime["end_to_end_latency_s"]=None
    runtime["reason_missing"]="Original K4 seal reports total 4-call generation wall only; adaptive seal meters additional model calls, not SQLite probe/e2e wall. Do not divide K4 wall by 4 as a measured first-call latency."
    result={
        "protocol":"BIRD-TRAIN16-MATCHED-LANGGRAPH-K4-4WAY-COST-COMPARISON-V1",
        "status":"FOUR_WAY_BIRD_STRICT_EX_AND_COST_RECONCILED",
        "sample":{"partition":"BIRD TRAIN", "n":N,"dbs":N_DB,
                  "question_and_physical_database_previously_exposed":True,
                  "source_sha256":meta["source_sha256"],
                  "pinned_original_Qwen3_5_9B_digest":DIGEST},
        "correct":{k:sums[k] for k in [
            "guarded_set","adaptive_set","k4_set","physical_k4_set",
            "guarded_bag","adaptive_bag","k4_bag","physical_k4_bag"
        ]},
        "differences":{k:sums[k] for k in [
            "adaptive_rescues","adaptive_regressions",
            "k4_rescues_vs_guarded","k4_regressions_vs_guarded",
            "physical_k4_rescues_vs_guarded","physical_k4_regressions_vs_guarded"
        ]},
        "coverage":{"adaptive_probed_questions":sums["probed_questions"],
                    "adaptive_fallback_questions":sums["fallback_questions"]},
        "model_calls":calls,"runtime_seconds":runtime,
        "cost_efficiency":{
            "guarded_logical_calls_per_correct_SET":round(calls["guarded_total_logical_calls"]/sums["guarded_set"],3) if sums["guarded_set"] else None,
            "adaptive_logical_calls_per_correct_SET":round(calls["adaptive_total_logical_calls"]/sums["adaptive_set"],3) if sums["adaptive_set"] else None,
            "k4_logical_calls_per_correct_SET":round(calls["k4_total_logical_calls"]/sums["k4_set"],3) if sums["k4_set"] else None,
            "physical_k4_logical_calls_per_correct_SET":round(calls["physical_k4_total_logical_calls"]/sums["physical_k4_set"],3) if sums["physical_k4_set"] else None,
        },
        "physical_database_stratified":{k:dict(v) for k,v in sorted(dbs.items())},
        "prior_k4_is_not_legacy_Guarded":True,
        "historic_BIRD_DEV100_33_to_39_is_NOT_same_cohort":True,
        "no_extra_inference_in_cost_report":True,
        "no_gold_fed_to_any_model":True,
        "new_DEV500_or_reserved_HOLDOUT300_used":False,
        "production_promotion_permitted":False,
        "pairs":cases,
    }
    if out.exists():
        raise FileExistsError("immutable cost report exists")
    out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    return result


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True,type=Path)
    p.add_argument("--manifest",required=True,type=Path)
    for key in ("adaptive-generated","adaptive-scored","k4-generated","k4-scored"):
        p.add_argument("--"+key,required=True,nargs=N_DB,type=Path)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    result=summarize(
        source=args.source,manifest=args.manifest,
        adaptive_generated=args.adaptive_generated,
        adaptive_scored=args.adaptive_scored,
        k4_generated=args.k4_generated,
        k4_scored=args.k4_scored,out=args.out,
    )
    result.pop("pairs",None)
    print("FINAL_BIRD_FOUR_WAY_EX_COST",json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
