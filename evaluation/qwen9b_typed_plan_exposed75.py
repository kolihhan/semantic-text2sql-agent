"""Large-sample extension of the already frozen Qwen9B historical DEV100 screen.

Preserves source hashing, original SQL, immutable Qwen digest, and gold-blind
decision/scoring split. 75 rows are from the THREE historically COMPLETE shards;
the fourth invalid shard is excluded, NOT silently replaced. Historical/exposed,
not an independent holdout or model-improvement promotion benchmark.

Each of 3 runner jobs processes 25 cases. Exactly the original 12 pilot rows
are a subset of this 75-case selection (pilot is not an independent replicate).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import statistics

from evaluation import qwen9b_typed_plan_exposed12 as core

N_TOTAL = 75
N_PER_SHARD = 25
N_SHARDS = 3
COHORT = "EXPOSED_HISTORICAL_DEV100_COMPLETE75_FIXED_QWEN9B_V1"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_full(f: dict) -> None:
    if (
        f.get("protocol") != "HISTORICALLY_EXPOSED_DEV100_SCV12_SOURCE_LOCKED_V1"
        or f.get("status") != "FROZEN_GOLD_FREE"
        or f.get("gold_sql_present") is not False
        or f.get("model_digest") != core.QWEN_DIGEST
        or f.get("source_sha256") != core.DEV100_SOURCE_SHA
        or len(f.get("selected", [])) != N_TOTAL
    ):
        raise ValueError("historical 75 cohort failed source-only integrity contract")
    ids = [x["case_id"] for x in f["selected"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate selected historical case")
    assert all("official_ex" not in x and "SQL" not in x for x in f["selected"])


def freeze75(source: Path, shards: Path, output: Path) -> None:
    core.N = N_TOTAL
    result = core.freeze(source=source, shards=shards, output=output)
    _validate_full(result)
    print("QWEN9B_EXPOSED75_GOLD_FREE_FROZEN", json.dumps({
        "count":N_TOTAL, "dbs":len({x["db_id"] for x in result["selected"]}),
        "cohort_sha256":sha256(json.dumps([x["case_id"] for x in result["selected"]]).encode()).hexdigest(),
        "historical_shard_provenance":"3 complete, 1 invalid excluded",
        "pilot_cases_retained":12,
    }),flush=True)


def split75(frozen: Path, shard: int, output: Path) -> None:
    f=_read(frozen)
    _validate_full(f)
    if shard not in range(N_SHARDS):
        raise ValueError("invalid 25-case shard index")
    subset=f["selected"][shard*N_PER_SHARD:(shard+1)*N_PER_SHARD]
    if len(subset)!=N_PER_SHARD:
        raise ValueError("missing frozen 25-case range")
    payload={**f, "sample_size":N_PER_SHARD, "selected":subset,
             "exposed75_cohort":COHORT,"shard":shard,
             "shard_original_ids_sha256":sha256(json.dumps([x["case_id"] for x in subset]).encode()).hexdigest()}
    core.write_exclusive(output,payload)
    print("QWEN9B_EXPOSED75_SHARD_GOLD_FREE",shard,len(subset),flush=True)


def infer25(frozen: Path, db_root: Path, output: Path) -> None:
    f=_read(frozen)
    if (
        f.get("exposed75_cohort")!=COHORT
        or f.get("sample_size")!=N_PER_SHARD
        or len(f.get("selected",[]))!=N_PER_SHARD
    ):
        raise ValueError("not a frozen 75-case shard")
    core.N=N_PER_SHARD
    core.infer(frozen=frozen,db_root=db_root,output=output)


def score25(frozen: Path, blind: Path, source: Path, db_root: Path, output: Path) -> None:
    f=_read(frozen)
    if f.get("exposed75_cohort")!=COHORT or len(f.get("selected",[]))!=N_PER_SHARD:
        raise ValueError("source not an approved frozen 75-case shard")
    d=_read(blind)
    if d.get("status")!="COMPLETED_GOLD_BLIND_DECISIONS" or len(d.get("cases",[]))!=N_PER_SHARD:
        raise ValueError("inference not sealed; NEVER open reference SQL")
    if [x["case_id"] for x in d["cases"]] != [x["case_id"] for x in f["selected"]]:
        raise ValueError("blinded decision case order mismatched")
    core.N=N_PER_SHARD
    core.score(frozen=frozen,blind=blind,source=source,db_root=db_root,output=output)


def _exact_p(rescues: int, regressions: int) -> float:
    n=rescues+regressions
    return min(1.0,2*sum(math.comb(n,k) for k in range(min(rescues,regressions)+1)) / (2**n)) if n else 1.0


def aggregate75(*, frozen: Path, scored: list[Path], blinds: list[Path], output: Path) -> dict:
    f=_read(frozen)
    _validate_full(f)
    if len(scored)!=N_SHARDS or len(blinds)!=N_SHARDS:
        raise ValueError("need 3 independently complete original source shards")
    outcomes=[]
    wall_time=defaultdict(list)
    unverified=[]
    for shard in range(N_SHARDS):
        s=_read(scored[shard])
        b=_read(blinds[shard])
        if (
            s.get("status")!="COMPLETE_HISTORICAL_SOURCE_ONLY_NOT_INDEPENDENT"
            or s.get("model_digest")!=core.QWEN_DIGEST
            or s.get("n_selected")!=N_PER_SHARD
            or b.get("status")!="COMPLETED_GOLD_BLIND_DECISIONS"
            or b.get("model_digest")!=core.QWEN_DIGEST
        ):
            raise ValueError("incomplete or non-original model decision/score shard")
        intended=f["selected"][shard*N_PER_SHARD:(shard+1)*N_PER_SHARD]
        if [x["case_id"] for x in s["cases"]] != [x["case_id"] for x in intended]:
            raise ValueError("score cases not the exact source cohort")
        if [x["case_id"] for x in b["cases"]] != [x["case_id"] for x in intended]:
            raise ValueError("blinded decisions not the exact source cohort")
        for c,d in zip(s["cases"],b["cases"]):
            if c["case_id"]!=d["case_id"] or c["db"]!=d["db_id"]:
                raise ValueError("blind/score mismatch")
            for arm in ("always","selective"):
                wall_time[arm].append(d[arm]["latency_s"])
                # A verifier's failure to verify is not equivalent to correctness.
                if d[arm]["status"]=="semantic_unverified":
                    unverified.append({"case_id":c["case_id"],"policy":arm})
        outcomes.extend(s["cases"])
    if len(outcomes)!=N_TOTAL or len({c["case_id"] for c in outcomes})!=N_TOTAL:
        raise ValueError("missing or duplicate sampled historical cases")
    eligible=[c for c in outcomes if c["scorable"]]
    by_db=defaultdict(list)
    for c in eligible:
        by_db[c["db"]].append(c)
    totals={}
    for arm in ("baseline","always","selective"):
        totals[arm]={
            "set_correct":sum(bool(c[arm]["set_ex"]) for c in eligible),
            "bag_correct":sum(bool(c[arm]["bag_ex"]) for c in eligible),
            "call_count_incremental":sum(c[arm]["model_calls"] for c in outcomes),
            "statuses":dict(Counter(c[arm]["status"] for c in outcomes)),
        }
        if arm!="baseline":
            rescues=sum(c[arm]["set_ex"] and not c["baseline"]["set_ex"] for c in eligible)
            regressions=sum(c["baseline"]["set_ex"] and not c[arm]["set_ex"] for c in eligible)
            totals[arm].update({
                "rescues":rescues,"regressions":regressions,"net_rescues_minus_regressions":rescues-regressions,
                "mean_latency_seconds":round(statistics.mean(wall_time[arm]),3),
                "paired_mcnemar_exact_two_sided_p":round(_exact_p(rescues,regressions),6),
            })
            # Fixed-seed physical-DB cluster bootstrap accounts for shared DB dependence,
            # not just 75 independent Bernoulli observations.
            db_ids=sorted(by_db)
            rng=random.Random(20261121)
            draws=[]
            for _ in range(2000):
                sample=[c for db in (rng.choice(db_ids) for i in db_ids) for c in by_db[db]]
                if sample:
                    draws.append((sum(bool(c[arm]["set_ex"])-bool(c["baseline"]["set_ex"]) for c in sample) /
                                  len(sample))*100)
            draws.sort()
            totals[arm]["db_cluster_bootstrap_paired_delta_pp_95ci"]=[
                round(draws[int(0.025*len(draws))],2),
                round(draws[int(0.975*len(draws))-1],2),
            ] if draws else None
    report={
        "protocol":COHORT, "status":"COMPLETE_DIAGNOSTIC_ONLY",
        "n_original":N_TOTAL,"n_scorable":len(eligible),
        "n_original_complete_historical_shards":3,
        "n_original_invalid_historical_shards_EXCLUDED":1,
        "n_previously_used_in_12_case_pilot":12,
        "n_additional_exposed_cases":63,
        "n_physical_databases":len(by_db),
        "source_model":"qwen3.5:9b-q4_K_M","source_model_digest":core.QWEN_DIGEST,
        "source_sha256":core.DEV100_SOURCE_SHA,
        "baseline_sql_frozen_and_shared_between_policies":True,
        "sql_model_training":False,
        "reference_gold_opened_only_after_blind_inference":True,
        "protected_DEV500_or_HOLDOUT300_opened":False,
        "not_independent":"historically exposed DEV100 and 12-case pilot included",
        "performance_promotion_allowed":False,
        "scores":totals,
        "by_database":[{
            "db":db,"n":len(v),
            "baseline_set_correct":sum(c["baseline"]["set_ex"] for c in v),
            "always_set_correct":sum(c["always"]["set_ex"] for c in v),
            "selective_set_correct":sum(c["selective"]["set_ex"] for c in v),
        } for db,v in sorted(by_db.items())],
        "paired_cases":outcomes,
    }
    core.write_exclusive(output,report)
    print("QWEN9B_EXPOSED75_PAIRED_STRICT_EX",json.dumps({
        k:v for k,v in report.items() if k not in ("paired_cases","by_database")
    },sort_keys=True),flush=True)
    return report


def main():
    p=argparse.ArgumentParser()
    x=p.add_subparsers(dest="phase",required=True)
    a=x.add_parser("freeze");a.add_argument("--source",type=Path,required=True);a.add_argument("--shards",type=Path,required=True);a.add_argument("--out",type=Path,required=True)
    a=x.add_parser("split");a.add_argument("--frozen",type=Path,required=True);a.add_argument("--shard",type=int,required=True);a.add_argument("--out",type=Path,required=True)
    a=x.add_parser("infer");a.add_argument("--frozen",type=Path,required=True);a.add_argument("--db-root",type=Path,required=True);a.add_argument("--out",type=Path,required=True)
    a=x.add_parser("score");a.add_argument("--frozen",type=Path,required=True);a.add_argument("--blind",type=Path,required=True);a.add_argument("--source",type=Path,required=True);a.add_argument("--db-root",type=Path,required=True);a.add_argument("--out",type=Path,required=True)
    a=x.add_parser("aggregate");a.add_argument("--frozen",type=Path,required=True);a.add_argument("--scored",type=Path,nargs=3,required=True);a.add_argument("--blinds",type=Path,nargs=3,required=True);a.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.phase=="freeze":freeze75(args.source,args.shards,args.out)
    elif args.phase=="split":split75(args.frozen,args.shard,args.out)
    elif args.phase=="infer":infer25(args.frozen,args.db_root,args.out)
    elif args.phase=="score":score25(args.frozen,args.blind,args.source,args.db_root,args.out)
    else:aggregate75(frozen=args.frozen,scored=args.scored,blinds=args.blinds,output=args.out)


if __name__=="__main__":
    main()
