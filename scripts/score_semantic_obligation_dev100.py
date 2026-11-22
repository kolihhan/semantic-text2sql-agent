"""Three-stage label-blind historical BIRD DEV100 semantic signal experiment.

No SQL modifications and no new model/SQLite executions. Detection sees only
the previously frozen question + selected SQL (never publisher SQL/EX labels).
Scores are retrospective associations, not new semantic accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.blind_physical_semantic_signals import SOURCE, SOURCE_SHA, MODEL_DIGEST, _canonical_json_hash
from scripts.semantic_obligation_probe import detect_obligations

SIGNALS = (
    "EXPLICIT_ID_BOUND_TO_NON_ID_ATTRIBUTE",
    "NAMED_ENTITY_TOKEN_FILTERED_BY_NONKEY",
    "GROWTH_RATE_WITHOUT_DIVISION",
    "PARENT_COUNT_ON_JOINED_CHILD_ROWS",
)

def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def save(path: Path, value: dict) -> None:
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2)+"\n", encoding="utf-8")

def detect(source: Path, output: Path) -> None:
    d=load(source)
    if (d.get("study") != SOURCE or d.get("status") != "frozen_unlabeled"
            or d.get("n") != 100 or d.get("source_sha256") != SOURCE_SHA
            or d.get("model_digest") != MODEL_DIGEST):
        raise ValueError("unlabeled original source/model mismatch")
    cases=d.get("cases")
    allowed={"case_id","db_id","question","selected_sql","shard"}
    if len(cases)!=100 or any(set(c)!=allowed for c in cases):
        raise ValueError("missing cases or attempted EX/gold label leakage")
    if len({str(c["case_id"]) for c in cases}) != 100:
        raise ValueError("duplicate cases")
    rows=[]
    for c in cases:
        flags=detect_obligations(c["question"], c["selected_sql"])
        rows.append({
            "case_id":str(c["case_id"]), "db_id":str(c["db_id"]),
            "shard":int(c["shard"]), **flags,
        })
    packet={
        "status":"FROZEN_GOLD_BLIND_100_SEMANTIC_OBLIGATIONS",
        "source_study":SOURCE,
        "source_sha256":SOURCE_SHA,
        "candidate_sha256":_canonical_json_hash(d),
        "n":100,
        "signals":list(SIGNALS),
        "cases":rows,
        "gold_accessed":False,
        "no_sql_rewrite":True,
        "no_new_model_calls":True,
    }
    save(output,packet)
    counts=Counter(s for r in rows for s in r["flags"])
    print("SEMANTIC_OBLIGATIONS_FROZEN",json.dumps({
        "n":100,"with_any_signal":sum(bool(r["flags"]) for r in rows),
        "signals":dict(counts),
    },sort_keys=True),flush=True)

def _group(rows: list[dict], name: str | None = None) -> dict:
    executable=[r for r in rows if r["execution_success"]]
    wrong=[r for r in executable if not r["strict_ex"]]
    correct=[r for r in executable if r["strict_ex"]]
    hits=[r for r in executable if name in r["flags"]] if name else [r for r in executable if r["flags"]]
    bad=[r for r in hits if not r["strict_ex"]]
    good=[r for r in hits if r["strict_ex"]]
    return {
        "cases":len(rows),"executable":len(executable),
        "executable_wrong":len(wrong),"executable_correct":len(correct),
        "flagged":len(hits),"flagged_wrong":len(bad),"flagged_correct":len(good),
        "wrong_case_flag_coverage":round(len(bad)/len(wrong),4) if wrong else None,
        "wrong_case_enrichment":round(len(bad)/len(hits),4) if hits else None,
        "wrong_database_count":len({x["db_id"] for x in bad}),
    }

def score(frozen: Path, labels: Path, output: Path) -> None:
    d,s=load(frozen),load(labels)
    if (d.get("status")!="FROZEN_GOLD_BLIND_100_SEMANTIC_OBLIGATIONS"
            or d.get("gold_accessed") is not False or d.get("source_sha256")!=SOURCE_SHA
            or d.get("signals")!=list(SIGNALS) or d.get("n")!=100):
        raise ValueError("unsealed or changed detector")
    if (s.get("status")!="sealed_original_scores" or s.get("study")!=SOURCE
            or s.get("source_sha256")!=SOURCE_SHA or s.get("n")!=100):
        raise ValueError("wrong publisher labels")
    if s.get("original_shards")!={"0":"complete","1":"complete","2":"complete","3":"invalid"}:
        raise ValueError("changed original shard status")
    labels_by={str(x["case_id"]):x for x in s["labels"]}
    if len(labels_by)!=100:
        raise ValueError("missing or duplicate labels")
    seen=set()
    rows=[]
    for r in d["cases"]:
        i=r["case_id"]
        if i in seen or i not in labels_by:
            raise ValueError("duplicate or missing IDs")
        seen.add(i)
        lab=labels_by[i]
        if lab["db_id"]!=r["db_id"] or int(lab["shard"])!=r["shard"]:
            raise ValueError("sealed cohort pairing mismatch")
        if not set(r["flags"]) <= set(SIGNALS):
            raise ValueError("unexpected unregistered signal")
        rows.append({**r,**lab})
    if sum(r["strict_ex"] for r in rows)!=39:
        raise ValueError("stored historical 39/100 source drift")
    valid=[r for r in rows if r["shard_status"]=="complete"]
    invalid=[r for r in rows if r["shard_status"]=="invalid"]
    if len(valid)!=75 or len(invalid)!=25:
        raise ValueError("invalid split not separated")
    def metrics(subset):
        return {"any":_group(subset),
                "signals":{name:_group(subset,name) for name in SIGNALS}}
    report={
        "status":"HISTORICAL_DEV100_SEMANTIC_SIGNAL_CORRELATION_ONLY",
        "historical_ex_correct":39,
        "historical_sample_size":100,
        "original_workflow_invalid_shard_3":True,
        "primary_valid75":metrics(valid),
        "exploratory_invalid25":metrics(invalid),
        "all100_descriptive_only":metrics(rows),
        "source_run":37307108955,
        "new_model_calls":0,
        "new_sql_executions":0,
        "new_execution_match_gains":None,
        "production_or_qlora_modified":False,
        "note":"Physical result nonemptiness is NOT available in original source flags; only execution_success is known. Post-hoc exposed data: signals are not causal correctness labels.",
    }
    save(output,report)
    print("SEMANTIC_OBLIGATIONS_SCORE",json.dumps({
        "primary75":report["primary_valid75"],
        "all100":report["all100_descriptive_only"],
    },sort_keys=True),flush=True)

def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="phase",required=True)
    a=sub.add_parser("detect")
    a.add_argument("--source",type=Path,required=True)
    a.add_argument("--output",type=Path,required=True)
    b=sub.add_parser("score")
    b.add_argument("--frozen",type=Path,required=True)
    b.add_argument("--labels",type=Path,required=True)
    b.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.phase=="detect":detect(args.source,args.output)
    else:score(args.frozen,args.labels,args.output)

if __name__=="__main__":
    main()
