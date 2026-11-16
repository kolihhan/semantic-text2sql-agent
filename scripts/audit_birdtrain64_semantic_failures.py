"""Explore *already exposed* source-valid BIRD TRAIN64 semantic EX failures.

This is a post-hoc human/AI-readable failure taxonomy, NOT a model
intervention, trained classifier, or independent semantic label adjudication.
The supplied IDs/categories MUST equal all physically executable but
reference-SQL nonmatching cases, or the audit fails closed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

SOURCE_SHA="cc112305bc9de33a03cea704f2edecfdfb45e62cc953ae498cea27fbed93b506"
BASELINE_DIGEST="56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
BASELINE_N=64
# Cases are grouped by ONE visible primary candidate mechanism. Group labels
# are hypotheses read from the question, publisher SQL and model SQL after
# scoring. Overlaps and subjective adjudication are possible. Not training tags.
TAXON={
    "suspected_reference_question_mismatch":(6969,5867),
    "projection_or_result_shape":(6951,6955,1560,1573,1785,5823,5844,5895),
    "table_join_filter_or_identifier_grounding":(6835,6958,1570,1572,1728,1766,1780,7059),
    "aggregation_ratio_or_topk_semantics":(6812,6855,1561,1584,1784),
}
# These specific source evidence contradictions are not a scalable oracle.
LABEL_WARNINGS={
    6969:"Question requires both count and customer names, reference SQL projects COUNT only.",
    5867:"Question requests smallest surface area and most crowded city, reference prioritizes city population DESC then country area DESC with no MIN area constraint.",
}

def audit(baseline:dict,validation_raw:bytes)->dict:
    if sha256(validation_raw).hexdigest()!=SOURCE_SHA:
        raise ValueError("Original BIRD TRAIN publisher-validation64 SHA mismatch")
    source=[json.loads(x) for x in validation_raw.decode("utf-8").splitlines() if x.strip()]
    if len(source)!=64 or len({r["source_index"] for r in source})!=64:
        raise ValueError("Original 64 label source incomplete or repeated")
    records=baseline.get("cases",[])
    if (baseline.get("n")!=BASELINE_N or baseline.get("status")!="complete"
        or baseline.get("source_validation_sha256")!=SOURCE_SHA
        or baseline.get("model_digest")!=BASELINE_DIGEST or len(records)!=64):
        raise ValueError("Original physical Qwen9B model/eval source mismatch")
    lookup={r["source_index"]:r for r in source}
    wrong={}
    for row in records:
        idx=row["source_index"]
        if idx not in lookup:
            raise ValueError("Baseline source index not present")
        out=row["guarded"]
        if out["execution_ok"] and not out["ex"]:
            wrong[idx]=row
    if len(wrong)!=23:
        raise ValueError("Original 23 physically executable wrong-result cases missing")
    labeled=[j for group in TAXON.values() for j in group]
    if len(labeled)!=len(set(labeled)) or set(labeled)!=set(wrong):
        raise ValueError("Human-led taxonomy IDs do not equal all observed failures")

    for idx in LABEL_WARNINGS:
        row=lookup[idx]
        question=row["messages"][1]["content"].split("\nSchema context:\n",1)[0].removeprefix("Question: ").lower()
        gold=row["messages"][2]["content"].lower()
        if idx==6969:
            if not ("how many" in question and "list those customer names" in question
                    and "count(" in gold and "first_name" not in gold and "last_name" not in gold):
                raise ValueError("Suspected question-vs-label contradiction changed")
        if idx==5867:
            if not ("smallest surface area" in question and "most crowded city" in question
                    and "surfacearea desc" in gold and "population desc" in gold
                    and "min(" not in gold):
                raise ValueError("Suspected semantic gold-label contradiction changed")
    score=baseline
    if (score.get("guarded_ex")!=31 or score.get("guarded_exec")!=54
        or score.get("direct_ex")!=27 or score.get("n")!=64):
        raise ValueError("Frozen baseline aggregate differs from verified original")
    report={
        "status":"POSTHOC_TAXONOMY_NOT_CAUSAL_OR_NEW_ACCURACY",
        "source":"original BIRD TRAIN physical DB-separated VAL64, Actions 43892823922",
        "source_validation_sha256":SOURCE_SHA,
        "base_model_digest":BASELINE_DIGEST,
        "n":64,
        "strict_ex":31,
        "executable":54,
        "executable_but_wrong":len(wrong),
        "nonexecuting":10,
        "taxonomy":{k:{"count":len(v),"source_indices":list(v)} for k,v in TAXON.items()},
        "suspected_gold_question_mismatches":[
            {"source_index":i,"reason":LABEL_WARNINGS[i]}
            for i in sorted(LABEL_WARNINGS)
        ],
        "claim_limitations":"Class labels are post-hoc hypotheses and can overlap; reference contradictions are candidates, not independently human-adjudicated. Do not train on these heldout64 labels or tune against them. A Qwen SQL prediction may be semantically reasonable even when EX disagrees with a flawed publisher reference.",
        "no_model_calls":True,
        "no_training":True,
        "new_accuracy_improvement":None,
    }
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--validation",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    result=audit(json.loads(args.baseline.read_text()),args.validation.read_bytes())
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("SQL64_ERROR_TAXONOMY",json.dumps(result,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
