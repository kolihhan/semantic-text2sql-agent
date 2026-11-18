"""TRAIN242 original BIRD evidence-vs-SQL contract and real Qwen VAL64 test.

The contract code is frozen by source blob SHA in the workflow. Publisher SQL
is used only to estimate false-positive risk on original TRAIN references.
Never read or use DEV500 or HOLDOUT300. Validation source consists of four
physically disjoint BIRD TRAIN databases, and the same original Qwen3.5-9B
real model outputs already recorded previously. No new model calls.
"""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
import argparse
import json
from pathlib import Path

from evaluation.evidence_sql_contract import evidence_signal,select_two_original_qwen_candidates

TRAIN_SOURCE_SHA="b6851fb0f2b5027833aff45a0b5e01326359b245be1c812edc66986d0df1d402"
VAL_SOURCE_SHA="cc112305bc9de33a03cea704f2edecfdfb45e62cc953ae498cea27fbed93b506"
FIXED_QWEN_DIGEST="56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
PROTOCOL="BIRD-TRAIN242-TO-TRAIN-VAL64-QWEN35-9B-EVIDENCE-VS-SQL-CONTRACT-V1"


def checked_source(path:Path,expected_sha:str,n:int,db_count:int)->list[dict]:
    raw=path.read_bytes()
    if sha256(raw).hexdigest()!=expected_sha:
        raise ValueError("Original BIRD TRAIN source SHA mismatch")
    rows=[json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows)!=n or len({x["source_index"] for x in rows})!=n:
        raise ValueError("Original BIRD TRAIN source row-count or unique IDs drift")
    if len({x["source_db_id"] for x in rows})!=db_count:
        raise ValueError("Original BIRD TRAIN physical database count mismatch")
    for r in rows:
        m=r["messages"]
        if ([x.get("role") for x in m]!=["system","user","assistant"] or
            sha256(m[1]["content"].encode()).hexdigest()!=r["input_sha256"]):
            raise ValueError("Original prompt and publisher reference SQL not source aligned")
    return rows


def train_reference_screen(rows:list[dict])->dict:
    if len(rows)!=242 or len({r["source_db_id"] for r in rows})!=16:
        raise ValueError("TRAIN242 source is not 16 physically distinct databases")
    counters=Counter()
    mismatches=Counter()
    mismatched_ids=[]
    for row in rows:
        prompt=row["messages"][1]["content"]
        sql=row["messages"][2]["content"]
        signal=evidence_signal(prompt,sql)
        if signal.covered:
            counters["reference_contract_covered"]+=1
            if signal.explicit_value_requirements>0:
                counters["explicit_value_covered"]+=1
            if signal.division_check!=0:
                counters["explicit_division_covered"]+=1
            if signal.violation:
                mismatched_ids.append(row["source_index"])
                mismatches[row["source_db_id"]]+=1
                if signal.missing_values:
                    counters["reference_literal_violation"]+=1
                if signal.division_check==-1:
                    counters["reference_division_violation"]+=1
    return {
        "n":242,"physical_dbs":16,"source_sha256":TRAIN_SOURCE_SHA,
        "coverage":dict(counters),
        "reference_violations":len(mismatched_ids),
        "reference_violation_rate_among_covered":(
            len(mismatched_ids)/counters["reference_contract_covered"]
            if counters["reference_contract_covered"] else 0),
        "reference_violation_dbs":dict(sorted(mismatches.items())),
        "reference_violation_source_indices":sorted(mismatched_ids),
        "claim_boundary":"Publisher reference SQL vs publisher-provided evidence discrepancies are UNADJUDICATED safety flags, not verified bad labels. Never show publisher SQL to student or selector."
    }


def independent_original_model64(rows:list[dict],original:dict,train_dbs:set[str])->dict:
    if train_dbs & {r["source_db_id"] for r in rows}:
        raise ValueError("16 training and 4 validation physical databases overlap")
    if (original.get("status")!="complete" or original.get("n")!=64 or
        original.get("source_validation_sha256")!=VAL_SOURCE_SHA or
        original.get("model_digest")!=FIXED_QWEN_DIGEST or
        original.get("direct_ex")!=27 or original.get("guarded_ex")!=31 or
        len(original.get("cases",[]))!=64):
        raise ValueError("Original Qwen3.5-9B fixed-model physical evaluation drift")
    lookup={r["source_index"]:r for r in rows}
    seen=set(); coverage=Counter()
    flagged_correct=Counter();flagged_wrong=Counter();covered_correct=Counter()
    pair=Counter();changed=[]
    for case in original["cases"]:
        idx=case["source_index"]
        if idx in seen or idx not in lookup:
            raise ValueError("Incomplete or duplicate original original-model case")
        seen.add(idx)
        row=lookup[idx]
        prompt=row["messages"][1]["content"]
        if case["db_id"]!=row["source_db_id"] or sha256(prompt.encode()).hexdigest()!=case["prompt_sha256"]:
            raise ValueError("Original Qwen question/prompt/db misaligned")
        arms={}
        for arm in ("direct","guarded"):
            signal=evidence_signal(prompt,case[arm+"_sql"])
            arms[arm]=signal
            if signal.covered:
                coverage[arm]+=1
                if case[arm]["ex"]:
                    covered_correct[arm]+=1
                if signal.violation:
                    (flagged_correct if case[arm]["ex"] else flagged_wrong)[arm]+=1
        result=select_two_original_qwen_candidates(
            prompt,case["direct_sql"],case["guarded_sql"],
            bool(case["direct"]["execution_ok"]),bool(case["guarded"]["execution_ok"]))
        if result["switch_to_guarded"]:
            pair["switched"]+=1
            if not case["direct"]["ex"] and case["guarded"]["ex"]:
                pair["rescues"]+=1
            elif case["direct"]["ex"] and not case["guarded"]["ex"]:
                pair["regressions"]+=1
            else:
                pair["ties"]+=1
            changed.append({"source_index":idx,"db_id":case["db_id"],
                            "direct_correct":case["direct"]["ex"],
                            "guarded_correct":case["guarded"]["ex"]})
    if len(seen)!=64:
        raise ValueError("Original four-database Qwen64 model outputs incomplete")
    positive_gate=(
        sum(flagged_wrong.values())>=5 and
        sum(flagged_correct.values())<=2 and
        pair["switched"]>=2 and pair["rescues"]>=2 and
        pair["regressions"]==0
    )
    return {
        "real_original_Qwen_n":64,"physical_dbs":4,
        "fixed_model_digest":FIXED_QWEN_DIGEST,
        "original_direct_ex":27,"original_guarded_ex":31,
        "covered_original_model_arm_counts":dict(coverage),
        "covered_but_correct_original_model_arm_counts":dict(covered_correct),
        "actual_wrong_model_SQL_flagged":dict(flagged_wrong),
        "actual_correct_model_SQL_falsely_flagged":dict(flagged_correct),
        "eligible_direct_to_guarded_switches":pair["switched"],
        "pair_rescues":pair["rescues"],"pair_regressions":pair["regressions"],
        "pair_equal_quality":pair["ties"],
        "pair_ex_baseline_direct_correct":27,
        "pair_ex_selected_correct":27+pair["rescues"]-pair["regressions"],
        "predeclared_high_precision_positive_gate":positive_gate,
        "selection_diagnostic_case_ids":[x["source_index"] for x in changed],
        "limits":"VAL64 source DB disjoint from 16 training DBs, but 64 original model cases were historically scored; this is a retrospective fixed-model feasibility test, NOT fresh independent generalization. Direct and guarded are heterogeneous pipelines, NOT frozen K8 candidate samples. Outcome labels are scored only after selecting source-only rule. No new EX or training."
    }


def audit(train:Path,val:Path,model:Path)->dict:
    tr=checked_source(train,TRAIN_SOURCE_SHA,242,16)
    va=checked_source(val,VAL_SOURCE_SHA,64,4)
    original=json.loads(model.read_text("utf-8"))
    return {
        "status":"ORIGINAL_QWEN_TRAIN242_TO_PHYSICAL_VAL64_EVIDENCE_CONTRACT_SCREEN_COMPLETE",
        "protocol":PROTOCOL,
        "original_publisher_train242":train_reference_screen(tr),
        "original_Qwen9B_physical_val64":independent_original_model64(
            va,original,{r["source_db_id"] for r in tr}),
        "new_qwen_inference_calls":0,"new_GPU_training_steps":0,
        "fixed_base_model_Qwen_3_5_9B":True,
        "DEV500_used":False,"HOLDOUT300_used":False,
        "next_gate":"If frozen positive gate fails, do not run DEV500 or original Qwen generation for this rule. If it passes, preregister larger physically new DB evaluation; do not score-tune on 64 previously exposed outcomes."
    }


def main()->None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--train",type=Path,required=True)
    ap.add_argument("--validation",type=Path,required=True)
    ap.add_argument("--model",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists():
        raise FileExistsError("cannot overwrite source experiment evidence")
    result=audit(a.train,a.validation,a.model)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("EVIDENCE_CONTRACT_TRAIN_VAL_ORIGINAL_QWEN_PHYSICAL_RESULT",
          json.dumps({**result,"original_Qwen9B_physical_val64":{
              k:v for k,v in result["original_Qwen9B_physical_val64"].items()
              if k!="selection_diagnostic_case_ids"}},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
