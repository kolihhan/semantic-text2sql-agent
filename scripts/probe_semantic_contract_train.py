"""Independent 16-DB BIRD TRAIN242 --> 4-DB TRAIN-VAL64 quantifier probe.

The two explicit rules live in evaluation/semantic_contract_selector.py and
are frozen BEFORE independent VAL64 is read. TRAIN242 publisher reference SQL
is used only to audit possible false-rejection of reference statements.
VAL64 has ORIGINAL actual Qwen3.5-9B direct and guarded SQL; it is diagnostic,
not the K8 generation distribution. Validation checks require exact upstream
SHA model/source hashes and fixed 64-case score. 0 new model calls.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from evaluation.semantic_contract_selector import (
    compatibility, contract_for_question, question_only,
)
from evaluation.kcache_protocol import MODEL_DIGEST

TRAIN242_SHA="b6851fb0f2b5027833aff45a0b5e01326359b245be1c812edc66986d0df1d402"
VAL64_SHA="cc112305bc9de33a03cea704f2edecfdfb45e62cc953ae498cea27fbed93b506"
MODEL="qwen3.5:9b-q4_K_M"
PROTOCOL="ORIGINAL-BIRD-TRAIN242-VAL64-QUANTIFIER-SEMANTIC-CONTRACT-V1"


def _lines(path:Path,expected_sha:str,expected_n:int)->list[dict]:
    raw=path.read_bytes()
    if sha256(raw).hexdigest()!=expected_sha:
        raise ValueError("frozen original source digest mismatch")
    rows=[json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(rows)!=expected_n or len({r["source_index"] for r in rows})!=expected_n:
        raise ValueError("original publisher source incomplete or repeated")
    for r in rows:
        messages=r["messages"]
        if ([m["role"] for m in messages]!=["system","user","assistant"] or
            sha256(messages[1]["content"].encode()).hexdigest()!=r["input_sha256"]):
            raise ValueError("original question/schema/label alignment drift")
    return rows


def train_reference_safety(rows:list[dict])->dict:
    if len(rows)!=242 or len({r["source_db_id"] for r in rows})!=16:
        raise ValueError("must use all 242 immutable TRAIN with 16 databases")
    sample=Counter(); violations=Counter(); names=Counter()
    for row in rows:
        question=question_only(row["messages"][1]["content"])
        category=contract_for_question(question)
        if category=="abstain":
            continue
        sample[category]+=1
        signal=compatibility(question,row["messages"][2]["content"])
        names[str(signal)]+=1
        if signal==-1:
            violations[category]+=1
    return {"n":len(rows),"dbs":16,"source_sha256":TRAIN242_SHA,
            "category_cases":dict(sample),"publisher_reference_flags":dict(violations),
            "all_contract_signals":dict(names),
            "publisher_label_quality_warning":"Reference SQL is not always semantically correct; source flags are NOT adjudicated true positives. No publisher SQL is shown to inference."}


def independent_real_model_validation(source:list[dict],baseline:dict,
                                      train_db_ids:set[str])->dict:
    if len(source)!=64 or len({x["source_db_id"] for x in source})!=4:
        raise ValueError("expected exact four independent TRAIN validation physical DBs")
    validation_dbs={r["source_db_id"] for r in source}
    if train_db_ids & validation_dbs:
        raise ValueError("TRAIN and TRAIN VAL physical databases overlap")
    if (baseline.get("status")!="complete" or baseline.get("n")!=64 or
        baseline.get("model_digest")!=MODEL_DIGEST or
        baseline.get("source_validation_sha256")!=VAL64_SHA or
        baseline.get("direct_ex")!=27 or baseline.get("guarded_ex")!=31 or
        len(baseline.get("cases",[]))!=64):
        raise ValueError("original 64 real Qwen results/model/source drift")
    lookup={x["source_index"]:x for x in source}
    counts=Counter()
    flagged_correct=Counter(); flagged_wrong=Counter(); kinds=Counter()
    changed=[]
    rescues=regressions=0
    observed_direct=observed_guarded=0
    for case in baseline["cases"]:
        idx=case["source_index"]
        if idx not in lookup or case["db_id"]!=lookup[idx]["source_db_id"]:
            raise ValueError("real model case not matched to original source")
        prompt=lookup[idx]["messages"][1]["content"]
        if sha256(prompt.encode()).hexdigest()!=case["prompt_sha256"]:
            raise ValueError("real model input prompt SHA misaligned")
        question=question_only(prompt)
        category=contract_for_question(question)
        kinds[category]+=1
        d=case["direct"];g=case["guarded"]
        if not isinstance(d.get("ex"),bool) or not isinstance(g.get("ex"),bool):
            raise ValueError("real physical direct/guarded label missing")
        observed_direct+=int(d["ex"]);observed_guarded+=int(g["ex"])
        direct_signal=compatibility(question,case["direct_sql"])
        guarded_signal=compatibility(question,case["guarded_sql"])
        for pair,signal in ((d,direct_signal),(g,guarded_signal)):
            if signal==-1:
                (flagged_correct if pair["ex"] else flagged_wrong)[category]+=1
        # Diagnostic candidate pair, NOT re-generating or re-scoring SQL.
        # Restrict to originally both physically executable candidates, so
        # compiler/tool repair is not confused with semantic selection.
        eligible=(d["execution_ok"] and g["execution_ok"])
        switch=bool(eligible and direct_signal==-1 and guarded_signal==1)
        if switch:
            counts["switched"]+=1
            rescues+=int(not d["ex"] and g["ex"])
            regressions+=int(d["ex"] and not g["ex"])
            changed.append({"source_index":idx,"db_id":case["db_id"],
                            "source_category":category,
                            "direct_correct":d["ex"],"guarded_correct":g["ex"]})
        elif (eligible and direct_signal==-1 and guarded_signal==-1):
            counts["both_violate"]+=1
    if observed_direct!=27 or observed_guarded!=31:
        raise ValueError("original source 64 physical score mismatch")
    if len({row["source_index"] for row in changed})!=len(changed):
        raise ValueError("changed pair duplicated source IDs")
    changed_out=counts["switched"]
    return {
        "physical_validation_n":64,"physical_validation_dbs":4,
        "validation_sha256":VAL64_SHA,"model_digest":MODEL_DIGEST,
        "original_direct_ex":27,"original_guarded_ex":31,
        "question_contract_types":dict(kinds),
        "flagged_correct_model_sql_by_contract":dict(flagged_correct),
        "flagged_wrong_model_sql_by_contract":dict(flagged_wrong),
        "actionable_direct_vs_guarded_pairs":changed_out,
        "pair_rescues":rescues,"pair_regressions":regressions,
        "pair_net":rescues-regressions,
        "pair_same_source_baseline_correct":27,
        "pair_semantic_choice_correct":27+rescues-regressions,
        "pairs_where_both_candidates_violate_contract":counts["both_violate"],
        "pair_change_categories":dict(Counter(x["source_category"] for x in changed)),
        # These conditions are frozen before independent VAL64; failure is
        # NOT a statistical proof of no semantic-verification research value.
        "minimum_positive_feasibility_gate":(
            changed_out>=5 and rescues>=2 and regressions==0 and
            sum(flagged_correct.values())<=1
        ),
        "source_indices_selected_for_diagnostic_not_training":[x["source_index"] for x in changed],
        "claim_boundary":"Two heterogeneous ORIGINAL Qwen direct/guarded candidate paths on 64 independent TRAIN-source database questions; guarded may use repair/model calls. Pair switching diagnostic is not a clean fixed-K8 selector accuracy estimate. No original model new calls, no DEV500/holdout access, no full-field semantic adjudication."
    }


def audit(train_path:Path,val_path:Path,summary_path:Path)->dict:
    train=_lines(train_path,TRAIN242_SHA,242)
    source_val=_lines(val_path,VAL64_SHA,64)
    physical_train_dbs={x["source_db_id"] for x in train}
    baseline=json.loads(summary_path.read_text("utf-8"))
    result={
        "status":"INDEPENDENT_TRAIN_DATABASE_SEPARATED_QUANTIFIER_CONTRACT_DIAGNOSTIC_COMPLETE",
        "protocol":PROTOCOL,
        "training_reference_safety":train_reference_safety(train),
        "independent_physical_qwen9b_val64":independent_real_model_validation(
            source_val,baseline,physical_train_dbs),
        "new_llm_calls":0,"new_gpu_calls":0,"sql_qlora_training":False,
        "bird_dev500_used":False,"bird_holdout300_used":False,
        "not_a_deployed_selector_or_prospective_EX_gain":True,
        "next_gate":"Only if predeclared 64-case feasibility succeeds, freeze a semantic candidate reranking method for further cost-controlled evaluation on independent source; do not tune this rule on observed validation labels."
    }
    return result


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--validation",type=Path,required=True)
    p.add_argument("--model-summary",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():
        raise FileExistsError("never overwrite source-locked experiment evidence")
    result=audit(a.train,a.validation,a.model_summary)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("INDEPENDENT_TRAIN64_REAL_MODEL_CONTRACT_PROBE",
          json.dumps({**result,"training_reference_safety":result["training_reference_safety"],
              "independent_physical_qwen9b_val64":{
                  k:v for k,v in result["independent_physical_qwen9b_val64"].items()
                  if k!="source_indices_selected_for_diagnostic_not_training"}},sort_keys=True),
          flush=True)


if __name__=="__main__":
    main()
