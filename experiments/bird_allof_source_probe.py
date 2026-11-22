"""Cheap BIRD-only source-first coverage screen for 'both' relational division.

Reads only question/DB from the pinned original BIRD TRAIN source (never SQL,
gold label, reference result or prior model outputs). Never calls Qwen.
This tests whether a source-grounded ALL-of physical counterexample is worth
a real-model experiment before spending even one GPU/CPU inference call.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

from scripts.prepare_birdtrain_qwen9b_sft_source import TRAIN_SHA, load_checked

PROTOCOL="BIRD-TRAIN-ALLOF-QUESTION-ONLY-SOURCE-READINESS-V1"
PROTECTED=frozenset({"simpson_episodes","works_cycles"})
SEED="bird-allof-question-only-freeze-before-qwen-20261123"


def probe(rows: list[dict], source_sha: str) -> dict:
    eligible=[]
    by_db=Counter()
    for index,row in enumerate(rows):
        # Strictly source-only fields; never read row["SQL"] or row["difficulty"].
        db=row.get("db_id")
        question=row.get("question")
        if (not isinstance(db,str) or db in PROTECTED
                or not isinstance(question,str)
                or re.search(r"\bboth\b",question,re.I) is None):
            continue
        # Restrict to actual conjunctive semantics, not 'both groups' as
        # two individually requested answer columns or "both ... or ...".
        if re.search(r"\bboth\b.{0,110}\band\b",question,re.I) is None:
            continue
        qhash=sha256(question.encode()).hexdigest()
        eligible.append({"source_index":index,"db_id":db,
                         "question_sha256":qhash})
        by_db[db]+=1
    eligible.sort(key=lambda row:sha256(
        f"{SEED}:{row['source_index']}:{row['db_id']}".encode()).hexdigest())
    selected=[]
    per_db=Counter()
    for row in eligible:
        if per_db[row["db_id"]]>=2 or len(selected)>=24:
            continue
        selected.append(row)
        per_db[row["db_id"]]+=1
    return {"protocol":PROTOCOL,"status":"SOURCE_ONLY_FEASIBILITY_NOT_BIRD_EX",
            "benchmark":"BIRD TRAIN","original_source_sha256":source_sha,
            "eligible_both_questions":len(eligible),
            "eligible_physical_dbs":len(by_db),
            "selected_question_only_n":len(selected),
            "selected_db_count":len(per_db),
            "selected":selected,
            "source_question_only_gold_SQL_not_accessed":True,
            "protected_two_TRAIN_DBS_untouched":True,
            "HOLDOUT300_unused":True,
            "real_Qwen_calls":0,
            "warning":"Selected IDs can overlap historical exposed questions: this is a CPU eligibility screen, not an independent EX cohort or model performance improvement."}


def main() -> None:
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    rows=load_checked(a.train,TRAIN_SHA)
    result=probe(rows,TRAIN_SHA)
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("BIRD_ALLOF_SOURCE_SCREEN",json.dumps(
        {k:v for k,v in result.items() if k!="selected"},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
