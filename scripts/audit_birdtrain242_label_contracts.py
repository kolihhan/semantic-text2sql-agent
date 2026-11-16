"""CPU-only source-label quality screening on immutable BIRD TRAIN SQL SFT candidates.

Detect high-precision question/SQL contract inconsistencies independent of
model predictions and BIRD DEV. Report flags, NOT a semantic oracle.
No automatic label rewrites, no model training, no hidden-holdout access.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

PINNED_TRAIN_SHA="b6851fb0f2b5027833aff45a0b5e01326359b245be1c812edc66986d0df1d402"
EXPECTED_N=242
EXPECTED_DATABASES=16

def _plain(text:str)->str:
    return " ".join(str(text).casefold().split())

def question_sql_flags(question:str, sql:str)->list[str]:
    """Conservative *candidate* source-reference mismatches, never model labels."""
    q=_plain(question)
    s=_plain(sql)
    risks=[]
    # Asked to COUNT items, but SQL returns one row for each distinct id.
    # Exceptions (COUNT over subquery etc.) are prevented by also requiring
    # SELECT DISTINCT at query start and no aggregate COUNT anywhere.
    if re.search(r"\bhow many\b",q) and re.match(r"^select\s+distinct\b",s) and not re.search(r"\bcount\s*\(",s):
        risks.append("asked_count_but_reference_lists_distinct_rows")
    # Highest WINNING PERCENTAGE is not necessarily the most wins. Sorting
    # by winning COUNT alone, with no games played denominator, is suspect.
    if (re.search(r"\bwinning percentage\b",q)
        and re.search(r"\border\s+by\s+count\s*\(",s)
        and not re.search(r"[/]|\b(avg|sum)\s*\(|\bpercent\b",s)):
        risks.append("asked_win_rate_but_reference_ranks_win_count")
    return risks

def audit(train_source:Path)->dict:
    blob=train_source.read_bytes()
    digest=sha256(blob).hexdigest()
    if digest!=PINNED_TRAIN_SHA:
        raise RuntimeError("Frozen original BIRD TRAIN242 source SHA-256 drift")
    rows=[json.loads(line) for line in blob.decode("utf-8").splitlines() if line.strip()]
    if len(rows)!=EXPECTED_N or len({row["source_index"] for row in rows})!=EXPECTED_N:
        raise ValueError("Expected 242 unique original BIRD TRAIN source rows")
    if len({r["source_db_id"] for r in rows})!=EXPECTED_DATABASES:
        raise ValueError("Expected exactly 16 original publisher TRAIN DBs")
    flags=[]
    counts=Counter()
    for row in rows:
        messages=row["messages"]
        if (len(messages)!=3
            or [m.get("role") for m in messages]!=["system","user","assistant"]
            or sha256(messages[1]["content"].encode()).hexdigest()!=row["input_sha256"]):
            raise ValueError("Original publisher prompt/output integrity failure")
        question=messages[1]["content"].split("\nSchema context:\n",1)[0].removeprefix("Question: ").strip()
        gold_sql=messages[2]["content"]
        risks=question_sql_flags(question,gold_sql)
        for risk in risks:
            counts[risk]+=1
            flags.append({
                "source_index":row["source_index"],"db_id":row["source_db_id"],
                "flag":risk,"question":question,"publisher_sql":gold_sql,
            })
    report={
        "status":"ORIGINAL_TRAIN_SOURCE_LABEL_QUALITY_FLAGS_NOT_SEMANTIC_ORACLE",
        "train_sha256":digest,"n_train":len(rows),
        "physical_source_dbs":EXPECTED_DATABASES,
        "flag_count":len(flags),
        "flagged_source_ids":sorted({r["source_index"] for r in flags}),
        "flag_types":dict(sorted(counts.items())),
        "flags":flags,
        "label_quality_claim":"Each flag requires independent human/source semantic adjudication. SQL previously passed SQLite preflight but may conflict with natural-language intent. No automatic gold rewriting.",
        "do_not_train_on_heldout":True,
        "no_bird_dev_used":True,
        "no_model_inference":True,
        "no_accuracy_gain_claim":True,
    }
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    result=audit(a.train)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+"\n")
    print("SQL_TRAIN242_LABEL_CONTRACT_SCREEN",json.dumps({k:v for k,v in result.items() if k!="flags"},sort_keys=True),flush=True)

if __name__=="__main__":
    main()
