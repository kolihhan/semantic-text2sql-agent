"""Original BIRD TRAIN VAL64 source-only feasibility screen for physical value linker.

Read *only* original question text and 4 source SQLite files; never open
publisher gold SQL or model outputs. This checks if the new evidence exists,
NOT whether it improves Qwen SQL exact execution match.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from hashlib import sha256
import json
from pathlib import Path

from evaluation.physical_value_grounding import candidate_literals,physical_value_columns
from scripts.prepare_birdtrain_qwen9b_sft_source import fetch_database,fetch_manifest

VAL_SHA="cc112305bc9de33a03cea704f2edecfdfb45e62cc953ae498cea27fbed93b506"


def source_audit(path:Path,work:Path)->dict:
    data=path.read_bytes()
    if sha256(data).hexdigest()!=VAL_SHA:
        raise ValueError("Publisher original VAL64 source SHA mismatch")
    rows=[json.loads(line) for line in data.decode().splitlines() if line.strip()]
    if len(rows)!=64 or len({r["source_index"] for r in rows})!=64:
        raise ValueError("Publisher original VAL64 missing source case")
    groups=defaultdict(list)
    for r in rows:groups[r["source_db_id"]].append(r)
    if len(groups)!=4:raise ValueError("Expected four original validation databases")
    manifest=fetch_manifest()
    records=[]
    for db,rs in sorted(groups.items()):
        source_db=fetch_database(db,manifest[db],work)
        for row in rs:
            user=row["messages"][1]["content"]
            if not user.startswith("Question: ") or "\nSchema context:\n" not in user:
                raise ValueError("Frozen original question prompt changed")
            question=user[10:].split("\nSchema context:\n",1)[0]
            mentions=candidate_literals(question)
            evidence=physical_value_columns(source_db,question)
            records.append({
                "source_index":row["source_index"],
                "db":db,
                "candidate_mentions_n":len(mentions),
                "physical_match_n":len(evidence),
                "matched_literals":sorted({x["value"] for x in evidence}),
                "candidate_value_columns":evidence,
                "no_gold_sql_read":True,
            })
    if len(records)!=64:
        raise ValueError("Publisher 64 source audit incomplete")
    c=Counter(x["db"] for x in records if x["physical_match_n"])
    return {
        "status":"SOURCE_ONLY_VALUE_LINKER_FEASIBILITY_NOT_MODEL_GAIN",
        "validation_sha256":VAL_SHA,
        "questions":64,"physical_databases":4,
        "with_candidate_mentions":sum(x["candidate_mentions_n"]>0 for x in records),
        "with_physical_value_matches":sum(x["physical_match_n"]>0 for x in records),
        "by_db_matched":dict(c),
        "exact_value_column_links":sum(x["physical_match_n"] for x in records),
        "model_calls":0,"no_reference_sql_access":True,
        "no_model_predictions_accessed":True,"no_accuracy_claim":True,
        "cases":records,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--validation",type=Path,required=True)
    p.add_argument("--db-work",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    result=source_audit(args.validation,args.db_work)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+"\n")
    print("SOURCE_PHYSICAL_VALUE_GROUNDING_FEASIBILITY",
          json.dumps({k:v for k,v in result.items() if k!="cases"},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
