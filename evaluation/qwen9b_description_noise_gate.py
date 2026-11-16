"""Gold-blind feasibility of dropping irrelevant BIRD column descriptions.

Original Qwen3.5-9B stays unchanged. Preserve EXACT physical table, column and
FK lines produced by original selector; change only the optional unstructured
column-description tail. One deterministic question-relevance filter is frozen
BEFORE observing any per-case SQL correctness.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from evaluation.qwen_schema_grounding_pair import (
    MODEL, MODEL_DIGEST, SOURCE_SHA, checked_source, case_id, legacy_tokens,
)
from evaluation.run_bird import _resolve_database
from evaluation.schema_context import select_schema_context
from semantic_sql.catalog import DatabaseCatalog

PROTOCOL="qwen9b-goldblind-description-relevance-coverage-original-bird100-v1"
MAX_DESCRIPTIONS=12
MAX_PER_TABLE=4


def prune_descriptions(context: str, question: str):
    """Keep all physical schema and FK facts verbatim; no model or labels."""
    head, marker, tail=context.partition("\nColumn descriptions:\n")
    if not marker:
        return context,{"original_descriptions":0,"retained_descriptions":0}
    rows=tail.split("\n")
    query_tokens=legacy_tokens(question)
    choices=[]
    for index,line in enumerate(rows):
        if not line.startswith("- ") or ": " not in line:
            raise ValueError("unknown physical column-description format: refuse silent drop")
        pair,_,description=line[2:].partition(": ")
        if "." not in pair:
            raise ValueError("unqualified physical description")
        table,column=pair.rsplit(".",1)
        if not table or not column:
            raise ValueError("invalid physical column identity")
        score=3*len(query_tokens & legacy_tokens(table+" "+column))
        score+=2*len(query_tokens & legacy_tokens(description))
        choices.append((index,line,table,score))
    winners=[]
    count=Counter()
    for index,line,table,score in sorted(choices,key=lambda x:(-x[3],x[0])):
        if score<=0:continue
        if len(winners)>=MAX_DESCRIPTIONS:break
        if count[table]>=MAX_PER_TABLE:continue
        winners.append(index)
        count[table]+=1
    kept=[line for index,line in enumerate(rows) if index in set(winners)]
    revised=head+("\nColumn descriptions:\n"+"\n".join(kept) if kept else "")
    if revised.split("\nColumn descriptions:")[0]!=head:
        raise AssertionError("physical schema facts changed")
    return revised,{
        "original_descriptions":len(rows),"retained_descriptions":len(kept),
    }


def gate(source:Path,root:Path,reference:Path):
    data=checked_source(source)
    ledger=json.loads(reference.read_text())
    if (ledger.get("status")!="complete" or ledger.get("sample_size")!=100
        or len(ledger.get("cases",[]))!=100):
        raise ValueError("original frozen BIRD100 identities missing")
    wanted={str(row["case_id"]) for row in ledger["cases"]}
    if len(wanted)!=100:
        raise ValueError("original BIRD100 contains duplicate IDs")
    indexes=[(case_id(row,i),row) for i,row in enumerate(data) if case_id(row,i) in wanted]
    if len(indexes)!=100:
        raise ValueError("original BIRD100 not fully matched to pinned BIRD source")
    original_by_db={}
    records=[]
    for cid,row in indexes:
        dbid=str(row["db_id"])
        if dbid not in original_by_db:
            database=_resolve_database(root,dbid)
            original_by_db[dbid]=DatabaseCatalog.from_sqlite(database)
        question=str(row["question"])
        original,meta=select_schema_context(original_by_db[dbid],root,dbid,question,tokenizer=legacy_tokens)
        shorter,diagnostic=prune_descriptions(original,question)
        changed=original!=shorter
        if len(shorter)>len(original):
            raise AssertionError("relevance prune expanded context")
        rec={
            "case_id":cid,"database_id":dbid,
            "changed":changed,
            "physical_schema_unchanged":original.split("\nColumn descriptions:")[0]==shorter.split("\nColumn descriptions:")[0],
            "original_context_sha256":meta["schema_context_sha256"],
            "pruned_context_sha256":sha256(shorter.encode()).hexdigest(),
            "original_chars":len(original),"revised_chars":len(shorter),
            **diagnostic
        }
        records.append(rec)
    affected=[x for x in records if x["changed"]]
    affected_dbs={x["database_id"] for x in affected}
    retained=sum(x["retained_descriptions"] for x in affected)
    total=sum(x["original_descriptions"] for x in affected)
    result={
        "protocol":PROTOCOL,"source_sha256":SOURCE_SHA,
        "original100_matched":len(records),"model":MODEL,"model_digest":MODEL_DIGEST,
        "no_sql_training":True,"model_calls":0,"gold_answer_accessed":False,
        "original100_affected":len(affected),
        "affected_physical_dbs":len(affected_dbs),
        "description_rows_before":total,"description_rows_after":retained,
        "affected_context_char_reduction":sum(x["original_chars"]-x["revised_chars"] for x in affected),
        "all_physical_schema_facts_preserved":all(x["physical_schema_unchanged"] for x in records),
        "prospective_inference_worthwhile":len(affected)>=15 and len(affected_dbs)>=6,
        "original_BIRD100_semantic_EX_unchanged":39,
        "cases":records
    }
    if not result["all_physical_schema_facts_preserved"]:
        raise AssertionError("lost a table, physical column or FK")
    print("QWEN9B_DESCRIPTION_RELEVANCE_SOURCE_FEASIBILITY",
          json.dumps({k:v for k,v in result.items() if k!="cases"},sort_keys=True),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--reference",type=Path,default=Path("runs/bird-paired-dev-v5/paired.json"))
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError("don't overwrite original coverage artifact")
    result=gate(args.source,args.root,args.reference)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")


if __name__=="__main__":
    main()
