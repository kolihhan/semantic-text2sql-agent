"""Physical CPU-only check of original BIRD TRAIN validation64 FK reference accuracy.

No model calls, no use of gold answers or BIRD DEV. Verifies SHA-pinned
publisher SQLite, compares original vs corrected input field by field.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

from evaluation.implicit_fk_grounding import resolve_implicit_fk_targets
from evaluation.schema_context import select_schema_context
from scripts.prepare_birdtrain_qwen9b_sft_source import fetch_database,fetch_manifest
from semantic_sql.catalog import DatabaseCatalog

VALIDATION_SHA="cc112305bc9de33a03cea704f2edecfdfb45e62cc953ae498cea27fbed93b506"


def audit(validation: Path, work: Path) -> dict:
    data=validation.read_bytes()
    if sha256(data).hexdigest()!=VALIDATION_SHA:
        raise ValueError("Original source validation64 SHA mismatch")
    rows=[json.loads(s) for s in data.decode().splitlines() if s.strip()]
    if len(rows)!=64 or len({r["source_index"] for r in rows})!=64:
        raise ValueError("Expected 64 unique original untouched tasks")
    ids={r["source_db_id"] for r in rows}
    if len(ids)!=4:raise ValueError("Expected 4 physical source DBs")
    manifest=fetch_manifest()
    by_db=defaultdict(list)
    for r in rows:by_db[r["source_db_id"]].append(r)
    summaries={}
    for dbid,samples in sorted(by_db.items()):
        db=fetch_database(dbid,manifest[dbid],work)
        cat=DatabaseCatalog.from_sqlite(db)
        fixed,resolution=resolve_implicit_fk_targets(cat)
        if fixed.tables!=cat.tables:raise RuntimeError("Altered physical columns")
        old_total=0;new_total=0;changed=0
        before_lines=0;after_lines=0
        for row in samples:
            query=row["messages"][1]["content"].split("\nSchema context:\n",1)[0][10:]
            original,_=select_schema_context(cat,work,dbid,query)
            corrected,_=select_schema_context(fixed,work,dbid,query)
            old_prompt=row["messages"][1]["content"]
            old_context=old_prompt.split("\nSchema context:\n",1)[1].split("\nExternal evidence:\n",1)[0]
            if original!=old_context:
                raise RuntimeError("Original frozen model input doesn't match current selector")
            old_non_fk=[l for l in original.splitlines() if not l.startswith("FK ")]
            new_non_fk=[l for l in corrected.splitlines() if not l.startswith("FK ")]
            if old_non_fk!=new_non_fk:
                raise RuntimeError("Non-FK schema context changed; not a single-variable treatment")
            before_lines+=sum(l.startswith("FK ") for l in original.splitlines())
            after_lines+=sum(l.startswith("FK ") for l in corrected.splitlines())
            old_total+=original.count(".None")
            new_total+=corrected.count(".None")
            changed+=int(original!=corrected)
        summaries[dbid]={
            "tasks":len(samples),
            "catalog_implicit_missing":resolution.original_missing,
            "inferred_pk_target":resolution.inferred,
            "unresolved":resolution.unresolved,
            "explicit_fk":resolution.original_explicit,
            "fk_lines_old":before_lines,"fk_lines_new":after_lines,
            "old_none_occurrences":old_total,
            "fixed_none_occurrences":new_total,
            "contexts_changed":changed,
        }
    return {
        "protocol":"original-birdtrain-source-fk-target-structural-audit-v1",
        "status":"SOURCE_SCHEMA_DIAGNOSTIC_ONLY",
        "validation_sha256":VALIDATION_SHA,
        "n":64,"physical_dbs":4,
        "original_none_occurrences":sum(r["old_none_occurrences"] for r in summaries.values()),
        "remaining_none_occurrences":sum(r["fixed_none_occurrences"] for r in summaries.values()),
        "modified_question_count":sum(r["contexts_changed"] for r in summaries.values()),
        "source_only_no_answers_used":True,
        "no_model_calls":True,
        "no_accuracy_claim":True,
        "original_pipeline_unchanged":True,
        "databases":summaries,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--validation",type=Path,required=True)
    p.add_argument("--work",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=audit(a.validation,a.work)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("SQL_FK_SOURCE_AUDIT",json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
