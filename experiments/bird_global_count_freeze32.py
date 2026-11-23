"""Pre-Qwen BIRD TRAIN source-only 32-case global-count/aggregation selection.

New original question identities excluded against the recent BIRD19 and ALLof11
pilots. This is *not* physical-database-disjoint from all older BIRD experiments;
last two protected TRAIN physical DBs remain untouched. Never inspect SQL,
publisher outputs, model outcomes or heldout labels to choose questions.
"""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re

from scripts.prepare_birdtrain_qwen9b_sft_source import TRAIN_SHA,load_checked,fetch_manifest

PROTOCOL="BIRD-GLOBAL-CARDINALITY-TRAIN32-SOURCE-FREEZE-V1"
SEED="bird-global-cardinality-prospective-question-only-20261123"
PROTECTED=frozenset(("simpson_episodes","works_cycles"))
MAX_DB_BYTES=96*1024*1024
N=32
N_DB_MIN=12
_COUNT=re.compile(r"\b(?:how many|number of|total number)\b",re.I)
_AGG=re.compile(
    r"\b(?:total|sum|average|avg|greater than|less than|more than|"
    r"at least|at most|above|below|price|sold|cost|revenue|salary|"
    r"spent|threshold|exceed|over \d+)\b",re.I
)
_MULTI=re.compile(
    r"\b(?:for each|per (?:each|every|group|customer|department|year|"
    r"month|person)|breakdown|list (?:them|each)|indicate each|show each)\b",re.I
)


def prior_indices(paths:list[Path])->set[int]:
    if len(paths)!=2:
        raise ValueError("both actual prior gold-free frozen cohorts required")
    seen=set()
    allowed={"BIRD_COUNT_GRAIN_GOLD_FREE_QUESTION_ONLY":19,
             "BIRD_ALLOF_GOLD_FREE_QUESTION_ONLY":11}
    for path in paths:
        obj=json.loads(path.read_text(encoding="utf-8"))
        status=obj.get("status")
        if status not in allowed or obj.get("original_source_sha256")!=TRAIN_SHA:
            raise ValueError("prior source not original BIRD gold-free")
        records=obj.get("cases")
        if len(records)!=allowed[status] or any("SQL" in row for row in records):
            raise ValueError("prior cohort drift or leaked SQL")
        for row in records:
            if row["source_index"] in seen:
                raise ValueError("prior source cohort overlap")
            seen.add(row["source_index"])
    return seen


def freeze(train:Path,prior:list[Path],out:Path)->dict:
    rows=load_checked(train,TRAIN_SHA)
    excluded=prior_indices(prior)
    manifest=fetch_manifest()
    eligible=defaultdict(list)
    for ix,row in enumerate(rows):
        # No read of row['SQL'] or any publisher-correctness field.
        q=row.get("question")
        db=row.get("db_id")
        if (ix in excluded or not isinstance(q,str)
            or not isinstance(db,str) or db in PROTECTED or db not in manifest
            or manifest[db]["size"]>MAX_DB_BYTES
            or not _COUNT.search(q) or not _AGG.search(q)
            or _MULTI.search(q)):
            continue
        record={
            "source_index":ix,"db_id":db,
            "question_sha256":sha256(q.encode()).hexdigest(),
            "evidence_sha256":sha256(str(row.get("evidence") or "").encode()).hexdigest(),
        }
        eligible[db].append(record)
    for db in eligible:
        eligible[db].sort(key=lambda r:sha256(
            f"{SEED}:{r['source_index']}".encode()).hexdigest())
    ordered_db=sorted(eligible,key=lambda db:sha256(
        f"{SEED}:{db}".encode()).hexdigest())
    chosen=[];per_db=defaultdict(int)
    for round_id in (0,1,2):
        for db in ordered_db:
            if len(chosen)>=N:break
            if len(eligible[db])<=round_id:continue
            row=eligible[db][round_id]
            chosen.append(row)
            per_db[db]+=1
    if len(chosen)!=N or len(per_db)<N_DB_MIN:
        raise ValueError(
            f"BLOCK_MODEL: only {len(chosen)}/{N} source-only questions across {len(per_db)} DB"
        )
    obj={
        "protocol":PROTOCOL,
        "status":"SOURCE_ONLY_BIRD_TRAIN32_FROZEN_BEFORE_QWEN",
        "original_source_sha256":TRAIN_SHA,
        "seed":SEED,"n":N,"n_db":len(per_db),
        "all_eligible_question_count":sum(map(len,eligible.values())),
        "all_eligible_db_count":len(eligible),
        "excluded_previous_question_ids":len(excluded),
        "no_recent_BIRD11_or_BIRD19_question_overlap":True,
        "last_two_protected_TRAIN_DBS_unused":True,
        "BIRD_HOLDOUT300_unused":True,
        "publisher_SQL_never_read":True,
        "no_Qwen_inference":True,
        "historical_DB_exposure_not_independent":True,
        "selection":chosen,
    }
    if out.exists():raise FileExistsError("immutable cohort already frozen")
    out.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")
    return obj


def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--prior",type=Path,nargs=2,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    obj=freeze(a.train,a.prior,a.out)
    print("BIRD_GLOBAL32_PREMODEL_SOURCE_FREEZE",json.dumps(
        {k:v for k,v in obj.items() if k!="selection"},sort_keys=True),flush=True)

if __name__=="__main__":
    main()
