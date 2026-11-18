"""Freeze the LAST unexposed BIRD TRAIN physical database families, source-only.

BIRD TRAIN only, pinned original source and physical DB manifest. Prevent the
common error "new question" == "new database": replay all ten physically
published historical source cohorts from immutable SHA-matched Actions files.
If and ONLY if the physical DB roster is exactly the original 51, all prior
cohorts match, and their union is exactly 49, identify the two remaining DBs.

Never inspect source publisher SQL, use source EX labels, run an LLM, train a
model, or touch BIRD DEV500 / reserved HOLDOUT300. Save only shuffled
question identities + SHA, NOT question text, gold SQL or reference answers.
This is a protected source holdout, not a new gain or a valid large OOD sample:
two DB clusters alone cannot establish narrow confidence intervals.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from scripts.prepare_birdtrain_qwen9b_sft_source import (
    TRAIN_SHA, fetch_manifest, load_checked,
)

PROTOCOL = "BIRD-TRAIN-LAST-TWO-PHYSICAL-DATABASES-EXPOSURE-LEDGER-V1"
MODEL_DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
EXPECTED_TOTAL_DBS = 51
EXPECTED_USED_DBS = 49
EXPECTED_REMAINING_DBS = 2
MAX_PHYSICAL_DB_BYTES = 96 * 1024 * 1024
MAX_QUESTION_IDS_PER_DB = 32
QUESTION_SEED = "source-DB-level-OOD-protection-no-ex-outcomes-20261118"

# Label, immutable SHA, actual source row count, source file field name.
# The original "train" and "validation" are two files from ONE source artifact.
# The 49-family cumulative union includes the latest 2026-11-18 fanout16,
# even though its four selected DBs are not four globally new DB families.
PREVIOUS_SOURCE_SPECS = {
    "original_train242": ("b6851fb0f2b5027833aff45a0b5e01326359b245be1c812edc66986d0df1d402", 242),
    "original_validation64": ("cc112305bc9de33a03cea704f2edecfdfb45e62cc953ae498cea27fbed93b506", 64),
    "fk32": ("ee1b5035302d36a1241952bd2e5acd3dfc2518d1fba2581f557e04e179960535", 32),
    "value16": ("4de8471341b38d0fb00ded1f17f265d33b528ec78b9779eb94937914df196955", 16),
    "value16_replica": ("faa4f7ba1de9923cbf8e51a500727dfe4c76a08c3f6366735cf21bff152655b7", 16),
    "role8": ("3be2fc49ba3cff4171e20afcaf70f1fc698c8be401ac332a29fa1f48b781168f", 8),
    "preview8": ("85c3e3abc45d90f23418dbc14261111abf90feadfef2b155988b77b40a4fe0ca", 8),
    "column8": ("e7e822dbefadeb57dd2ba5a79553d38f5b5148a4fab33b9c0563b2e624ca6d74", 8),
    "grain16": ("ca68478128821b253dca9db600b101b9140fa645f2e38ed2e3493603db7efd8e", 16),
    "fanout16": ("62eb19b49d43f5e40a5fc2c79d095b9a6294caea781aeebf6b71cda57df5c04c", 16),
}
ARTIFACT_PROVENANCE = {
    "original_train242": (43888747753, 14234409041),
    "original_validation64": (43888747753, 14234409041),
    "fk32": (43904889296, 14245905732),
    "value16": (43920461939, 14253926660),
    "value16_replica": (43926576493, 14256077196),
    "role8": (44012242171, 14297082076),
    "preview8": (44040520571, 14309376574),
    "column8": (44042440721, 14310310770),
    "grain16": (44053157511, 14314153717),
    "fanout16": (44289030583, 14420113552),
}


def historical_source_ledger(paths: dict[str, Path]) -> tuple[set[str], dict]:
    if set(paths) != set(PREVIOUS_SOURCE_SPECS):
        raise ValueError("missing exact source history: cannot certify held-out DB")
    used: set[str] = set()
    counted: dict = {}
    seen_ids: set[int] = set()
    for label, (digest, expected_n) in PREVIOUS_SOURCE_SPECS.items():
        raw = paths[label].read_bytes()
        if sha256(raw).hexdigest() != digest:
            raise ValueError("immutable historical BIRD TRAIN source SHA drift: " + label)
        records = [json.loads(x) for x in raw.decode("utf-8").splitlines() if x.strip()]
        if len(records) != expected_n:
            raise ValueError("historical BIRD TRAIN row count drift: " + label)
        dbs = set()
        for r in records:
            db = r.get("source_db_id") or r.get("db_id")
            ix = r.get("source_index")
            if not isinstance(db, str) or not isinstance(ix, int):
                raise ValueError("no physical DB or original question index: " + label)
            dbs.add(db)
            seen_ids.add(ix)
        counted[label] = {
            "n_questions": len(records),
            "n_databases": len(dbs),
            "new_database_families": len(dbs-used),
            "source_sha256": digest,
            "source_run_id": ARTIFACT_PROVENANCE[label][0],
            "source_artifact_id": ARTIFACT_PROVENANCE[label][1],
            "physical_dbs": sorted(dbs),
        }
        used.update(dbs)
    return used, {"prior_sources": counted, "previously_used_question_ids":len(seen_ids)}


def original_question_heldout_indices(train: list[dict], db: str) -> tuple[list[dict], int]:
    choices = []
    for ix, row in enumerate(train):
        if row.get("db_id") != db:
            continue
        question = row.get("question")
        if not isinstance(question,str) or not question.strip():
            continue
        evidence = row.get("evidence") or ""
        if not isinstance(evidence,str):
            raise ValueError("original BIRD evidence format drift")
        choices.append({
            "source_index":ix,
            "source_db_id":db,
            "question_sha256":sha256(question.encode("utf-8")).hexdigest(),
            "evidence_sha256":sha256(evidence.encode("utf-8")).hexdigest(),
        })
    choices.sort(key=lambda r:sha256(
        f"{QUESTION_SEED}:{db}:{r['source_index']}".encode()
    ).hexdigest())
    return choices[:MAX_QUESTION_IDS_PER_DB],len(choices)


def make_report(train: list[dict], physical_manifest: dict[str,dict],
                used:set[str], ledger:dict) -> dict:
    all_dbs=set(physical_manifest)
    if len(all_dbs)!=EXPECTED_TOTAL_DBS:
        raise ValueError("the pinned original BIRD TRAIN must have EXACTLY 51 physical DBs")
    if not used<=all_dbs:
        raise ValueError("previously used source DB absent from pinned original TRAIN DB list")
    if len(used)!=EXPECTED_USED_DBS or len(all_dbs-used)!=EXPECTED_REMAINING_DBS:
        raise ValueError("historical DB roster or exposure count changed; fail closed")
    remaining=sorted(all_dbs-used)
    frozen=[]
    detail=[]
    for db in remaining:
        meta=physical_manifest[db]
        if not isinstance(meta.get("size"),int) or meta["size"]<=0:
            raise ValueError("missing pinned physical original DB file size")
        records,total=original_question_heldout_indices(train,db)
        if not records:
            raise ValueError("unseen physical database has no source questions")
        frozen.extend(records)
        detail.append({
            "db_id":db,"original_training_question_count":total,
            "manifest_size_bytes":meta["size"],
            "within_existing_96MiB_source_inference_limit":meta["size"]<=MAX_PHYSICAL_DB_BYTES,
            "frozen_question_count":len(records),
        })
    if len({(x["source_db_id"],x["source_index"]) for x in frozen})!=len(frozen):
        raise ValueError("duplicate frozen source question identity")
    if any(x["source_db_id"] in used for x in frozen):
        raise ValueError("previously exposed physical database entered final holdout")
    return {
        "status":"FROZEN_RESIDUAL_BIRD_TRAIN_DATABASE_HOLDOUT_QUESTION_IDENTITIES",
        "protocol":PROTOCOL,
        "original_train_sha256":TRAIN_SHA,
        "original_train_physical_database_count":len(all_dbs),
        "verified_historically_exposed_database_count":len(used),
        "verified_unexposed_database_count":len(remaining),
        "verified_exposed_database_ids":sorted(used),
        "unexposed_database_summaries":detail,
        "frozen_case_n":len(frozen),
        "frozen_question_identities":frozen,
        "prior_ledger":ledger,
        "fixed_base_model_digest":MODEL_DIGEST,
        "no_question_text_or_gold_sql_persisted":True,
        "no_gold_sql_used_to_select":True,
        "new_Qwen_inference_calls":0,
        "new_training_steps":0,
        "bird_DEV500_used":False,
        "bird_HOLDOUT300_used":False,
        "reserve_do_not_evaluate_until_mechanism_fixed":True,
        "generalization_boundary":(
            "Only two unseen physical DBs remain in this exact BIRD TRAIN pool. "
            "Question counts do not increase the number of independent DB clusters. "
            "Not an independent large-sample benchmark; no model metric. "
            "Even in fully unseen physical DBs, pretrained foundation-model "
            "exposure remains unknown. Freeze IDs without opening publisher SQL."
        ),
    }


def freeze(train_file: Path, old:dict[str,Path], out:Path)->dict:
    # Full original publisher file is checked by SHA. Critically, we do NOT
    # inspect SQL or reference outcomes, nor serialize natural-language questions.
    train=load_checked(train_file,TRAIN_SHA)
    used,ledger=historical_source_ledger(old)
    manifest=fetch_manifest()
    r=make_report(train,manifest,used,ledger)
    if out.exists():
        raise FileExistsError("will not overwrite frozen source holdout")
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(r,indent=2,sort_keys=True,ensure_ascii=False)+"\n")
    print("NEW_DATABASE_OOD_BIRD_TRAIN_SOURCE_FREEZE",json.dumps({
        k:v for k,v in r.items()
        if k not in {"frozen_question_identities","prior_ledger","verified_exposed_database_ids"}
    },sort_keys=True),flush=True)
    return r


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--prior",type=Path,nargs=10,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    names=list(PREVIOUS_SOURCE_SPECS)
    result=freeze(args.train,dict(zip(names,args.prior)),args.out)
    if result["verified_unexposed_database_count"]!=2:
        raise ValueError("cannot freeze anything except exactly two remaining DBs")


if __name__ == "__main__":
    main()
