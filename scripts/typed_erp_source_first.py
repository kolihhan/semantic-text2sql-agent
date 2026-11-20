"""Source-first, SQL-blind candidate eligibility for BIRD TRAIN (CPU only).

NO publisher SQL field, original model SQL, execution labels, DEV/HOLDOUT.
The original publisher TRAIN JSON contains SQL, but this tool projects ONLY
question/db_id/evidence after SHA verification and never accesses row["SQL"].
Labels are automatic CANDIDATES, not semantic ground truth.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3

from scripts.prepare_birdtrain_qwen9b_sft_source import (
    TRAIN_SHA, fetch_manifest, fetch_database, load_checked, DB_MAX_BYTES,
)
from scripts.freeze_birdtrain_residual_db_holdout import (
    PREVIOUS_SOURCE_SPECS, historical_source_ledger,
)
from scripts.train48_original_qwen_unit_gate_source_then_score import source_rows

PROTOCOL = "BIRD-TRAIN-TYPED-ERP-SOURCE-FIRST-CANDIDATES-V1"
SEED = "typed-erp-source-first-uniform-12db-20q-20261120"
SEALED = frozenset(("simpson_episodes", "works_cycles"))
N_DATABASES = 12
N_PER_DATABASE = 20

RATIO = re.compile(r"\b(percent(?:age)?|ratio|proportion|fraction|share of|rate of|out of every|divided by)\b", re.I)
COUNT = re.compile(r"\b(how many|number of|count of|total number of|how many different)\b", re.I)
RELATION = re.compile(
    r"\b(both|either|neither|without|which have|who have|whose|belong(?:s|ing)? to|"
    r"associated with|common to|shared by|in all|exist(?:s)? in|never|none of|"
    r"at least one|all of the|intersection|union of)\b", re.I,
)
TARGET = re.compile(r"\b(?:how many|number of|count of|total number of)\s+"
                    r"(?:(?:different|distinct|unique|individual)\s+)?"
                    r"([a-z][a-z0-9 _-]{0,64}?)(?:\s+(?:are|were|have|had|did|do|does|"
                    r"with|in|from|for|among|per|that|who|whose|which|by|can)\b|[?.,]|$)", re.I)
WORDS = re.compile(r"[a-z0-9]+")
STOP = frozenset(("the","a","an","of","on","and","or","is","are","to","in","by",
                  "from","with","how","many","number","count","total","for",
                  "distinct","unique","different","records","entries","data","info"))

def digest(x: bytes) -> str:
    return sha256(x).hexdigest()

def norm_question(s: str) -> str:
    return " ".join(s.lower().split())

def tokens(s: str) -> set[str]:
    out = set()
    for word in WORDS.findall(s.lower().replace("_", " ")):
        if word in STOP: continue
        word = word[:-3] + "y" if word.endswith("ies") else (
            word[:-1] if word.endswith("s") and not word.endswith("ss") else word)
        if len(word)>1: out.add(word)
    return out

def category(question: str) -> str:
    q=question.strip()
    if not q or len(tokens(q))<2:
        return "UNKNOWN"
    if RATIO.search(q):
        return "RATIO_PERCENTAGE"
    if COUNT.search(q):
        return "ENTITY_COUNT"
    if RELATION.search(q):
        return "RELATION_SET"
    return "OTHER"

def schema_for(path: Path) -> dict:
    """Read only declared SQLite schema; never select application values."""
    result={"tables":{}, "foreign_keys":[]}
    with sqlite3.connect(path.as_uri()+"?mode=ro",uri=True) as con:
        con.execute("PRAGMA query_only=ON")
        for (name,) in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ):
            if not isinstance(name,str) or len(name)>200: raise ValueError("invalid schema table")
            quoted='"'+name.replace('"','""')+'"'
            info=con.execute("PRAGMA table_info("+quoted+")").fetchall()
            pk=[r[1] for r in sorted(info,key=lambda r:r[5]) if r[5]]
            result["tables"][name]={"pk":pk,"columns":[r[1] for r in info]}
            for r in con.execute("PRAGMA foreign_key_list("+quoted+")"):
                result["foreign_keys"].append({
                    "child":name,"column":r[3],"parent":r[2],"parent_column":r[4],
                    "fk_id":r[0],"seq":r[1],
                })
    return result

def match_tables(phrase: str, schema: dict) -> list[str]:
    ts=tokens(phrase)
    if not ts: return []
    return sorted(t for t in schema["tables"] if tokens(t) and tokens(t)<=ts)

def type_source(row: dict, schema: dict) -> dict:
    """Never infer a unique entity or relation from mere EX labels."""
    question=row["question"]
    kind=category(question)
    mentions=match_tables(question,schema)
    target=None
    reason="NO_UNAMBIGUOUS_TARGET"
    if kind=="ENTITY_COUNT":
        m=TARGET.search(question)
        if m:
            candidates=match_tables(m.group(1),schema)
            if len(candidates)==1:
                target=candidates[0]
                reason="SOURCE_TARGET_EXACT_SCHEMA_NAME"
            else:
                reason="TARGET_TABLE_AMBIGUOUS_OR_UNMAPPED"
    fk=[f for f in schema["foreign_keys"]
        if f["child"] in mentions and f["parent"] in mentions]
    typed_relation=(fk[0] if len(fk)==1 else None)
    if kind=="RELATION_SET" and len(fk)!=1:
        reason="RELATION_NO_UNIQUE_DECLARED_FK"
    elif kind=="RELATION_SET":
        reason="DECLARED_FK_BETWEEN_MENTIONED_TABLES"
    counting_unit=None
    if target:
        pk=schema["tables"][target]["pk"]
        if len(pk)==1:
            counting_unit={"table":target,"pk":pk[0],
                           "explicit_unique_requested":bool(re.search(r"\b(unique|distinct|different|individual)\b",question,re.I))}
        else:
            reason="TARGET_HAS_NO_SINGLE_DECLARED_PK"
    # Ratio denominator/population cannot be assumed from column/table names.
    population=None
    numerator=None
    denominator=None
    if kind=="RATIO_PERCENTAGE":
        reason="NUMERATOR_DENOMINATOR_SCOPE_NOT_PROVEN"
    status=("GROUNDED_CANDIDATE" if
            (kind=="ENTITY_COUNT" and counting_unit) or
            (kind=="RELATION_SET" and typed_relation)
            else "ABSTAIN" if kind in ("ENTITY_COUNT","RATIO_PERCENTAGE","RELATION_SET","UNKNOWN")
            else "NOT_APPLICABLE")
    return {
        "candidate_label":kind,"candidate_only_not_ground_truth":True,
        "status":status,"abstain_or_scope_reason":reason if status=="ABSTAIN" else None,
        "target_entity":target,
        "relation":typed_relation,
        "population":population,"counting_unit":counting_unit,
        "numerator":numerator,"denominator":denominator,
        "mentioned_schema_tables":mentions,
        "schema_evidence":{"table_primary_keys":{t:schema["tables"][t]["pk"] for t in mentions},
                           "declared_foreign_keys":fk},
    }

def historical_ids(paths: dict[str, Path]) -> tuple[set[int],set[str],dict]:
    used_dbs,ledger=historical_source_ledger(paths)
    ids=set()
    for label in PREVIOUS_SOURCE_SPECS:
        for line in paths[label].read_text(encoding="utf-8").splitlines():
            if line.strip(): ids.add(json.loads(line)["source_index"])
    assert len(ids)==ledger["previously_used_question_ids"]
    return ids,used_dbs,ledger

def freeze(train_path: Path, prior: dict[str,Path], old_source: Path,
           old_freeze: Path, dbcache: Path, out: Path, git_sha: str) -> dict:
    if out.exists(): raise FileExistsError("source freeze must be immutable")
    if not re.fullmatch(r"[0-9a-f]{40}",git_sha): raise ValueError("require exact commit SHA")
    # Protect both unexposed DBs BEFORE accessing their question text.
    used_ids,exposed,ledger=historical_ids(prior)
    if len(exposed)!=49 or len(exposed|SEALED)!=51 or exposed&SEALED:
        raise ValueError("source exposure roster drift")
    old,old_meta=source_rows(old_source,old_freeze)  # question-only historic source
    old_ids={r["source_index"] for r in old}
    if len(old_ids)!=48 or set(old_meta["dbs"]) - exposed:
        raise ValueError("TRAIN48 overlap/exposure uncertainty")
    train=load_checked(train_path,TRAIN_SHA)  # NEVER index the publisher SQL field
    manifest=fetch_manifest()
    if set(manifest)!=exposed|SEALED:
        raise ValueError("pinned 51-database universe drift")
    excluded=used_ids|old_ids
    by_db=defaultdict(list)
    seen_text=set()
    eligible_types=Counter()
    for ix,raw in enumerate(train):
        db=raw.get("db_id")
        if db not in exposed or ix in excluded:
            continue
        q=raw.get("question")
        if not isinstance(q,str) or not q.strip(): continue
        norm=norm_question(q)
        if norm in seen_text: continue
        seen_text.add(norm)
        evidence=raw.get("evidence") or ""
        if not isinstance(evidence,str): raise ValueError("invalid source evidence")
        row={"source_index":ix,"db_id":db,"question":q,"evidence":evidence,
             "question_sha256":digest(q.encode()),"evidence_sha256":digest(evidence.encode())}
        by_db[db].append(row)
        eligible_types[category(q)]+=1
    # Uniform SHA sampling, independent of question label or model output.
    dbs=sorted((d for d in exposed if d in by_db and len(by_db[d])>=N_PER_DATABASE
                and 0<manifest[d]["size"]<=DB_MAX_BYTES),
               key=lambda d:digest((SEED+":db:"+d).encode()))[:N_DATABASES]
    if len(dbs)!=N_DATABASES:raise ValueError("insufficient source DB clusters")
    old_dbs={r["db_id"] for r in old}
    if not old_dbs <= exposed:raise ValueError("historical source used sealed DB")
    schemas={}
    for db in sorted(set(dbs)|old_dbs):
        physical=fetch_database(db,manifest[db],dbcache)
        schemas[db]=schema_for(physical)
    frozen=[]
    for db in dbs:
        selected=sorted(by_db[db],key=lambda r:digest(
            (SEED+":case:"+str(r["source_index"])).encode()))[:N_PER_DATABASE]
        for row in selected:
            frozen.append({**row,**type_source(row,schemas[db])})
    historical=[{**row,**type_source(row,schemas[row["db_id"]])} for row in old]
    all_rows=frozen+historical
    if len({r["source_index"] for r in all_rows})!=len(all_rows):
        raise ValueError("source row reused across frozen/historical pools")
    assert all(r["db_id"] not in SEALED and "SQL" not in r and "gold" not in r
               for r in all_rows)
    source_code_sha=digest(Path(__file__).read_bytes())
    result={
        "protocol":PROTOCOL,"status":"FROZEN_BEFORE_MODEL_SQL_OR_SCORE",
        "source_selection_seed":SEED,"git_commit_sha":git_sha,
        "classifier_code_sha256":source_code_sha,
        "original_train_sha256":TRAIN_SHA,"old48_source_sha256":digest(old_source.read_bytes()),
        "historical_exposure_sha256":{n:PREVIOUS_SOURCE_SPECS[n][0] for n in PREVIOUS_SOURCE_SPECS},
        "source_universe_db_count":51,"historically_exposed_db_count":49,
        "sealed_untouched_db_ids":sorted(SEALED),
        "known_prior_question_ids":len(used_ids),
        "additional_historical_train48_ids":len(old_ids-used_ids),
        "remaining_source_question_count":sum(map(len,by_db.values())),
        "all_eligible_question_only_candidate_types":dict(sorted(eligible_types.items())),
        "source_db_count":len(dbs),"per_db":N_PER_DATABASE,
        "selected_database_ids":sorted(dbs),
        "source_candidates":sorted(frozen,key=lambda x:x["source_index"]),
        "original_Qwen48_source_only_controls":sorted(historical,key=lambda x:x["source_index"]),
        "source_candidate_type_counts":dict(sorted(Counter(x["candidate_label"] for x in frozen).items())),
        "source_candidate_status_counts":dict(sorted(Counter(x["status"] for x in frozen).items())),
        "candidate_labels_not_ground_truth":True,
        "source_frozen_no_model_or_score_or_gold_SQL_access":True,
        "no_Qwen_calls":True,"no_training":True,"no_DEV500_or_HOLDOUT300":True,
        "limitations":"Development cohort shares historically exposed physical DBs; independent clean test still sealed. Automatic labels require external validation.",
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False)+"\n",encoding="utf-8")
    return result

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--train",required=True,type=Path)
    p.add_argument("--prior",nargs=10,required=True,type=Path)
    p.add_argument("--old-source",required=True,type=Path)
    p.add_argument("--old-freeze",required=True,type=Path)
    p.add_argument("--db-cache",required=True,type=Path)
    p.add_argument("--git-sha",required=True)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    r=freeze(a.train,dict(zip(PREVIOUS_SOURCE_SPECS,a.prior)),a.old_source,
             a.old_freeze,a.db_cache,a.out,a.git_sha)
    print(json.dumps({k:v for k,v in r.items()
                      if k not in ("source_candidates","original_Qwen48_source_only_controls",
                                    "historical_exposure_sha256")},sort_keys=True))

if __name__=="__main__":
    main()
