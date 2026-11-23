"""Pre-gold sealed retrospective BIRD TRAIN19 global cardinality intervention.

Original Qwen generation and original Guarded predictions are reused byte-for-
byte from Action 45100836322. No new LLM inference/training, no model change.
Candidate production sees question/schema/physical DB and original SQL, never
publisher SQL/correctness. Scoring happens only after proposal seal, and this
already-exposed sample is DIAGNOSTIC ONLY, not prospective evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from research_prototypes.grouped_global_cardinality import (
    _QUERY,_COUNT_REQUEST,find_global_cardinality_contradiction,
)
from scripts.prepare_birdtrain_qwen9b_sft_source import (
    TRAIN_SHA,fetch_database,fetch_manifest,load_checked,
)
from semantic_sql.execution import execute_readonly,SQLExecutionError
from semantic_sql.verifier import verify_sql_preflight

PROTOCOL="BIRD-GROUPED-SCALAR-COUNT-TRAIN19-RETRO-PRE-GOLD-V1"
QWEN_PROTOCOL="BIRD-QWEN9B-TRAIN19-COUNT-GRAIN-LANGGRAPH-PAIRED-V1"
DIGEST="56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
N=19
K=4


def _read_gold_free(source:Path,gen:list[Path]) -> tuple[dict[int,dict],list[dict]]:
    s=json.loads(source.read_text(encoding="utf-8"))
    if (s.get("status")!="BIRD_COUNT_GRAIN_GOLD_FREE_QUESTION_ONLY"
        or s.get("original_source_sha256")!=TRAIN_SHA
        or len(s.get("cases",[]))!=N or len(gen)!=K):
        raise ValueError("not frozen BIRD19 GOLD-FREE question sources")
    questions={r["source_index"]:r for r in s["cases"]}
    if len(questions)!=N or any(
        "SQL" in r or "sql" in r or "gold" in r or "answer" in r
        for r in questions.values()
    ):
        raise ValueError("gold-leakage source or duplicate question")
    rows=[];used_shards=set();seen=set()
    for path in gen:
        doc=json.loads(path.read_text(encoding="utf-8"))
        if (doc.get("protocol")!=QWEN_PROTOCOL
            or doc.get("phase")!="GOLD_BLIND_ORIGINAL_QWEN_PREDICTIONS_SEALED"
            or doc.get("no_gold_in_model_or_graph") is not True
            or doc.get("model_digest")!=DIGEST
            or doc.get("shard") in used_shards):
            raise ValueError("Qwen original SQL not sealed / wrong model / duplicate shard")
        used_shards.add(doc["shard"])
        for x in doc["cases"]:
            i=x["source_index"]
            if i not in questions or i in seen:
                raise ValueError("unknown/duplicate BIRD question")
            q=questions[i]
            if (q["question_sha256"]!=x["question_sha256"]
                or q["db_id"]!=x["db_id"]):
                raise ValueError("original question DB drift")
            seen.add(i);rows.append(x)
    if len(rows)!=N or used_shards!=set(range(K)):
        raise ValueError("incomplete original Qwen predictions")
    return questions,rows


def infer(source:Path,generated:list[Path],work:Path,out:Path) -> dict:
    questions,rows=_read_gold_free(source,generated)
    manifest=fetch_manifest()
    files={}
    frozen=[]
    for original in sorted(rows,key=lambda x:x["source_index"]):
        q=questions[original["source_index"]]
        sql=original["original_guarded_sql"]
        proposed=sql
        found=False
        witness_summary="NO_ACTIONABLE_GLOBAL_GROUPED_COUNT"
        if _QUERY.fullmatch(sql.strip()) is not None and _COUNT_REQUEST.search(q["question"]):
            db=q["db_id"]
            if db in ("simpson_episodes","works_cycles") or manifest[db]["size"]>96*1024*1024:
                raise ValueError("protected/out-of-budget original BIRD physical database")
            if db not in files:
                files[db]=fetch_database(db,manifest[db],work)
            witness=find_global_cardinality_contradiction(files[db],q["question"],sql)
            if witness is not None:
                check=verify_sql_preflight(files[db],witness.proposed_sql)
                if not check.ok:raise ValueError("proposed SQL failed SQLite read-only compiler")
                proposed=witness.proposed_sql
                found=True
                witness_summary="PHYSICAL_MULTIGROUP_UNARY_COUNT_VS_ONE_GLOBAL_SCALAR"
        frozen.append({
            "source_index":q["source_index"],"db_id":q["db_id"],
            "question_sha256":q["question_sha256"],
            "original_guarded_sql":sql,
            "proposal_sql":proposed,
            "changed":found,
            "witness_reason":witness_summary,
            "extra_Qwen_calls":0,
        })
        print("BIRD19_GROUPED_SCALAR_BLIND",q["source_index"],q["db_id"],
              "changed",found,flush=True)
    obj={
        "protocol":PROTOCOL,
        "status":"PRE_GOLD_GLOBAL_COUNT_PROPOSALS_SEALED",
        "base_protocol":QWEN_PROTOCOL,
        "fixed_model_digest":DIGEST,
        "source_sha256":TRAIN_SHA,
        "n":N,
        "changed_sql_n":sum(x["changed"] for x in frozen),
        "physical_db_download_n":len(files),
        "extra_Qwen_calls":0,
        "no_publisher_gold_in_selection":True,
        "known_exposed_and_scored_BIRD19":True,
        "not_independent":True,
        "HOLDOUT300_used":False,
        "protected_last_2_TRAIN_DBs_used":False,
        "cases":frozen,
    }
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False)+"\n")
    return obj


def _sql_result(db:Path,sql:str):
    if not sql.strip():return None
    try:
        r=execute_readonly(db,sql,max_rows=50000)
        if r.truncated:return None
        data=[tuple(row) for row in r.rows]
        return set(data),Counter(data)
    except (SQLExecutionError,TypeError):
        return None


def score(source:Path,blind:Path,scored:list[Path],work:Path,out:Path)->dict:
    source_rows=load_checked(source,TRAIN_SHA)
    d=json.loads(blind.read_text(encoding="utf-8"))
    if (d.get("protocol")!=PROTOCOL
        or d.get("status")!="PRE_GOLD_GLOBAL_COUNT_PROPOSALS_SEALED"
        or d.get("n")!=N or d.get("no_publisher_gold_in_selection") is not True
        or d.get("fixed_model_digest")!=DIGEST):
        raise ValueError("gold-free candidate seal invalid")
    if len(scored)!=K:raise ValueError("all four original score shards required")
    controls={}
    for file in scored:
        old=json.loads(file.read_text(encoding="utf-8"))
        if old.get("phase")!="POST_SEAL_PUBLISHER_BIRD_STRICT_SCORE":
            raise ValueError("original score not sealed")
        for row in old["cases"]:
            i=row["source_index"]
            if i in controls:raise ValueError("duplicated original score")
            controls[i]=row
    manifest=fetch_manifest();files={}
    rows=[]
    for row in d["cases"]:
        i=row["source_index"];dbid=row["db_id"];gold=source_rows[i]
        original=controls.get(i)
        if (not original or not original["scorable"] or original["db_id"]!=dbid
            or gold["db_id"]!=dbid
            or sha256(gold["question"].encode()).hexdigest()!=row["question_sha256"]):
            raise ValueError("original Qwen/score/source mismatch")
        base=bool(original["control_set"]);base_bag=bool(original["control_bag"])
        treatment=base;treatment_bag=base_bag
        if row["changed"]:
            if dbid not in files:
                if dbid in ("simpson_episodes","works_cycles"):
                    raise ValueError("protected DB in scored candidate")
                files[dbid]=fetch_database(dbid,manifest[dbid],work)
            ref=_sql_result(files[dbid],str(gold.get("SQL") or ""))
            revised=_sql_result(files[dbid],row["proposal_sql"])
            if ref is None:
                raise ValueError("original publisher BIRD reference unscorable")
            treatment=bool(revised and revised[0]==ref[0])
            treatment_bag=bool(revised and revised[1]==ref[1])
        rows.append({"source_index":i,"db_id":dbid,
                     "original_set":base,"treatment_set":treatment,
                     "original_bag":base_bag,"treatment_bag":treatment_bag,
                     "changed":row["changed"]})
    if len(rows)!=N or len({r["source_index"] for r in rows})!=N:
        raise ValueError("not all BIRD19 original source identities")
    stats=Counter()
    for r in rows:
        stats["control_set"]+=int(r["original_set"])
        stats["treatment_set"]+=int(r["treatment_set"])
        stats["control_bag"]+=int(r["original_bag"])
        stats["treatment_bag"]+=int(r["treatment_bag"])
        stats["rescues"]+=int(r["treatment_set"] and not r["original_set"])
        stats["regressions"]+=int(r["original_set"] and not r["treatment_set"])
    obj={
        "protocol":PROTOCOL,
        "status":"BIRD_TRAIN19_POSTHOC_GROUPED_GLOBAL_COUNT_STRICT_EX",
        "n":N,"database_count":13,
        "changed_sql_n":d["changed_sql_n"],
        "counts":dict(stats),
        "net_selected_set":stats["rescues"]-stats["regressions"],
        "new_Qwen_calls":0,
        "known_exposed_before_hypothesis":True,
        "promotion_allowed":False,
        "not_a_prospective_semantic_gain":True,
        "protected_BIRD_holdouts_unused":True,
        "next_gate":"If promising, freeze a disjoint source-only new BIRD cohort before inference and run original-Qwen paired Strict EX with right controls",
        "cases":rows,
    }
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")
    return obj


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("infer","score"))
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--generated",type=Path,nargs="+")
    p.add_argument("--blind",type=Path)
    p.add_argument("--scored",type=Path,nargs="+")
    p.add_argument("--work",type=Path,default=Path("physical"))
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    value=infer(a.source,a.generated,a.work,a.out) if a.mode=="infer" else score(a.source,a.blind,a.scored,a.work,a.out)
    print("BIRD19_GLOBAL_COUNT_REAL_PHYSICAL",json.dumps(
        {k:v for k,v in value.items() if k!="cases"},sort_keys=True),flush=True)

if __name__=="__main__":
    main()
