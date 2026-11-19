"""Two-stage extra DB-split count-grain hypothesis check for 3 predeclared cases.

Phase source: original physical SQL + question + PK/FK, NO publisher SQL.
Phase score: consumes immutable source artifact before loading original source
SQL; compares actual physical result sets only, no Qwen call.
These are previously public/seen DEV cases, not a new holdout or Guarded A/B.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
from scripts.postblind_bird_dev500_count_probe5 import run_query,locate,SOURCE_SHA

TARGETS={"463":"card_games","593":"codebase_community",
         "1525":"debit_card_specializing"}


def digest(src:bytes|str):
    if isinstance(src,str):src=src.encode()
    return hashlib.sha256(src).hexdigest()


def proposed(cid:str,sql:str)->str:
    if cid=="463":
        assert "JOIN set_translations" in sql and "COUNT(*)" in sql
        return sql.replace("COUNT(*)","COUNT(DISTINCT set_translations.id)",1)
    if cid=="593":
        assert "FROM badges b" in sql and "JOIN users u" in sql
        return sql.replace("COUNT(*)","COUNT(DISTINCT u.Id)",1)
    if cid=="1525":
        split="FROM customers c"
        assert sql.count(split)==1
        before,rest=sql.split(split)
        assert before.strip().startswith("SELECT CAST(SUM(CASE WHEN c.Currency = 'EUR'")
        new=("SELECT CAST(COUNT(DISTINCT CASE WHEN c.Currency = 'EUR' "
             "THEN c.CustomerID END) AS FLOAT) * 100.0 "
             "/ COUNT(DISTINCT c.CustomerID) ")
        return new+split+rest
    raise ValueError(cid)


def source_first(a):
    x=json.loads(a.original_81_source_first.read_text())
    assert x["remaining_count"]==81 and x["database_count"]==11
    assert x["no_gold_reference_consulted"]
    byid={r["case_id"]:r for r in x["source_first_cases"]}
    assert all(cid in byid for cid in TARGETS)
    rows=[]
    for cid,db in sorted(TARGETS.items(),key=lambda x:int(x[0])):
        row=byid[cid]
        assert row["db_id"]==db and row["publisher_reference_sql_read"] is False
        candidate=row["first_original_qwen_sql"]
        modified=proposed(cid,candidate)
        assert modified != candidate
        physical=locate(a.db_root,db)
        base=run_query(physical,candidate)
        treated=run_query(physical,modified)
        rows.append({
            "case_id":cid,"db":db,"question":row["question"],
            "question_sha":row["question_sha256"],
            "question_intent":"count explicitly named entity before joined row",
            "original_first_generated_SQL":candidate,
            "source_only_fixed_grain_SQL":modified,
            "original_SQL_sha":digest(candidate),"new_SQL_sha":digest(modified),
            "base":base,"changed":treated,
            "physical_result_changes":(
                base["status"]=="ok" and treated["status"]=="ok"
                and base["set_sha256"]!=treated["set_sha256"]),
            "publisher_reference_seen":False
        })
    return {
        "protocol":"QWEN35-DEV500-EXTRA3-COUNT-GRAIN-SOURCE-FIRST-V1",
        "n":3,"db_count":3,
        "cases":rows,
        "upstream_source_first_run":"https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44555827205",
        "original_model_fixed":"Qwen3.5-9B",
        "new_model_calls":0,"gold_reference_not_seen":True,
        "no_training":True,"no_Guarded_change":True,
        "HOLDOUT300_untouched":True,
        "scope":"Original 3 public exposed DEV K8-zero-oracle cases. Non-adaptive "
                "source-first original SQL → deterministic count-entity correction. "
                "No claim of reference correctness until separate postblind scorer."
    }


def postblind(a):
    sourcebytes=a.official_dev.read_bytes()
    assert digest(sourcebytes)==SOURCE_SHA
    frozen=json.loads(a.extra_source_first.read_text())
    assert frozen["protocol"]=="QWEN35-DEV500-EXTRA3-COUNT-GRAIN-SOURCE-FIRST-V1"
    assert frozen["gold_reference_not_seen"] and frozen["n"]==3
    assert set(z["case_id"] for z in frozen["cases"])==set(TARGETS)
    byid={str(x["question_id"]):x for x in json.loads(sourcebytes)}
    scores=[]
    for row in frozen["cases"]:
        cid=row["case_id"]
        source=byid[cid]
        assert source["db_id"]==TARGETS[cid]
        assert source["question"]==row["question"]
        assert digest(source["question"])==row["question_sha"]
        assert digest(row["original_first_generated_SQL"])==row["original_SQL_sha"]
        assert digest(row["source_only_fixed_grain_SQL"])==row["new_SQL_sha"]
        db=locate(a.db_root,row["db"])
        gold=run_query(db,source["SQL"])
        original=run_query(db,row["original_first_generated_SQL"])
        changed=run_query(db,row["source_only_fixed_grain_SQL"])
        assert original==row["base"] and changed==row["changed"]
        b=(gold["set_sha256"]==original["set_sha256"]
           if gold["status"]=="ok" and original["status"]=="ok" else None)
        t=(gold["set_sha256"]==changed["set_sha256"]
           if gold["status"]=="ok" and changed["status"]=="ok" else None)
        scores.append({
            "case_id":cid,"db":row["db"],
            "scorable":b is not None and t is not None,
            "original_match":b,"changed_match":t,
            "rescue":b is False and t is True,
            "regression":b is True and t is False,
            "physical_original_changed":row["physical_result_changes"],
            "source_immutable":True,
        })
    valid=[x for x in scores if x["scorable"]]
    return {
        "protocol":"QWEN35-DEV500-EXTRA3-COUNT-GRAIN-POSTBLIND-V1",
        "predeclared_source_first":True,"source_first_3_db":True,
        "n":3,"paired_scorable":len(valid),
        "original_match":sum(x["original_match"] for x in valid),
        "treatment_match":sum(x["changed_match"] for x in valid),
        "rescues":sum(x["rescue"] for x in valid),
        "regressions":sum(x["regression"] for x in valid),
        "case_results":scores,"new_model_calls":0,
        "not_full_Guarded_BIRD_EX":True,"HOLDOUT300_untouched":True,
        "do_not_promote_without_fresh_frozen_matched_Guarded_ab":True
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=["source","score"])
    p.add_argument("--original-81-source-first",type=Path)
    p.add_argument("--extra-source-first",type=Path)
    p.add_argument("--official-dev",type=Path)
    p.add_argument("--db-root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError("immutable output already exists")
    if a.phase=="source":
        assert a.original_81_source_first and not a.official_dev
        x=source_first(a)
    else:
        assert a.extra_source_first and a.official_dev
        x=postblind(a)
    a.out.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")
    print("QWEN35_DEV500_GRAIN_"+a.phase.upper(),json.dumps({
        "n":x["n"],"db":x.get("db_count",3),
        "result_changes":[[r["case_id"],r["physical_result_changes"]]
                          for r in x["cases"]] if a.phase=="source" else
                         [[r["case_id"],r["original_match"],r["changed_match"]]
                          for r in x["case_results"]],
        "model_calls":0
    },sort_keys=True))


if __name__=="__main__":
    main()
