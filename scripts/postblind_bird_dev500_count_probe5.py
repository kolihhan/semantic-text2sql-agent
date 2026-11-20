"""Post-blind publisher SQL check for PREDECLARED five physical COUNT probes.

Requires the already completed immutable source-FIRST artifact from physical
Actions 44555827205. DO NOT use reference SQL to formulate candidate.
No model calls, repair changes, or official full-BIRD-EX promotion claim.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time

SOURCE_SHA="ffd8018378ddb1a8794753e0a31cfc81862ff7318a5184c22f3dc4ce03a03feb"
SOURCE_FIRST_PROTOCOL="ORIGINAL-QWEN9B-BIRD-DEV500-81-SOURCE-FIRST-PHYSICAL-GRAIN-V1"
EXPECTED_IDS={"109","1028","1256","1289","1441"}


def run_query(db:Path,sql:str):
    if not re.match(r"(?is)^\s*(SELECT|WITH)\b", sql):
        return {"status":"non_readonly"}
    conn=sqlite3.connect(db.as_uri()+"?mode=ro",uri=True)
    conn.execute("PRAGMA query_only=ON")
    start=time.monotonic()
    conn.set_progress_handler(lambda:int(time.monotonic()-start>8.0),5000)
    try:
        cursor=conn.execute(sql)
        rows=cursor.fetchmany(50001)
        if len(rows)>50000:return {"status":"capped"}
        normalized=sorted({json.dumps(list(row),sort_keys=True,default=str)
                           for row in rows})
        return {"status":"ok","n_rows":len(rows),
                "set_size":len(normalized),
                "set_sha256":hashlib.sha256(
                    json.dumps(normalized,sort_keys=True).encode()).hexdigest()}
    except sqlite3.Error as exc:
        return {"status":"error","exception":type(exc).__name__}
    finally:
        conn.set_progress_handler(None,0)
        conn.close()


def locate(root:Path,db:str)->Path:
    a=list(root.rglob(db+".sqlite"))
    if len(a)!=1:raise RuntimeError("original DB physical source drift:"+db)
    return a[0]


def main():
    p=argparse.ArgumentParser()
    for key in ("source_first","original_dev","db_root","out"):
        p.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError("same evidence cannot be overwritten")
    original=a.original_dev.read_bytes()
    assert hashlib.sha256(original).hexdigest()==SOURCE_SHA
    first=json.loads(a.source_first.read_text())
    assert first["protocol"]==SOURCE_FIRST_PROTOCOL
    assert first["remaining_count"]==81 and first["database_count"]==11
    assert first["no_gold_reference_consulted"]
    assert first["no_guarded_code_changed"]
    probes=first["predeclared_count_grain_probes"]
    assert {x["case_id"] for x in probes}==EXPECTED_IDS and len(probes)==5
    dev={str(x["question_id"]):x for x in json.loads(original)}
    rows=[]
    for probe in sorted(probes,key=lambda x:int(x["case_id"])):
        cid=probe["case_id"]
        source=dev[cid]
        assert source["db_id"]==probe["db_id"]
        assert source["question"]==next(
            x["question"] for x in first["source_first_cases"] if x["case_id"]==cid)
        db=locate(a.db_root,probe["db_id"])
        gold=run_query(db,source["SQL"])
        control=run_query(db,probe["original_source_sql"])
        candidate=run_query(db,probe["count_distinct_counterfactual_sql"])
        assert control["status"]==probe["original_result"]["status"]
        assert candidate["status"]==probe["distinct_result"]["status"]
        if gold["status"]=="ok" and control["status"]=="ok":
            base=gold["set_sha256"]==control["set_sha256"]
        else:
            base=None
        if gold["status"]=="ok" and candidate["status"]=="ok":
            treat=gold["set_sha256"]==candidate["set_sha256"]
        else:
            treat=None
        rows.append({
            "case_id":cid,"db":probe["db_id"],
            "source_first_question_sha256":hashlib.sha256(
                source["question"].encode()).hexdigest(),
            "original_sql_sha256":hashlib.sha256(
                probe["original_source_sql"].encode()).hexdigest(),
            "predeclared_only_count_distinct_sql_sha256":hashlib.sha256(
                probe["count_distinct_counterfactual_sql"].encode()).hexdigest(),
            "reference_sql_sha256":hashlib.sha256(source["SQL"].encode()).hexdigest(),
            "reference":gold,"original":control,"count_distinct":candidate,
            "original_set_match":base,"count_distinct_set_match":treat,
            "original_wrong_recovered":base is False and treat is True,
            "new_regression":base is True and treat is False,
        })
    scorable=[x for x in rows if x["original_set_match"] is not None
              and x["count_distinct_set_match"] is not None]
    summary={
        "protocol":"BIRD-DEV500-COUNT-KEY-POST-BLIND-SAME-SOURCE-SQL-EX-CHECK-V1",
        "source_first_run":"https://github.com/kolihhan/semantic-text2sql-agent/actions/runs/44555827205",
        "physical_source_first_complete_before_reference":True,
        "source_first_cases_81_11_db":True,
        "original_qwen_digest":"56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90",
        "population_predeclared_physical_probes":5,
        "scorable_both_arms":len(scorable),
        "original_correct":sum(x["original_set_match"] is True for x in scorable),
        "distinct_correct":sum(x["count_distinct_set_match"] is True for x in scorable),
        "rescues":sum(x["original_wrong_recovered"] for x in scorable),
        "regressions":sum(x["new_regression"] for x in scorable),
        "case_results":rows,
        "no_model_calls":True,"no_Guarded_end_to_end_2_repair_ab":True,
        "no_training":True,"no_hidden_holdout":True,
        "claim_boundary":"Post-blind retrospective 5-case SQL replacement diagnostic is NOT a transferable Guarded intervention or official BIRD EX uplift. Any positive within one DB requires independent multiple-DB mechanistic evidence before a new frozen matched A/B.",
    }
    a.out.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print("POST_BLIND_FIVE_PROBE_SCORED",json.dumps({
        "denominator":summary["scorable_both_arms"],
        "baseline":summary["original_correct"],
        "distinct":summary["distinct_correct"],
        "rescues":summary["rescues"],"regressions":summary["regressions"],
        "case_results":[[x["case_id"],x["original_set_match"],
                          x["count_distinct_set_match"],x["reference"]["status"]]
                         for x in rows]
    },sort_keys=True))


if __name__=="__main__":
    main()
