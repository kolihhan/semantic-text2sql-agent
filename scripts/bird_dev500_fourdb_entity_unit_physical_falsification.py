"""Predeclared CPU-only physical entity-unit falsification, NO reference/gold SQL.

Four source-selected error hypotheses over four public BIRD DEV DBs, plus
three previously proposed but physically invariant COUNT DISTINCT controls.
Runs only read-only, bounded aggregate SELECTs, exports COUNTs and relational
keys; no raw DB records, task rewrite, Qwen inference or HOLDOUT300 access.

Source-selected cases are known DEV failures, so even verified DB cardinality
is diagnostic support, NOT unbiased benchmark improvement.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import time

from scripts.audit_dev500_81_physical_keys import get_db

PROTOCOL="BIRD-DEV500-FOUR-DB-TYPED-ENTITY-UNIT-PHYSICAL-FALSIFICATION-V1"
MODEL_SHA="56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"

# Output counts only. The probe is grounded in source-first question and
# actual schema keys, never in publisher/gold SQL.
PROBES={
"219":{
  "db":"toxicology","target":"molecule","row_unit":"bond",
  "primary_table":"bond","target_table":"molecule",
  "probes":{
    "triple_bond_rows":"SELECT COUNT(*) FROM bond WHERE bond_type='#'",
    "triple_bond_unique_molecules":"SELECT COUNT(DISTINCT molecule_id) FROM bond WHERE bond_type='#'",
    "carcinogenic_triple_bond_rows":"SELECT COUNT(*) FROM bond b JOIN molecule m ON b.molecule_id=m.molecule_id WHERE b.bond_type='#' AND m.label='+'",
    "carcinogenic_triple_bond_unique_molecules":"SELECT COUNT(DISTINCT b.molecule_id) FROM bond b JOIN molecule m ON b.molecule_id=m.molecule_id WHERE b.bond_type='#' AND m.label='+'",
  },
  "a":"triple_bond_rows","b":"triple_bond_unique_molecules",
  "mechanism":"molecule rather than bond denominator/target among triple bonds"
},
"571":{
  "db":"codebase_community","target":"user_vote","row_unit":"post_join_vote",
  "primary_table":"votes","target_table":"users",
  "probes":{
    "posts_by_user24":"SELECT COUNT(*) FROM posts WHERE OwnerUserId=24",
    "votes_cast_by_user24":"SELECT COUNT(*) FROM votes WHERE UserId=24",
    "votes_cast_by_user24_on_own_posts":"SELECT COUNT(*) FROM votes v JOIN posts p ON v.PostId=p.Id WHERE v.UserId=24 AND p.OwnerUserId=24",
    "votes_anyone_cast_on_user24_posts":"SELECT COUNT(*) FROM votes v JOIN posts p ON v.PostId=p.Id WHERE p.OwnerUserId=24",
  },
  "a":"votes_cast_by_user24","b":"votes_cast_by_user24_on_own_posts",
  "mechanism":"votes by user versus votes by user on own posts (join-filter scope)"
},
"1256":{
  "db":"thrombosis_prediction","target":"patient","row_unit":"laboratory_record",
  "primary_table":"Laboratory","target_table":"Patient",
  "probes":{
    "abnormal_CRP_lab_rows":"SELECT COUNT(*) FROM Laboratory L JOIN Patient P ON L.ID=P.ID WHERE L.CRP='+' AND P.Description IS NULL",
    "abnormal_CRP_unique_patients":"SELECT COUNT(DISTINCT P.ID) FROM Laboratory L JOIN Patient P ON L.ID=P.ID WHERE L.CRP='+' AND P.Description IS NULL",
  },
  "a":"abnormal_CRP_lab_rows","b":"abnormal_CRP_unique_patients",
  "mechanism":"patient count versus multiple laboratory snapshots"
},
"1525":{
  "db":"debit_card_specializing","target":"customer","row_unit":"transaction",
  "primary_table":"transactions_1k","target_table":"customers",
  "probes":{
    "all_transaction_rows_on_date":"SELECT COUNT(*) FROM transactions_1k t JOIN customers c ON c.CustomerID=t.CustomerID WHERE t.Date='2012-08-25'",
    "all_unique_customers_on_date":"SELECT COUNT(DISTINCT c.CustomerID) FROM transactions_1k t JOIN customers c ON c.CustomerID=t.CustomerID WHERE t.Date='2012-08-25'",
    "EUR_customer_transaction_rows_on_date":"SELECT COUNT(*) FROM transactions_1k t JOIN customers c ON c.CustomerID=t.CustomerID WHERE t.Date='2012-08-25' AND c.Currency='EUR'",
    "EUR_unique_customers_on_date":"SELECT COUNT(DISTINCT c.CustomerID) FROM transactions_1k t JOIN customers c ON c.CustomerID=t.CustomerID WHERE t.Date='2012-08-25' AND c.Currency='EUR'",
  },
  "a":"all_transaction_rows_on_date","b":"all_unique_customers_on_date",
  "mechanism":"EUR customer percentage customer-vs-transaction denominator"
},
}
CONTROL_IDS=("109","1028","1441")


def count(conn:sqlite3.Connection,sql:str)->int:
    assert sql.lstrip().upper().startswith("SELECT ")
    start=time.monotonic()
    conn.set_progress_handler(lambda:1 if time.monotonic()-start>8.0 else 0,1000)
    try:
        row=conn.execute(sql).fetchone()
        assert row and isinstance(row[0],int) and row[0]>=0
        return row[0]
    finally:
        conn.set_progress_handler(None,0)


def schema(conn,table):
    cols=conn.execute('PRAGMA table_info("'+table+'")').fetchall()
    assert cols, "missing predeclared table "+table
    pk=[x[1] for x in cols if x[5]]
    fk=[{"from":x[3],"to_table":x[2],"to_col":x[4]}
        for x in conn.execute('PRAGMA foreign_key_list("'+table+'")')]
    return {"pk":pk,"fks":fk,"n_declared_columns":len(cols)}


def run(srcpath:Path, dbroot:Path):
    data=json.loads(srcpath.read_text())
    assert data["remaining_count"]==81 and data["database_count"]==11
    assert data["original_model_digest"]==MODEL_SHA
    assert data["no_gold_reference_consulted"] is True
    original={str(x["case_id"]):x for x in data["source_first_cases"]}
    controls={str(x["case_id"]):x for x in data["predeclared_count_grain_probes"]}
    assert all(cid in original for cid in PROBES)
    assert all(cid in controls for cid in CONTROL_IDS)
    entries=[]
    for cid,study in PROBES.items():
        old=original[cid]
        assert old["db_id"]==study["db"] and "question" in old
        with sqlite3.connect(get_db(dbroot,study["db"]).as_uri()+"?mode=ro",uri=True) as conn:
            conn.execute("PRAGMA query_only=ON")
            s1=schema(conn,study["primary_table"])
            s2=schema(conn,study["target_table"])
            values={k:count(conn,sql) for k,sql in study["probes"].items()}
        gap=values[study["a"]]!=values[study["b"]]
        # Relationship semantics are CASE-SPECIFIC and source-anchored.
        valid_schema=bool(s1["n_declared_columns"] and s2["n_declared_columns"])
        entries.append({
          "case_id":cid,"db":study["db"],
          "question_target_unit":study["target"],
          "original_query_counting_or_join_unit":study["row_unit"],
          "schema_declared_PK_source":s1["pk"],
          "schema_declared_PK_target":s2["pk"],
          "schema_declared_FK_source":s1["fks"],
          "source_question_first_SQL_sha256":old["original_eight_sql_sha256"],
          "aggregate_physical_counts":values,
          "predeclared_unit_pair":[study["a"],study["b"]],
          "unit_pair_physically_different":gap,
          "schema_both_tables_exist":valid_schema,
          "mechanism_hypothesis":study["mechanism"],
          "gold_correctness_cannot_be_inferred":True,
        })
    negative=[]
    for cid in CONTROL_IDS:
        p=controls[cid]
        assert p["no_gold_consulted"] and p["not_new_model_generated_sql"]
        # These were source-selected wrong-K8 cases, NOT original-right controls.
        negative.append({
            "case_id":cid,"db":p["db_id"],
            "row_and_distinct_previously_physically_identical":not p["actual_physical_result_changed"],
            "not_a_correct_result_control":True,
        })
    assert len(entries)==4 and len({e["db"] for e in entries})==4
    assert len(negative)==3 and all(e["row_and_distinct_previously_physically_identical"] for e in negative)
    changed=sum(x["unit_pair_physically_different"] for x in entries)
    return {
      "protocol":PROTOCOL,"source_type":"known_failed_DEV; not prospectively sampled",
      "n_tested":4,"n_physical_databases":4,
      "n_immutable_prior_negative_controls":3,
      "physical_unit_pair_diff_cases":changed,
      "diagnostic_cross_db_physical_variation_seen":changed>=3,
      "cases":entries,"negative_invariant_controls":negative,
      "gold_sql_not_opened":True,"new_model_calls":0,"no_training":True,
      "no_new_bird_ex":True,"holdout300_untouched":True,
      "does_not_demonstrate_a_general_SQL_repair_rule":True,
      "next_action_if_gate_satisfied":"Design independent matched question-ID-agnostic constraint generation only after human semantics + correct-case controls; no immediate model A/B",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True,type=Path)
    p.add_argument("--db-root",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError("immutable evidence file")
    obj=run(args.source,args.db_root)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")
    print("FOUR_DB_PHYSICAL_TYPED_ENTITY_UNIT_DIAGNOSTIC",json.dumps({
        "n":obj["n_tested"],"physical_unit_pair_diff_cases":obj["physical_unit_pair_diff_cases"],
        "by_case":[{"case_id":v["case_id"],"db":v["db"],
                    "difference":v["unit_pair_physically_different"]}
                    for v in obj["cases"]],"qwen_calls":0},sort_keys=True))

if __name__=="__main__":main()
