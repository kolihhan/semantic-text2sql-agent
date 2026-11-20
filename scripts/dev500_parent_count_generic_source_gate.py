"""ID-agnostic parent-entity COUNT(*) join-fanout gate; public BIRD source only.

Preregistered two-stage CPU-only audit: the source job CANNOT fetch publisher
reference SQL. Detector uses only English requested entity, SQL text, declared
PK/FK and a direct equijoin. Physical read-only row-set hashes are measured
BEFORE gold is opened in a separate scoring job. Diagnostic, not deployable.

Caveat: this rule was formulated after reviewing known BIRD DEV error cases;
cross-database or new-case generalization is NOT established.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3

from scripts.postblind_bird_dev500_count_probe5 import locate, run_query, SOURCE_SHA

ROOT_COUNT = re.compile(
    r"^\s*SELECT\s+COUNT\s*\(\s*\*\s*\)(?:\s+(?:AS\s+)?\w+)?\s+FROM\b", re.I|re.S)
REQUESTED = re.compile(
    r"^\s*(?:how\s+many|what\s+is\s+the\s+number\s+of)\s+(\w+)\b", re.I)
TABLES = re.compile(
    r"\b(FROM|JOIN)\s+([A-Za-z_]\w*)(?:\s+(?:AS\s+)?([A-Za-z_]\w*))?", re.I)
RESERVED = frozenset(("on","where","group","order","limit","having","join",
                      "inner","left","right","cross","full","union","offset","using"))
PROTOCOL = "QWEN35-DEV500-ID-AGNOSTIC-FK-COUNT-GATE-SOURCE-FIRST-V1"
MODEL_DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"


def digest(data):
    if isinstance(data,str):
        data=data.encode()
    return sha256(data).hexdigest()


def singular(token):
    token=token.casefold()
    if token.endswith("ies") and len(token)>3:
        return token[:-3]+"y"
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tables_and_aliases(sql):
    seen=[]
    for m in TABLES.finditer(sql):
        table=m.group(2)
        alias=m.group(3) or table
        if alias.casefold() in RESERVED:
            alias=table
        seen.append((table,alias))
    return seen


def detect(question, sql, schemas):
    """Return a structural hypothesis, never an outcome/probability prediction.

    Fail closed on a complex root query, missing / compound PK, implicit FK,
    ambiguous target or non-direct child-parent ON equality.
    """
    if not isinstance(question,str) or not isinstance(sql,str) or not isinstance(schemas,dict):
        return None
    noun=REQUESTED.match(question)
    if noun is None or ROOT_COUNT.match(sql) is None:
        return None
    if re.search(r"\b(GROUP\s+BY|HAVING|UNION|INTERSECT|EXCEPT|WITH)\b",sql,re.I):
        return None
    rows=tables_and_aliases(sql)
    if len(rows)<2 or len({t.casefold() for t,a in rows})!=len(rows):
        return None
    matches=[(table,alias) for table,alias in rows if singular(table)==singular(noun.group(1))]
    if len(matches)!=1:
        return None
    parent, pa=matches[0]
    parent_schema=next((v for k,v in schemas.items() if k.casefold()==parent.casefold()),None)
    if not isinstance(parent_schema,dict):
        return None
    pk=parent_schema.get("primary_key",parent_schema.get("pk"))
    if not isinstance(pk,list) or len(pk)!=1:
        return None
    parent_col=pk[0]
    if not isinstance(parent_col,str) or not re.fullmatch(r"[A-Za-z_]\w*",parent_col):
        return None
    for child,ca in rows:
        if child.casefold()==parent.casefold():
            continue
        child_schema=next((v for k,v in schemas.items() if k.casefold()==child.casefold()),None)
        if not isinstance(child_schema,dict):
            continue
        for fk in child_schema.get("declared_fk",[]):
            if not isinstance(fk,dict):
                continue
            if str(fk.get("to_table","")).casefold()!=parent.casefold():
                continue
            if str(fk.get("to_col","")).casefold()!=parent_col.casefold():
                continue
            child_col=fk.get("from")
            if not isinstance(child_col,str) or not re.fullmatch(r"[A-Za-z_]\w*",child_col):
                continue
            a=rf"\b{re.escape(ca)}\.{re.escape(child_col)}\s*=\s*{re.escape(pa)}\.{re.escape(parent_col)}\b"
            b=rf"\b{re.escape(pa)}\.{re.escape(parent_col)}\s*=\s*{re.escape(ca)}\.{re.escape(child_col)}\b"
            if not (re.search(a,sql,re.I) or re.search(b,sql,re.I)):
                continue
            proposed,n=re.subn(r"\bCOUNT\s*\(\s*\*\s*\)",
                               f"COUNT(DISTINCT {pa}.{parent_col})",sql,count=1,flags=re.I)
            if n!=1:
                continue
            return {"parent":parent,"parent_pk":parent_col,"parent_alias":pa,
                    "child":child,"child_fk":child_col,"proposed_sql":proposed}
    return None


def physical_schema(db_file,sql):
    """Source-only PRAGMA: needed for prior K8-selector-right controls."""
    schemas={}
    conn=sqlite3.connect(db_file.as_uri()+"?mode=ro",uri=True)
    try:
        conn.execute("PRAGMA query_only=ON")
        for table,alias in tables_and_aliases(sql):
            if table in schemas:
                continue
            quoted=table.replace('"','""')
            cols=list(conn.execute(f'PRAGMA table_info("{quoted}")'))
            if not cols:
                continue
            fks=list(conn.execute(f'PRAGMA foreign_key_list("{quoted}")'))
            schemas[table]={
                "primary_key":[x[1] for x in sorted(cols,key=lambda c:c[5])
                               if x[5]>0],
                "declared_fk":[{"from":x[3],"to_col":x[4],"to_table":x[2]}
                               for x in fks],
            }
    finally:
        conn.close()
    return schemas


def source_phase(source_first,db_root):
    source=json.loads(source_first.read_text())
    assert source["remaining_count"]==81 and source["database_count"]==11
    assert source["n_fully_executable_wrong"]==97
    assert source["no_gold_reference_consulted"] and source["no_new_model_calls"]
    assert source["original_model_digest"]==MODEL_DIGEST
    rows=[]
    for obj in source["source_first_cases"]:
        cid=obj["case_id"]; db=obj["db_id"]
        question=obj["question"]; original=obj["first_original_qwen_sql"]
        hypothesis=detect(question,original,obj["source_physical_schema_and_declared_keys"])
        record={"case_id":cid,"db":db,"question_sha256":digest(question),
                "original_sql_sha256":digest(original),"detected":hypothesis is not None,
                "cohort":"KNOWN_ORIGINAL_K8_ALL_WRONG",
                "no_reference_sql_access":True}
        if hypothesis is not None:
            physical=locate(db_root,db)
            original_result=run_query(physical,original)
            candidate_result=run_query(physical,hypothesis["proposed_sql"])
            record.update({"hypothesis":{k:v for k,v in hypothesis.items()
                                         if k!="proposed_sql"},
                           "proposed_sql_sha256":digest(hypothesis["proposed_sql"]),
                           "original_sql":original,"proposed_sql":hypothesis["proposed_sql"],
                           "source_only_original_result":original_result,
                           "source_only_candidate_result":candidate_result,
                           "physical_result_changed":(original_result["status"]=="ok"
                               and candidate_result["status"]=="ok"
                               and original_result["set_sha256"]!=candidate_result["set_sha256"])})
        rows.append(record)
    controls=[]
    for obj in source["existing_correct_controls"]:
        db=obj["db_id"]; question=obj["question"]; original=obj["first_original_sql"]
        physical=locate(db_root,db)
        hypothesis=detect(question,original,physical_schema(physical,original))
        control={"case_id":obj["case_id"],"db":db,
                 "question_sha256":digest(question),"original_sql_sha256":digest(original),
                 "cohort":"K8_SELECTED_CORRECT_FIRST_CANDIDATE_UNVERIFIED",
                 "k8_selector_known_correct":True,"first_candidate_known_correct":False,
                 "detected":hypothesis is not None}
        if hypothesis is not None:
            orig_result=run_query(physical,original)
            changed_result=run_query(physical,hypothesis["proposed_sql"])
            control.update({
                "original_sql":original,"proposed_sql":hypothesis["proposed_sql"],
                "original_result":orig_result,"candidate_result":changed_result,
                "physical_result_changed":(
                    orig_result["status"]=="ok" and changed_result["status"]=="ok"
                    and orig_result["set_sha256"]!=changed_result["set_sha256"])})
        controls.append(control)
    return {
        "protocol":PROTOCOL,"phase":"SOURCE_ONLY","population":len(rows),
        "source_11_dbs":len({x["db"] for x in rows}),
        "detected":sum(x["detected"] for x in rows),
        "actual_physical_result_changes":sum(x.get("physical_result_changed",False) for x in rows),
        "cases":rows,"k8_selector_correct_controls":controls,
        "n_controls":len(controls),"n_controls_detected":sum(x["detected"] for x in controls),
        "first_candidate_right_controls_independently_verified":False,
        "source_outcome_conditioned":True,"posthoc_rule_design":True,
        "no_publisher_reference_read":True,"no_new_Qwen_calls":True,
        "no_guarded_change":True,"no_training":True,"holdout300_untouched":True,
        "promotion_forbidden":True,
    }


def score_phase(frozen,official_dev,db_root):
    x=json.loads(frozen.read_text())
    assert x["phase"]=="SOURCE_ONLY" and x["protocol"]==PROTOCOL
    assert x["population"]==81 and x["no_publisher_reference_read"]
    official=official_dev.read_bytes()
    assert digest(official)==SOURCE_SHA
    gold={str(obj["question_id"]):obj for obj in json.loads(official)}
    records=[]
    for row in x["cases"]:
        if not row["detected"]:
            continue
        source=gold[row["case_id"]]
        assert source["db_id"]==row["db"]
        assert digest(source["question"])==row["question_sha256"]
        physical=locate(db_root,row["db"])
        expected=run_query(physical,source["SQL"])
        original=run_query(physical,row["original_sql"])
        changed=run_query(physical,row["proposed_sql"])
        assert original==row["source_only_original_result"]
        assert changed==row["source_only_candidate_result"]
        before=(expected["set_sha256"]==original["set_sha256"]
                if expected["status"]=="ok" and original["status"]=="ok" else None)
        after=(expected["set_sha256"]==changed["set_sha256"]
               if expected["status"]=="ok" and changed["status"]=="ok" else None)
        records.append({"case_id":row["case_id"],"db":row["db"],
                        "original_match":before,"candidate_match":after,
                        "rescue":before is False and after is True,
                        "regression":before is True and after is False})
    controls=[]
    for row in x["k8_selector_correct_controls"]:
        source=gold[row["case_id"]]
        physical=locate(db_root,row["db"])
        original=run_query(physical,source["SQL"])
        ctrl_sql=next(c["first_original_sql"] for c in
                      [z for z in json.loads(frozen.read_text())["k8_selector_correct_controls"]]
                      if c["case_id"]==row["case_id"]) if False else None
        # Source stage only retained SQL for controls when a rule fired;
        # source never saw reference and controls without trigger cannot
        # be independently certified as FIRST-original-candidate right.
        controls.append({"case_id":row["case_id"],
                         "k8_selector_correct":True,
                         "first_candidate_correct":None,
                         "unsupported_claim":"cannot certify first candidate from selected K8 success"})
    valid=[v for v in records if v["original_match"] is not None
           and v["candidate_match"] is not None]
    return {
        "protocol":PROTOCOL,"phase":"POSTBLIND_DIAGNOSTIC",
        "source_activation_denominator":81,
        "source_activated":x["detected"],
        "physical_result_changes":x["actual_physical_result_changes"],
        "scorable_activated":len(valid),
        "rescues":sum(v["rescue"] for v in valid),
        "regressions":sum(v["regression"] for v in valid),
        "case_results":records,"control_stratum":controls,
        "actual_first_correct_control_denominator":0,
        "no_new_Qwen_calls":True,"no_training":True,
        "no_Guarded_end_to_end_BIRD_EX":True,
        "HOLDOUT300_untouched":True,
        "promotion_forbidden":True,
        "failure_boundary":"No independent first-SQL-correct negative controls; outcome-conditioned public DEV81.",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=["source","score"])
    for flag in ("source_first","sealed","official_dev","db_root","out"):
        p.add_argument("--"+flag.replace("_","-"),type=Path)
    a=p.parse_args()
    if not a.db_root or not a.out:
        p.error("--db-root and --out required")
    if a.out.exists():
        raise FileExistsError("source evidence must be immutable")
    if a.phase=="source":
        assert a.source_first and not a.official_dev
        result=source_phase(a.source_first,a.db_root)
    else:
        assert a.sealed and a.official_dev
        result=score_phase(a.sealed,a.official_dev,a.db_root)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("ID_AGNOSTIC_FK_COUNT_GATE",json.dumps({
        "phase":result["phase"],
        "source_activation_denominator":result.get("population",result.get("source_activation_denominator")),
        "source_activated":result.get("detected",result.get("source_activated")),
        "physical_result_changes":result.get("actual_physical_result_changes",result.get("physical_result_changes")),
        "rescues":result.get("rescues"),"regressions":result.get("regressions"),
        "new_qwen_calls":0,"promotion_forbidden":True,
    },sort_keys=True))


if __name__=="__main__":
    main()
