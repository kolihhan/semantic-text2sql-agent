"""Post-freeze historical Qwen48 structural unit probe and label-only audit.

Strict order: (1) independent source-only frozen artifact, (2) original SQL,
(3) original historical SET EX labels. No new inference/gold SQL/rewrite.
Model-set EX correctness is NOT proof of a unit/relation semantic mismatch.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re

from scripts.typed_erp_source_first import PROTOCOL
from scripts.train48_original_qwen_unit_gate_source_then_score import (
    _original_shards, OLD_PROTO, MODEL_SHA, TRAIN_SHA,
)

PROBE_PROTO="BIRD-TRAIN-ERP-POSTSOURCE-PRESCORE-SQL-PROBE-V1"
SCORE_PROTO="BIRD-TRAIN-ERP-POSTSEAL-MIXED-CONTROL-AUDIT-V1"
_IDENT=r'["\`]?([A-Za-z_][A-Za-z0-9_]*)["\`]?'
TABLE=re.compile(r"\b(FROM|JOIN)\s+"+_IDENT+r"(?:\s+(?:AS\s+)?"+_IDENT+r")?",re.I)
COUNT=re.compile(r"\bCOUNT\s*\(\s*(DISTINCT\s+)?(?:(("+_IDENT+r")\.)?"+_IDENT+r"|\*)\s*\)",re.I)
BLOCK=re.compile(r"\b(WITH|UNION|INTERSECT|EXCEPT|GROUP\s+BY|HAVING|OVER|LATERAL|CROSS\s+JOIN)\b",re.I)
RESERVED=frozenset(("on","where","join","left","right","inner","outer","group",
                    "order","limit","having","and","or","using","as","cross","union"))
def checksum(x: bytes) -> str: return sha256(x).hexdigest()

def _parse_count(sql: str) -> dict|None:
    if not isinstance(sql,str) or len(sql)>10000 or "--" in sql or "/*" in sql:
        return None
    s=sql.strip().rstrip(";")
    if not re.match(r"(?is)^SELECT\b",s) or ";" in s or BLOCK.search(s):
        return None
    if len(re.findall(r"\bSELECT\b",s,re.I))!=1: return None
    matches=list(COUNT.finditer(s))
    if len(matches)!=1: return None
    m=matches[0]
    # Require a single simple count in the select clause. No count in subquery.
    from_match=re.search(r"\bFROM\b",s,re.I)
    if not from_match or m.start()>from_match.start():return None
    tables={}
    for t in TABLE.finditer(s):
        name=t.group(2).lower()
        alias=(t.group(3) or name).lower()
        if alias in RESERVED:alias=name
        if alias in tables:return None
        tables[alias]=name
    if not tables:return None
    expr=m.group(0)
    inside=expr[expr.find("(")+1:expr.rfind(")")].strip()
    cm=re.fullmatch(r"(?is)(DISTINCT\s+)?(?:(?:(\w+)\.)?(\w+)|(\*))",inside)
    if cm is None: return None
    return {"distinct":bool(cm.group(1)),"alias":(cm.group(2) or "").lower(),
            "column":cm.group(3),"star":bool(cm.group(4)),"tables":tables}

def decide(question_record: dict,sql: str)->dict:
    """A conflict is only a candidate until independent source-intent review."""
    label=question_record["candidate_label"]
    if label!="ENTITY_COUNT":
        return {"decision":"ABSTAIN","reason":"MECHANISM_NOT_APPLICABLE_OR_UNPROVEN","semantic_conflict_verified":False}
    unit=question_record.get("counting_unit")
    if not unit or not unit["explicit_unique_requested"]:
        return {"decision":"ABSTAIN","reason":"EXPLICIT_UNIQUE_UNIT_NOT_GROUNDED","semantic_conflict_verified":False}
    parsed=_parse_count(sql)
    if parsed is None:
        return {"decision":"ABSTAIN","reason":"SQL_STRUCTURE_OUT_OF_SCOPE","semantic_conflict_verified":False}
    if parsed["star"] or not parsed["distinct"]:
        return {"decision":"ABSTAIN","reason":"COUNT_STAR_OR_NO_DISTINCT_NEEDS_PHYSICAL_PROOF","semantic_conflict_verified":False}
    if not parsed["alias"]:
        return {"decision":"ABSTAIN","reason":"UNQUALIFIED_COUNT_COLUMN","semantic_conflict_verified":False}
    actual_table=parsed["tables"].get(parsed["alias"])
    if actual_table is None:
        return {"decision":"ABSTAIN","reason":"COUNT_ALIAS_UNRESOLVED","semantic_conflict_verified":False}
    expected=unit["table"].lower()
    if expected not in parsed["tables"].values():
        return {"decision":"ABSTAIN","reason":"EXPECTED_TARGET_TABLE_NOT_IN_SQL","semantic_conflict_verified":False}
    if actual_table==expected and parsed["column"].lower()==unit["pk"].lower():
        return {"decision":"CONSISTENT_UNIT","reason":"COUNT_DISTINCT_EXACT_TARGET_PK_ONLY","semantic_conflict_verified":False}
    if actual_table==expected:
        return {"decision":"ABSTAIN","reason":"NON_PK_UNIQUENESS_UNKNOWN","semantic_conflict_verified":False}
    actual_pks=question_record["schema_evidence"]["table_primary_keys"].get(actual_table,[])
    declared=question_record["schema_evidence"]["declared_foreign_keys"]
    direct=any(f["child"]==actual_table and f["parent"]==expected for f in declared)
    if direct and len(actual_pks)==1 and parsed["column"].lower()==actual_pks[0].lower():
        return {"decision":"CONFLICT_CANDIDATE","reason":"EXPLICIT_TARGET_PK_VS_DISTINCT_CHILD_PK",
                "semantic_conflict_verified":False,"target_table":expected,"actual_table":actual_table}
    return {"decision":"ABSTAIN","reason":"COUNTED_UNIT_ROLE_OR_FK_UNPROVEN","semantic_conflict_verified":False}

def read_frozen(path:Path)->dict:
    a=json.loads(path.read_text(encoding="utf-8"))
    assert a["protocol"]==PROTOCOL and a["status"]=="FROZEN_BEFORE_MODEL_SQL_OR_SCORE"
    assert a["source_frozen_no_model_or_score_or_gold_SQL_access"] is True
    assert a["original_train_sha256"]==TRAIN_SHA
    assert a["source_db_count"]==12 and a["per_db"]==20
    assert len(a["source_candidates"])==240 and len(a["original_Qwen48_source_only_controls"])==48
    assert len({r["source_index"] for r in a["source_candidates"]})==240
    assert not any("SQL" in r or "gold" in r for r in a["original_Qwen48_source_only_controls"])
    return a

def probe(frozen:Path,generated_dir:Path)->dict:
    a=read_frozen(frozen)
    source=a["original_Qwen48_source_only_controls"]
    sqls=_original_shards(generated_dir,source,a["old48_source_sha256"])
    decisions=[]
    for row in source:
        sql=sqls[row["source_index"]]
        result=decide(row,sql)
        decisions.append({"source_index":row["source_index"],"db_id":row["db_id"],
                          "question_sha256":row["question_sha256"],
                          "original_sql_sha256":checksum(sql.encode()),
                          **result})
    return {
        "protocol":PROBE_PROTO,"source_frozen_file_sha256":checksum(frozen.read_bytes()),
        "source_frozen_git_sha":a["git_commit_sha"],"old48_source_sha256":a["old48_source_sha256"],
        "model_digest":MODEL_SHA,"n":48,"n_databases":len({r["db_id"] for r in decisions}),
        "decision_histogram":dict(sorted(Counter(d["decision"] for d in decisions).items())),
        "decisions":sorted(decisions,key=lambda d:d["source_index"]),
        "no_outcomes_or_reference_gold_read":True,"no_new_Qwen_or_rewrite_or_training":True,
        "candidate_not_semantically_verified":True,
        "no_DEV500_or_HOLDOUT300":True,
    }

def score(probe_path:Path,scored_dir:Path)->dict:
    a=json.loads(probe_path.read_text())
    assert a["protocol"]==PROBE_PROTO and a["n"]==48 and a["no_outcomes_or_reference_gold_read"]
    cases={x["source_index"]:x for x in a["decisions"]}
    outcomes={}
    for shard in range(8):
        d=json.loads((scored_dir/f"scored-{shard}.json").read_text())
        assert d["protocol"]==OLD_PROTO and d["model_digest"]==MODEL_SHA
        assert d["source_sha256"]==a["old48_source_sha256"]
        assert d["status"]=="PUBLISHER_STRICT_PHYSICAL_SET_BAG_EX_SCORED_POST_SEAL"
        assert d["n"]==6 and d["shard"]==shard
        for item in d["cases"]:
            cid=item["source_index"]
            if cid not in cases or cid in outcomes:raise ValueError("score identity mismatch")
            outcome=item["control_selected_set_EX"] if item["reference_scorable"] else None
            if outcome is not None and not isinstance(outcome,bool):
                raise ValueError("non-boolean old SET EX outcome")
            outcomes[cid]=outcome
    if len(outcomes)!=48:raise ValueError("missing original SQL mixed controls")
    ct=defaultdict(Counter)
    bydb=defaultdict(Counter)
    for cid,d in cases.items():
        x=outcomes[cid]
        status="right" if x is True else "wrong" if x is False else "censored"
        ct[d["decision"]][status]+=1
        bydb[d["db_id"]][d["decision"]]+=1
    row=lambda k: {v:ct[k][v] for v in ("right","wrong","censored")}
    matrix={key:row(key) for key in ("CONSISTENT_UNIT","CONFLICT_CANDIDATE","ABSTAIN")}
    assert sum(x["right"] for x in matrix.values())==26
    assert sum(x["wrong"] for x in matrix.values())==20
    assert sum(x["censored"] for x in matrix.values())==2
    conflict=matrix["CONFLICT_CANDIDATE"]
    flagged=conflict["right"]+conflict["wrong"]
    return {
        "protocol":SCORE_PROTO,"probe_sha256":checksum(probe_path.read_bytes()),
        "source_frozen_git_sha":a["source_frozen_git_sha"],"n":48,"n_db":a["n_databases"],
        "model_digest":MODEL_SHA,"correct_controls":26,"incorrect_controls":20,
        "censored_controls":2,"cross_tab":matrix,
        "by_database_decisions":{db:dict(c) for db,c in sorted(bydb.items())},
        "flagged_wrong_EX_not_proven_semantic_error":conflict["wrong"],
        "flagged_right_EX_false_positive_controls":conflict["right"],
        "conflict_candidate_exposure":flagged,
        "wrong_EX_flag_rate":conflict["wrong"]/20,
        "right_EX_false_alarm_rate":conflict["right"]/26,
        "EX_label_conditional_precision_proxy":conflict["wrong"]/flagged if flagged else None,
        "verified_semantic_conflicts":0,
        "verified_semantic_precision":None,
        "verified_semantic_detection_coverage":None,
        "source_sample_never_used_for_new_model_inference":True,
        "reused_train48_is_not_independent_confirmation":True,
        "P3_decision":"REJECT_NEW_QWEN_AB_NOT_YET_VALIDATED",
        "reason":"Auto-grounding labels unreviewed; SQL flags not independently validated semantic contradictions; no clean generalization replication.",
        "no_new_Qwen_calls":True,"no_train":True,"holdout300_untouched":True,
    }

def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="step",required=True)
    x=sub.add_parser("probe")
    x.add_argument("--frozen",required=True,type=Path)
    x.add_argument("--generated-dir",required=True,type=Path)
    x.add_argument("--out",required=True,type=Path)
    y=sub.add_parser("score")
    y.add_argument("--probe",required=True,type=Path)
    y.add_argument("--scored-dir",required=True,type=Path)
    y.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError("immutable artifact already exists")
    obj=probe(args.frozen,args.generated_dir) if args.step=="probe" else score(args.probe,args.scored_dir)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False)+"\n")
    print(json.dumps({k:v for k,v in obj.items() if k!="decisions"},sort_keys=True))

if __name__=="__main__":
    main()
