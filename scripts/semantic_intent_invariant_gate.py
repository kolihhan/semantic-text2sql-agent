"""Gold-free, conservative semantic invariants for proposed SQL repairs.
Research-only. No reference SQL, execution-match, or case-specific rules.
"""
from __future__ import annotations
import re

_QUOTED = re.compile(r''''((?:''|[^']){2,120})'|"([^"]{2,120})"''', re.I)
_EXPLICIT_ID = re.compile(r"\b(?:id|identifier)\b\s*(?:is|equals|=|:|#)?\s*(\d{1,15})\b", re.I)
_NUMERIC_PRED = re.compile(r'''\b(?:(?P<alias>[A-Za-z_]\w*)\s*\.\s*)?(?P<column>"?[A-Za-z_]\w*"?)\s*(?P<op>=|LIKE)\s*['"]?(?P<value>\d{1,15})(?P<wildcard>%?)['"]?''', re.I)
_AGG = re.compile(r"\b(?:COUNT|SUM|AVG|MIN|MAX)\s*\(", re.I)
_LIKE = re.compile(r"\bLIKE\b", re.I)
_EXPLICIT_PARTIAL = re.compile(r"\b(?:starts?\s+with|begins?\s+with|contains?|substring|partially\s+matches?)\b", re.I)

def _lits(text: str) -> set[str]:
    return {(a or b).replace("''", "'").strip().casefold() for a,b in _QUOTED.findall(text)}

def numeric_id_binding_conflict(question: str, sql: str) -> bool:
    """Question explicitly requests ID N but SQL predicates N on non-ID field.
    A signal, not a proof that a replacement predicate is correct.
    """
    ids=set(_EXPLICIT_ID.findall(question))
    if len(ids)!=1:
        return False
    num=next(iter(ids))
    seen=False
    for m in _NUMERIC_PRED.finditer(sql):
        if m.group("value")!=num:
            continue
        col=m.group("column").strip('"').casefold()
        if col=="id" or col.endswith("_id"):
            return False
        seen=True
    return seen

def candidate_violations(question: str, original_sql: str, proposed_sql: str) -> list[str]:
    violations=[]
    original_literals=_lits(original_sql)
    new_literals=_lits(proposed_sql)
    required=original_literals & _lits(question)
    if required and not required.issubset(new_literals):
        violations.append("QUESTION_NAMED_ENTITY_REPLACED")
    stable_names={s for s in original_literals if re.search(r"[A-Za-z]",s)}
    stable_new={s for s in new_literals if re.search(r"[A-Za-z]",s)}
    if stable_names and stable_new and not (stable_names & stable_new):
        violations.append("ORIGINAL_NAMED_LITERAL_REPLACED")
    if numeric_id_binding_conflict(question, proposed_sql):
        violations.append("NUMERIC_ID_VALUE_BOUND_TO_NON_ID_COLUMN")
    if _AGG.search(original_sql) and not _AGG.search(proposed_sql):
        violations.append("AGGREGATION_REMOVED")
    if not _EXPLICIT_PARTIAL.search(question) and (required or _EXPLICIT_ID.search(question)) and _LIKE.search(proposed_sql) and not _LIKE.search(original_sql):
        violations.append("EXACT_FILTER_BROADENED_TO_LIKE")
    return sorted(set(violations))

def freeze_decisions(unlabeled: dict, frozen: dict)->dict:
    cases=unlabeled.get("cases")
    predictions=frozen.get("cases")
    if not isinstance(cases,list) or not isinstance(predictions,list) or len(cases)!=64 or len(predictions)!=64:
        raise ValueError("original unlabeled TRAIN64 cohort expected")
    allowed={"source_index","db_id","question","sql","prompt_sha256"}
    if any(set(row)!=allowed for row in cases):
        raise ValueError("gold/outcome field found in unlabeled input")
    byid={str(row["source_index"]):row for row in cases}
    if len(byid)!=64 or len({str(row["source_index"]) for row in predictions})!=64:
        raise ValueError("duplicate case IDs")
    decisions=[]
    for row in predictions:
        idx=str(row["source_index"])
        if idx not in byid or row["db_id"]!=byid[idx]["db_id"]:
            raise ValueError("original identity mismatch")
        if row["status"]!="FROZEN_CANDIDATE":
            decisions.append({"source_index":idx,"db_id":row["db_id"],"action":"UNCHANGED","flags":[]})
            continue
        sql=row.get("candidate_sql")
        if not isinstance(sql,str) or not sql:
            raise ValueError("missing candidate SQL")
        flags=candidate_violations(byid[idx]["question"],byid[idx]["sql"],sql)
        decisions.append({"source_index":idx,"db_id":row["db_id"],"action":"REJECT_CANDIDATE" if flags else "ACCEPT_CANDIDATE","flags":flags})
    return {"status":"FROZEN_INTENT_INVARIANT_DECISIONS_NO_GOLD","n":64,"decisions":decisions,"no_gold_accessed":True,"default_pipeline_modified":False}
