"""Conservative Question–Schema–SQL counted-unit consistency gate.

Only checks the *counted unit*, not whole-query semantic correctness.
Returns exactly CONSISTENT, CONFLICT, ABSTAIN. Never generates SQL as a
repair or uses publisher reference results. Only a single plain COUNT(*)
or COUNT(DISTINCT target_pk) SELECT is in scope; all other constructs abstain.

This is an intentionally narrow mechanism, not a general SQL parser.
Unknown aliases/FKs/groups/subqueries cannot become "CONFLICT".
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

DECISIONS=("CONSISTENT","CONFLICT","ABSTAIN")
COUNT_SELECT=re.compile(
    r"(?is)^\s*SELECT\s+COUNT\s*\(\s*(\*|DISTINCT\s+[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)\s*\)"
    r"\s*(?:AS\s+[A-Za-z_]\w*|[A-Za-z_]\w*)?\s+FROM\b")
TABLE_MARKER=re.compile(
    r'(?i)\b(FROM|JOIN)\s+(?:"([A-Za-z_]\w*)"|\x60([A-Za-z_]\w*)\x60|([A-Za-z_]\w*))'
    r'\s*(?:AS\s+)?([A-Za-z_]\w*)?')
Q_MARKER=re.compile(r"\b(?:how\s+many|number\s+of|count\s+of)\b",re.I)
UNSAFE=re.compile(r"(?i)\b(?:WITH|UNION|EXCEPT|INTERSECT|GROUP\s+BY|HAVING|OVER|WINDOW|RECURSIVE)\b")
RESERVED={"JOIN","ON","WHERE","INNER","LEFT","RIGHT","FULL","CROSS","OUTER",
          "GROUP","HAVING","ORDER","LIMIT","OFFSET","USING","UNION"}
PROTOCOL="BIRD-QUESTION-SCHEMA-SQL-UNIT-GATE-CAUTIOUS-V1"


def _singular(text:str)->str:
    s=text.lower()
    if s.endswith("ies") and len(s)>4:return s[:-3]+"y"
    if s.endswith("sses"):return s[:-2]
    if s.endswith("s") and not s.endswith(("ss","us")):return s[:-1]
    return s


def _schema(conn:sqlite3.Connection)->dict:
    names=[r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    data={}
    for t in names:
        ident=t.replace('"','""')
        cols=conn.execute('PRAGMA table_info("'+ident+'")').fetchall()
        fk=conn.execute('PRAGMA foreign_key_list("'+ident+'")').fetchall()
        pk=sorted([(r[5],r[1]) for r in cols if r[5]])
        data[t.lower()]={"name":t,"pk":pk,
                          "fk":[{"from":str(r[3] or "").lower(),
                                 "to_table":str(r[2] or "").lower(),
                                 # SQLite allows REFERENCES parent without
                                 # an explicit target column. Do not guess
                                 # which target PK the constraint denotes.
                                 "to_col":str(r[4] or "").lower()} for r in fk]}
    return data


def _question_table(question:str,table_catalog:dict)->str|None:
    """Require explicit count marker and a directly named schema table unit."""
    match=Q_MARKER.search(question)
    if match is None:return None
    tail=question[match.end():match.end()+100].lower()
    if re.match(r"\s*(more|fewer|less|of\s+them|times|much)\b",tail):
        return None
    options=[]
    for t in table_catalog:
        canonical=_singular(t)
        if len(canonical)<4:continue
        variants={canonical,t.lower(),canonical+"s"}
        if canonical.endswith("y"):
            variants.add(canonical[:-1]+"ies")
        if canonical.endswith(("s","x","ch","sh")):
            variants.add(canonical+"es")
        for word in variants:
            m=re.search(r"(?<![a-z0-9_])"+re.escape(word)+r"(?![a-z0-9_])",tail)
            if m:
                options.append((m.start(),t))
    if not options:return None
    earliest=min(x[0] for x in options)
    winners={t for p,t in options if p==earliest}
    if len(winners)!=1:return None
    # The first mentioned schema entity needs to be directly associated with
    # the count request, not hidden 80 characters after an unrelated concept.
    if earliest>48:return None
    return next(iter(winners))


def _table_refs(sql:str,schema:dict):
    refs=[]
    for m in TABLE_MARKER.finditer(sql):
        table=next((x for x in (m.group(2),m.group(3),m.group(4)) if x),None)
        if table is None or table.lower() not in schema:
            return None
        alias=m.group(5)
        if alias is None or alias.upper() in RESERVED:alias=table
        refs.append({"kind":m.group(1).upper(),"table":table.lower(),"alias":alias})
    return refs


def _one_to_many_child_of(parent:str,child:str,schema:dict)->bool:
    pk=schema[parent]["pk"]
    if len(pk)!=1:return False
    target_pk=pk[0][1].lower()
    return any(f["to_table"]==parent and f["to_col"]==target_pk
               for f in schema[child]["fk"])


def _single_fk_to_any_parent(child:str,parents:list[str],schema:dict)->bool:
    if not parents:return True
    for parent in parents:
        if parent==child:return False
        if not _one_to_many_child_of(parent,child,schema):
            return False
    return True


def _bounded_scalar(conn:sqlite3.Connection,sql:str)->int|None:
    start=time.monotonic()
    conn.set_progress_handler(lambda:1 if time.monotonic()-start>2.0 else 0,1000)
    try:
        res=conn.execute(sql).fetchall()
        if len(res)!=1 or len(res[0])!=1 or not isinstance(res[0][0],int):
            return None
        return res[0][0]
    except sqlite3.Error:
        return None
    finally:
        conn.set_progress_handler(None,0)


def decide(question:str,sql:str,conn:sqlite3.Connection)->dict:
    def out(state,reason,**kw):
        assert state in DECISIONS
        return {"decision":state,"reason":reason,"protocol":PROTOCOL,
                "whole_sql_correctness_not_inferred":True,**kw}
    if not isinstance(question,str) or not isinstance(sql,str) or len(sql)>10000:
        return out("ABSTAIN","INVALID_INPUT")
    clean=re.sub(r"'(?:''|[^'])*'","''",sql).strip().rstrip(";")
    if (UNSAFE.search(clean) or len(re.findall(r"(?i)\bSELECT\b",clean))!=1
        or re.search(r"(?i)\b(?:UPDATE|INSERT|DELETE|DROP|ALTER|PRAGMA|ATTACH)\b",clean)):
        return out("ABSTAIN","COMPLEX_OR_UNSAFE_SQL")
    match=COUNT_SELECT.search(clean)
    if match is None or len(re.findall(r"(?i)\bCOUNT\s*\(",clean))!=1:
        return out("ABSTAIN","NOT_SINGLE_SIMPLE_COUNT")
    schema=_schema(conn)
    target=_question_table(question,schema)
    if target is None:return out("ABSTAIN","ANSWER_UNIT_NOT_GROUNDED")
    refs=_table_refs(clean,schema)
    if not refs or refs[0]["kind"]!="FROM" or len({r["alias"].lower() for r in refs})!=len(refs):
        return out("ABSTAIN","SQL_TABLE_ALIAS_NOT_GROUNDED")
    if len(re.findall(r"(?i)\b(?:FROM|JOIN)\b",clean))!=len(refs):
        return out("ABSTAIN","INCOMPLETE_FROM_JOIN_PARSE")
    matching=[r for r in refs if r["table"]==target]
    if len(matching)!=1:return out("ABSTAIN","TARGET_UNIT_NOT_SINGLE_SOURCE_TABLE")
    unit=matching[0]
    primary=schema[target]["pk"]
    if len(primary)!=1:return out("ABSTAIN","TARGET_TABLE_HAS_NO_SIMPLE_PK")
    pk=primary[0][1]
    cnt=match.group(1).strip()
    if cnt.upper().startswith("DISTINCT"):
        col=cnt.split(None,1)[1]
        qualified=unit["alias"]+"."+pk
        if col.lower()==qualified.lower():
            return out("CONSISTENT","EXPLICIT_DISTINCT_TARGET_PRIMARY_KEY",
                       target_schema_table=target)
        return out("ABSTAIN","DISTINCT_EXPRESSION_NOT_PROVEN_TARGET_PK")
    assert cnt=="*"
    if len(refs)==1 and refs[0]["table"]==target:
        return out("CONSISTENT","COUNT_DIRECTLY_FROM_TARGET_TABLE",
                   target_schema_table=target)
    others=[r["table"] for r in refs if r["table"]!=target]
    if (refs[0]["table"]==target and len(others)==len(refs)-1
        and _single_fk_to_any_parent(target,others,schema)):
        return out("CONSISTENT","COUNT_CHILD_ROWS_JOINED_ONLY_TO_PK_PARENTS",
                   target_schema_table=target)
    # High-specificity conflict requires a documented child FK to answer
    # target, AND a *real* difference on the same WHERE/ON joined population.
    if len(others)==1 and _one_to_many_child_of(target,others[0],schema):
        # The sanitized SQL is ONLY for structural parsing. Never execute
        # it: changing literal strings would silently change WHERE filters.
        original_sql=sql.strip().rstrip(";")
        alternate=(
            original_sql[:match.start(1)]+"DISTINCT "+unit["alias"]+"."+pk
            +original_sql[match.end(1):])
        actual=_bounded_scalar(conn,original_sql)
        distinct_count=_bounded_scalar(conn,alternate)
        if actual is None or distinct_count is None:
            return out("ABSTAIN","PHYSICAL_SQL_UNSCORABLE")
        if distinct_count!=actual:
            return out("CONFLICT","ONE_TO_MANY_FK_PROVEN_DIFFERENT_ANSWER_UNIT",
                       target_schema_table=target,
                       changed_result_boolean=True)
        return out("ABSTAIN","ONE_TO_MANY_FK_BUT_NO_PHYSICAL_DIFFERENCE")
    return out("ABSTAIN","RELATION_UNIT_DIRECTION_UNPROVEN")


def one_row(question,sql,dbfile:Path):
    with sqlite3.connect(dbfile.as_uri()+"?mode=ro",uri=True) as conn:
        conn.execute("PRAGMA query_only=ON")
        return decide(question,sql,conn)
