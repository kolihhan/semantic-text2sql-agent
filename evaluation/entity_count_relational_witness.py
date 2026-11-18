"""Gold-blind physical SQLite count-of-entities candidate witness.

The original question must explicitly begin with 'How many ENTITY'.
The original model SQL must be simple scalar COUNT(*). If exactly one
referenced table has a physical ENTITY ID/KEY column, and running the same
FROM/JOIN/WHERE with COUNT(DISTINCT entity key) actually changes result,
offer that alternative. Do not consult gold SQL, labels, teacher or model.
A physical difference alone does not prove semantic correctness.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3
import time

_SELECT_COUNT = re.compile(r"(?is)^\s*SELECT\s+COUNT\s*\(\s*\*\s*\)\s*(?:AS\s+[A-Za-z_]\w*\s*)?FROM\b")
_COUNT = re.compile(r"(?i)COUNT\s*\(\s*\*\s*\)")
_TABLE = re.compile(r'(?i)\b(?:FROM|JOIN)\s+(?:"([^"]+)"|\x60([^\x60]+)\x60|\[([^\]]+)\]|([A-Za-z_]\w*))\s*(?:\bAS\s+)?([A-Za-z_]\w*)?')
_QUESTION = re.compile(r"(?i)^\s*Question:\s*How many\s+(?:(?:of\s+the|the|total|distinct|different|unique)\s+)*([A-Za-z][A-Za-z_-]{2,})\b")
_RESERVED = {"WHERE","JOIN","INNER","LEFT","RIGHT","OUTER","CROSS","ON","GROUP","ORDER","HAVING","LIMIT","USING","UNION","AS","INDEXED","NOT"}
_NON_ENTITY = {"time","type","percentage","percent","ratio","year","way","kind"}


@dataclass(frozen=True)
class EntityCountWitness:
    changed: bool
    sql: str
    reason: str
    original_row_count: int | None = None
    distinct_entity_count: int | None = None
    entity_table: str | None = None
    entity_column: str | None = None


def _identifier(text: str) -> str:
    return '"' + text.replace('"','""') + '"'


def _singular(noun: str) -> str:
    noun=noun.casefold().replace("_","").replace("-","")
    if noun.endswith("ies") and len(noun)>4:
        return noun[:-3]+"y"
    if noun.endswith("sses"):
        return noun[:-2]
    if noun.endswith("s") and not noun.endswith("ss"):
        return noun[:-1]
    return noun


def _simple_scalar_count(sql: str) -> bool:
    if any(x in sql for x in ("--","/*","*/","\x00")):
        return False
    raw=sql.strip()
    if raw.endswith(";"):
        raw=raw[:-1]
    return bool(
        ";" not in raw and _SELECT_COUNT.match(raw)
        and len(re.findall(r"(?i)\bSELECT\b",raw))==1
        and len(_COUNT.findall(raw))==1
        and not re.search(r"(?i)\b(?:GROUP\s+BY|HAVING|UNION|INTERSECT|EXCEPT|OVER\s*\()",raw)
    )


def _table_aliases(sql: str) -> list[tuple[str,str]]:
    matched=[]
    for m in _TABLE.finditer(sql):
        table=next((x for x in m.groups()[:4] if x is not None),None)
        alias=m.group(5)
        if table is None:
            continue
        if alias and alias.upper() in _RESERVED:
            alias=None
        matched.append((table,alias or table))
    return matched


def _execute_one(con: sqlite3.Connection, sql: str, seconds: float=5.0):
    deadline=time.monotonic()+seconds
    con.set_progress_handler(lambda:int(time.monotonic()>deadline),2000)
    try:
        return con.execute(sql).fetchone()
    finally:
        con.set_progress_handler(None,0)


def propose_entity_count_witness(db:Path,user_prompt:str,original_sql:str)->EntityCountWitness:
    unchanged=lambda reason:EntityCountWitness(False,original_sql,reason)
    if not _simple_scalar_count(original_sql):
        return unchanged("not_single_simple_scalar_count")
    question=user_prompt.split("\nSchema context:\n",1)[0]
    if re.search(r"(?i)\b(?:list\s+(?:those|their)|and\s+list|what\s+(?:are|is))\b",question):
        return unchanged("compound_answer_request")
    m=_QUESTION.match(question)
    if not m:
        return unchanged("no_explicit_entity_noun")
    noun=_singular(m.group(1))
    if noun in _NON_ENTITY:
        return unchanged("not_entity_count_intent")
    refs=_table_aliases(original_sql)
    if not refs or len(refs)>6 or len({a.casefold() for _,a in refs})!=len(refs):
        return unchanged("ambiguous_or_missing_base_tables")
    if not db.is_file():
        return unchanged("original_database_missing")
    try:
        with sqlite3.connect(db.resolve().as_uri()+"?mode=ro",uri=True) as con:
            con.execute("PRAGMA query_only=ON")
            candidates=[]
            for table,alias in refs:
                columns=[r[1] for r in con.execute("PRAGMA table_info("+_identifier(table)+")")]
                for column in columns:
                    norm=re.sub(r"[^a-z0-9]","",column.casefold())
                    if norm in {noun+"id",noun+"key"}:
                        candidates.append((table,alias,column))
            if len(candidates)!=1:
                return unchanged("no_unique_entity_key_mapping")
            table,alias,column=candidates[0]
            expression=_identifier(alias)+"."+_identifier(column)
            changed=_COUNT.sub("COUNT(DISTINCT "+expression+")",original_sql,count=1)
            if changed==original_sql:
                return unchanged("no_change")
            base=_execute_one(con,original_sql)
            alt=_execute_one(con,changed)
            if not (base is not None and alt is not None and len(base)==1 and len(alt)==1 and isinstance(base[0],int) and isinstance(alt[0],int)):
                return unchanged("unscorable")
            if base[0]==alt[0]:
                return unchanged("no_physical_duplicate_entity_witness")
            if not 1<=alt[0]<base[0]:
                return unchanged("invalid_entity_count_witness")
            return EntityCountWitness(True,changed,"physical_duplicate_entity_row_witness",
                                      base[0],alt[0],table,column)
    except (sqlite3.Error,OSError,ValueError,TimeoutError):
        return unchanged("physical_sql_failed_closed")
