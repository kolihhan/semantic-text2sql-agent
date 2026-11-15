"""Conservative source-only FK JOIN-key intervention on frozen Qwen SQL.

Never accesses gold SQL, gold tuples or task questions. Restricted to flat
single-SELECT equality JOINs, a unique verified physical FK and one matching
original endpoint. This is NOT a production default until paired strict EX
demonstrates transfer without regressions.
"""
from __future__ import annotations
from dataclasses import dataclass
import re
from semantic_sql.catalog import DatabaseCatalog

IDENT = r'(?:\x60[^\x60]+\x60|"[^"]+"|\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*)'
FIELD = rf'({IDENT})\s*\.\s*({IDENT})'
STOP = re.compile(r'\b(?:WHERE|GROUP\s+BY|HAVING|ORDER\s+BY|LIMIT|OFFSET)\b', re.I)
FROM = re.compile(
    rf'\bFROM\s+(?P<table>{IDENT})'
    rf'(?:\s+(?:AS\s+)?(?P<alias>{IDENT}))?'
    rf'(?=\s+(?:(?:INNER|LEFT|RIGHT|FULL|CROSS)\s+)?JOIN\b)', re.I
)
JOIN = re.compile(
    rf'\b(?:(?:INNER|LEFT|RIGHT|FULL|CROSS)\s+)?JOIN\s+'
    rf'(?P<table>{IDENT})(?:\s+(?:AS\s+)?(?P<alias>{IDENT}))?'
    rf'\s+ON\s+', re.I
)
EQUALITY = re.compile(rf'^\s*{FIELD}\s*=\s*{FIELD}\s*$', re.I|re.S)
KEYWORDS = {
    "WHERE", "JOIN", "INNER", "LEFT", "RIGHT", "FULL", "OUTER",
    "CROSS", "ON", "GROUP", "HAVING", "ORDER", "LIMIT", "OFFSET",
    "UNION", "AS"
}


def unquote(value: str) -> str:
    if len(value) >= 2 and value[0] in ('"', chr(96), '['):
        return value[1:-1]
    return value


def resolve(value: str, candidates) -> str | None:
    lowered=unquote(value).casefold()
    return next((v for v in candidates if v.casefold()==lowered),None)


@dataclass(frozen=True)
class JoinRewrite:
    sql: str
    changed: bool
    reason: str
    original_join: str | None=None
    physical_fk: str | None=None


def repair_one_unique_fk_join(sql: str, catalog: DatabaseCatalog) -> JoinRewrite:
    """Return unchanged SQL unless a single, source-anchored FK mismatch is proven."""
    def no(reason: str)->JoinRewrite:
        return JoinRewrite(sql,False,reason)
    if not isinstance(sql,str) or not sql.strip():
        return no("empty_sql")
    value=sql.strip()
    if (re.search(r'\b(?:WITH|UNION|INTERSECT|EXCEPT)\b',value,re.I)
        or len(re.findall(r'\bSELECT\b',value,re.I))!=1
        or value.count(";")>1 or (";" in value and not value.endswith(";"))
        or "--" in value or "/*" in value
        or re.search(r'\b(?:USING|NATURAL)\b',value,re.I)
        or re.search(r'\b(?:FROM|JOIN)\s*\(',value,re.I)):
        return no("complex_sql_refused")
    bases=list(FROM.finditer(value))
    joins=list(JOIN.finditer(value))
    if len(bases)!=1 or not 1<=len(joins)<=5:
        return no("unhandled_join_layout")
    first=bases[0]
    table=resolve(first.group("table"),catalog.tables)
    if table is None:
        return no("unknown_base_table")
    alias=unquote(first.group("alias") or first.group("table"))
    if alias.upper() in KEYWORDS:
        return no("reserved_alias")
    aliases={alias.casefold():table}
    changes=[]
    for index,match_join in enumerate(joins):
        new_table=resolve(match_join.group("table"),catalog.tables)
        if new_table is None:
            return no("unknown_join_table")
        next_alias=unquote(match_join.group("alias") or match_join.group("table"))
        if next_alias.upper() in KEYWORDS or next_alias.casefold() in aliases:
            return no("ambiguous_alias")
        aliases[next_alias.casefold()]=new_table
        start=match_join.end()
        end=joins[index+1].start() if index+1<len(joins) else len(value)
        between=value[start:end]
        terminator=STOP.search(between)
        if terminator:
            end=start+terminator.start()
            between=between[:terminator.start()]
        target=between.strip().rstrip(";").strip()
        expr=EQUALITY.fullmatch(target)
        if expr is None:
            continue
        a,ac,b,bc=expr.groups()
        alias_a,alias_b=unquote(a).casefold(),unquote(b).casefold()
        if alias_a==alias_b or alias_a not in aliases or alias_b not in aliases:
            continue
        table_a,table_b=aliases[alias_a],aliases[alias_b]
        if table_a==table_b or new_table not in (table_a,table_b):
            continue
        col_a=resolve(ac,catalog.tables[table_a].columns)
        col_b=resolve(bc,catalog.tables[table_b].columns)
        if col_a is None or col_b is None:
            continue
        current=frozenset(((table_a,col_a),(table_b,col_b)))
        links={
            frozenset(((c,cc),(p,pc)))
            for c,cc,p,pc in catalog.foreign_keys
            if {c.casefold(),p.casefold()}=={table_a.casefold(),table_b.casefold()}
        }
        if len(links)!=1:
            continue
        expected=next(iter(links))
        if expected==current or len(expected)!=2:
            continue
        if not (expected & current):
            continue
        by_table={table:column for table,column in expected}
        if set(by_table)!={table_a,table_b}:
            continue
        revised=f"{a}.{by_table[table_a]} = {b}.{by_table[table_b]}"
        changes.append((start,end,target,revised))
    if len(changes)!=1:
        return no("zero_or_multiple_safe_candidates")
    start,end,old,new=changes[0]
    fragment=value[start:end]
    found=re.search(re.escape(old),fragment)
    if found is None:
        return no("original_join_slice_drift")
    output=value[:start+found.start()]+new+value[start+found.end():]
    return JoinRewrite(output,output!=value,"one_endpoint_matches_unique_physical_fk",
                       old,new)
