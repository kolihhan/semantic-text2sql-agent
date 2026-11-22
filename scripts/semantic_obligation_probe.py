"""Gold-blind semantic obligations for executable Text2SQL candidates.

Research-only, does not execute or rewrite SQL. All flags are potential
question-to-SQL mismatches; never treat them as correctness proof.
"""
from __future__ import annotations

import re

_ID_REQ = re.compile(
    r"""\b(?:id|identifier)\b\s*(?:is|equals|=|:|#)?\s*['"]?"""
    r"""(?P<value>\d{1,16}|(?=[A-Za-z0-9_-]*\d)[A-Za-z][A-Za-z0-9_-]{3,38})['"]?(?![\w-])""",
    re.I,
)
_NAMED_ENTITY = re.compile(
    r"""\b(?:for|with|of)\s+(?:the\s+)?(?P<entity>[A-Za-z][A-Za-z0-9_]*)\s+"""
    r"""['"]?(?P<value>(?=[A-Za-z0-9_-]*\d)[A-Za-z][A-Za-z0-9_-]{3,38})['"]?(?![\w-])""",
    re.I,
)
_SQL_PRED = re.compile(
    r"""\b(?:(?P<alias>[A-Za-z_]\w*)\s*\.\s*)?(?P<column>"?[A-Za-z_]\w*"?)"""
    r"""\s*(?P<op>=|LIKE)\s*(?P<quote>['"]?)(?P<value>[A-Za-z0-9_-]{1,45})(?P<wildcard>%?)(?P=quote)""",
    re.I,
)
_TABLE = re.compile(
    r"""\b(?P<action>FROM|JOIN)\s+(?P<table>"?[A-Za-z_]\w*"?)"""
    r"""(?:\s+(?:AS\s+)?(?!(?:ON|WHERE|INNER|LEFT|RIGHT|JOIN|GROUP|ORDER|LIMIT|HAVING)\b)(?P<alias>[A-Za-z_]\w*))?""",
    re.I,
)
_GROWTH = re.compile(
    r"\b(?:growth rate|percentage change|percent change|percentage increase|percent increase)\b",
    re.I,
)
_COUNT = re.compile(r"\bCOUNT\s*\(\s*\*\s*\)", re.I)
_COUNT_DISTINCT = re.compile(r"\bCOUNT\s*\(\s*DISTINCT\b", re.I)
_COUNT_TARGET = re.compile(r"\b(?:how many|number of)\b", re.I)
_GROUP = re.compile(r"\b(?:GROUP\s+BY|DISTINCT|HAVING)\b", re.I)


def _key_column(name: str) -> bool:
    col=name.strip('"').casefold()
    return col=="id" or col.endswith("_id")


def _norm_plural(value: str) -> str:
    v=value.casefold().strip('"')
    if v.endswith("ies"):
        return v[:-3]+"y"
    return v[:-1] if v.endswith("s") and not v.endswith("ss") else v


def detect_obligations(question: str, sql: str) -> dict:
    """Returns flags and abstentions; cannot examine publisher labels/answers."""
    flags=[]
    skip=[]
    if not isinstance(question,str) or not isinstance(sql,str):
        raise ValueError("question and SQL must be text")
    statement=sql.strip()
    if not statement.upper().startswith(("SELECT ", "WITH ", "SELECT\n", "WITH\n")):
        return {"flags":[], "abstentions":["NOT_A_READONLY_QUERY"]}
    # A source ID might be numeric or long alpha-numeric: the value is part of
    # the question, never constructed from publisher or evaluation answers.
    id_values={m.group("value").casefold() for m in _ID_REQ.finditer(question)}
    predicates=list(_SQL_PRED.finditer(sql))
    for value in sorted(id_values):
        filtered=[p for p in predicates if p.group("value").casefold()==value]
        if not filtered:
            skip.append("ID_VALUE_NOT_IN_SUPPORTED_SQL_PREDICATE")
            continue
        if any(_key_column(p.group("column")) for p in filtered):
            continue
        flags.append("EXPLICIT_ID_BOUND_TO_NON_ID_ATTRIBUTE")

    # High-specificity entity literal: "for bond TR001_2_4" should not
    # be filtered by "bond_type" when the request names a particular bond.
    aliases={}
    tables=[]
    for m in _TABLE.finditer(sql):
        name=m.group("table").strip('"').casefold()
        alias=(m.group("alias") or name).casefold()
        aliases[alias]=name
        aliases[name]=name
        tables.append((m.group("action").upper(),name))
    for m in _NAMED_ENTITY.finditer(question):
        entity=_norm_plural(m.group("entity"))
        token=m.group("value").casefold()
        if entity not in {table for _,table in tables}:
            continue
        filtered=[
            p for p in predicates
            if p.group("value").casefold()==token
            and (not p.group("alias") or aliases.get(p.group("alias").casefold())==entity)
        ]
        if filtered and all(not _key_column(p.group("column")) for p in filtered):
            flags.append("NAMED_ENTITY_TOKEN_FILTERED_BY_NONKEY")
    if _GROWTH.search(question):
        # SQL can return data for a client to compute later; this is a warning.
        if "/" not in sql and not re.search(r"\b(?:PERCENT_CHANGE|GROWTH_RATE)\s*\(",sql,re.I):
            flags.append("GROWTH_RATE_WITHOUT_DIVISION")

    # Parent-level count on child-joined rows can duplicate the parent.
    # Only a hypothesis: must be tested against physical data later.
    if _COUNT.search(sql) and not _COUNT_DISTINCT.search(sql) and not _GROUP.search(sql) and _COUNT_TARGET.search(question) and len(tables)>=2:
        src=tables[0][1]
        joined={t for _,t in tables[1:]}
        qwords={_norm_plural(x) for x in re.findall(r"[A-Za-z_]+",question)}
        if src not in qwords and any(t in qwords for t in joined):
            flags.append("PARENT_COUNT_ON_JOINED_CHILD_ROWS")

    return {"flags":sorted(set(flags)), "abstentions":sorted(set(skip))}
