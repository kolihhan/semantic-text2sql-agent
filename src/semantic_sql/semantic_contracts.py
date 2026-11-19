"""Gold-free deterministic semantic SQL checks, research-only opt-in.

High-confidence contradictions may trigger the existing bounded repair path.
Uncertain join cardinality and relationship signals are advisory ONLY.
Default Guarded inference behavior and weights are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class SemanticConcern:
    code: str
    message: str
    blocking: bool


def outer_select(sql: str) -> str | None:
    """Get top-level SELECT projection, abstain on WITH/unsupported query."""
    head=re.match(r"(?is)^\s*SELECT\b", sql)
    if not head:
        return None
    depth=0
    quote=None
    i=head.end()
    while i<len(sql):
        c=sql[i]
        if quote is not None:
            if c==quote:
                if i+1<len(sql) and sql[i+1]==quote:
                    i+=2
                    continue
                quote=None
            i+=1
            continue
        if c in ("'", '"', chr(96)):
            quote=c
        elif c=="[":
            quote="]"
        elif c=="(":
            depth+=1
        elif c==")":
            depth-=1
            if depth<0:
                return None
        elif depth==0 and sql[i:i+4].casefold()=="from":
            previous=sql[i-1] if i>0 else " "
            following=sql[i+4] if i+4<len(sql) else " "
            if not (previous.isalnum() or previous=="_" or
                    following.isalnum() or following=="_"):
                return sql[head.end():i].strip()
        i+=1
    return None


def projection_columns(projection: str) -> tuple[str, ...]:
    depth=0
    quote=None
    start=0
    i=0
    parts=[]
    while i<len(projection):
        c=projection[i]
        if quote is not None:
            if c==quote:
                if i+1<len(projection) and projection[i+1]==quote:
                    i+=2
                    continue
                quote=None
        elif c in ("'",'"',chr(96)):
            quote=c
        elif c=="[":
            quote="]"
        elif c=="(":
            depth+=1
        elif c==")":
            depth=max(depth-1,0)
        elif c=="," and depth==0:
            parts.append(projection[start:i].strip())
            start=i+1
        i+=1
    parts.append(projection[start:].strip())
    return tuple(parts)


def detect_semantic_concerns(question: str, sql: str) -> tuple[SemanticConcern, ...]:
    """Uses only question and proposed SQL; zero model calls, no gold or labels."""
    if not isinstance(question,str) or not isinstance(sql,str):
        return ()
    projection=outer_select(sql)
    if not projection:
        return ()
    q=question.casefold()
    s=sql.casefold()
    p=re.sub(r"'(?:''|[^'])*'", "''",projection).casefold()
    found=[]
    def add(code: str,message: str,blocking: bool):
        if code not in {x.code for x in found}:
            found.append(SemanticConcern(code,message,blocking))

    if re.search(r"\bgrowth\s+rate\b",q):
        if ("/" not in p and not re.search(
            r"\b(?:growth_?rate|percentage_?change)\b",p)):
            add("growth_rate_not_computed",
                "Question asks for a GROWTH RATE but SQL projects no division. "
                "Compute the rate rather than returning numerator and denominator "
                "separately; protect against a zero denominator.",True)

    if re.search(r"\bage(?:\s+group|\s+bracket|\s+range|\s+distribution)?\b",q):
        if re.search(
            r"strftime\s*\(\s*'%y'\s*,\s*[^)]+\)\s*-\s*"
            r"strftime\s*\(\s*'%y'\s*,\s*'1900(?:-01-01)?'\s*\)",s):
            add("age_wrong_1900_reference",
                "SQL calculates age as birth year minus 1900. "
                "Use reference/current year minus birth year and preserve age bands.",True)

    year_range=re.search(
        r"\b((?:19|20)\d{2})\s*[-–—/]\s*((?:19|20)\d{2})\b",q)
    if year_range and "season" in q:
        first,second=year_range.groups()
        if (re.search(r"\bseason\s*=\s*['\"]?"+first+r"\b",s) and
            second not in s and str(int(second)+1) not in s):
            add("two_year_season_collapsed",
                "Question specifies season "+first+"–"+second+
                " but SQL only tests season="+first+". "
                "Check the true stored season encoding; retain both years.",True)

    from_match=re.search(r"\bFROM\s+([\w]+)",sql,flags=re.I)
    main=from_match.group(1).casefold() if from_match else ""
    joined=[x.casefold() for x in re.findall(
        r"\bJOIN\s+([\w]+)",sql,flags=re.I)]
    if re.search(r"\bCOUNT\s*\(\s*\*\s*\)",projection,flags=re.I):
        for table in joined:
            plural=table[:-1]+"ies" if table.endswith("y") else table+"s"
            count_request=bool(re.search(
                r"\bhow\s+many\b.{0,80}?\b"+re.escape(plural)+r"\b",q))
            ratio_request=bool(re.search(
                r"\bpercent(?:age)?\s+of\b.{0,60}?\b"+re.escape(plural)+r"\b",q))
            if table!=main and (count_request or ratio_request):
                add("possible_join_fanout_grain",
                    "Question targets "+plural+" but COUNT(*) is over "+
                    main+" joined to "+table+
                    ". Check entity cardinality and whether COUNT(DISTINCT target key) is needed; "
                    "a one-to-one join may already be correct.",False)
                break

    if ("connected to" in q and not re.search(
        r"\b(?:JOIN|EXISTS|INTERSECT|UNION)\b",sql,flags=re.I)):
        add("connectivity_relation_maybe_missing",
            "Question asks for connected entities but SQL has no explicit "
            "join/set relation. Check for an actual relationship or embedded link.",False)

    if (re.search(r"\bwith\s+their\s+ids?\b",q) and
        len(projection_columns(projection))==1 and
        re.fullmatch(r"(?:distinct\s+)?[\w.]*id\b(?:\s+as\s+\w+)?",
                     projection_columns(projection)[0],flags=re.I)):
        add("entity_and_id_projection_maybe_incomplete",
            "Question asks for entities WITH IDs, while SQL only projects "
            "an ID. Confirm required display columns.",False)

    return tuple(found)
