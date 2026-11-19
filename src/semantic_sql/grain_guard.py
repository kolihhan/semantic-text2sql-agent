"""Opt-in, fail-closed entity-grain correction in the ORIGINAL Guarded pipeline.

Works only on a single ungrouped SELECT with documented one-column physical
target-table PK, explicit entity noun in the question, a real JOIN, and
physical fanout after all original filters. Supports two proven shapes:
  (1) COUNT(*) of JOIN rows while question asks "how many <entity>"
  (2) SUM(CASE WHEN target condition THEN 1 ELSE 0 END)/COUNT(*)
      while question asks percentage of <entity>.
All evidence is SQL+physical DB only, no publisher reference, new model,
LLM judge, extra Qwen call, source/task ID heuristic, or hidden gold.
"""
from __future__ import annotations
from pathlib import Path
import re
import sqlite3
import time

COUNT_TARGET = re.compile(r"\bhow many\s+([A-Za-z_]+)\b",re.I)
PERCENT_TARGET = re.compile(
    r"\b(?:percentage|percent|proportion)\s+of\s+(?:the\s+)?([A-Za-z_]+)\b", re.I)
TABLE_PATTERN = re.compile(
    r"\b(?:FROM|JOIN)\s+([A-Za-z_]\w*)\s+(?:AS\s+)?([A-Za-z_]\w*)\b", re.I)
COUNT_PROJECTION = re.compile(
    r"(?is)^\s*SELECT\s+COUNT\s*\(\s*\*\s*\)(\s+AS\s+[A-Za-z_]\w*)?\s+FROM\s+")
PERCENT_PROJECTION = re.compile(
    r"(?is)^\s*SELECT\s+CAST\s*\(\s*SUM\s*\(\s*CASE\s+WHEN\s+(.+?)"
    r"\s+THEN\s+1\s+ELSE\s+0\s+END\s*\)\s+AS\s+(FLOAT|REAL|NUMERIC)"
    r"\s*\)\s*\*\s*100(?:\.0)?\s*/\s*COUNT\s*\(\s*\*\s*\)\s+FROM\s+")
SUSPICIOUS = re.compile(
    r"(?i)\b(WITH|GROUP\s+BY|HAVING|UNION|INTERSECT|EXCEPT|WINDOW|LIMIT|ORDER\s+BY|SELECT\s+DISTINCT)\b")
IDENT = re.compile(r"^[A-Za-z_]\w*$")


def singular(s: str) -> str:
    s=s.lower()
    if s.endswith("ies"):return s[:-3]+"y"
    if s.endswith("s") and not s.endswith("ss"):return s[:-1]
    return s


def _read_only_fanout(db: Path, suffix: str, alias: str, pk: str):
    # A read-only connection and short VM deadline. Any failure => no rewrite.
    conn=None
    try:
        conn=sqlite3.connect(db.resolve().as_uri()+"?mode=ro",uri=True,timeout=1)
        conn.execute("PRAGMA query_only=ON")
        began=time.monotonic()
        conn.set_progress_handler(lambda:int(time.monotonic()-began>3.0),3000)
        sql=(f'SELECT COUNT(*),COUNT(DISTINCT "{alias}"."{pk}") '+suffix)
        result=conn.execute(sql).fetchone()
        return (int(result[0]),int(result[1])) if result else None
    except (sqlite3.Error,OSError,TypeError,ValueError):
        return None
    finally:
        if conn:
            conn.set_progress_handler(None,0)
            conn.close()


def correct_entity_grain(database: str | Path, question: str, sql: str):
    """Return (SQL, reason). The default action is always no change.

    No classifier from benchmark-gold labels. Do not call on invalid SQL;
    the outer Guarded graph verifies/repairs first.
    """
    if not isinstance(question,str) or not isinstance(sql,str):
        return sql,"not_strings"
    original=sql.strip().rstrip(";").strip()
    if (not original.upper().startswith("SELECT ")
            or original.count(";")>0
            or SUSPICIOUS.search(original)
            or not re.search(r"\bJOIN\b",original,re.I)
            or len(re.findall(r"\bSELECT\b",original,re.I))!=1):
        return sql,"outside_single_ungrouped_join"
    is_count=bool(COUNT_PROJECTION.search(original))
    ratio=PERCENT_PROJECTION.search(original)
    if not is_count and not ratio:
        return sql,"not_supported_scalar_aggregate"
    target_match=(COUNT_TARGET.search(question) if is_count
                  else PERCENT_TARGET.search(question))
    if not target_match:
        return sql,"no_explicit_counted_entity"
    target=singular(target_match.group(1))
    tables=TABLE_PATTERN.findall(original)
    # In restricted single-SELECT mode, table=alias is always explicit.
    # Ambiguous aliases, CTEs, quoted physical names and self-joins abstain.
    matches=[(table,alias) for table,alias in tables if singular(table)==target]
    if len(matches)!=1 or len({alias.lower() for _,alias in tables})!=len(tables):
        return sql,"ambiguous_target_table"
    table,alias=matches[0]
    if not (IDENT.fullmatch(table) and IDENT.fullmatch(alias)):
        return sql,"unsupported_identifier"
    conn=None
    try:
        conn=sqlite3.connect(Path(database).resolve().as_uri()+"?mode=ro",uri=True)
        conn.execute("PRAGMA query_only=ON")
        column_info=conn.execute('PRAGMA table_info("'+table+'")').fetchall()
        keys=[r[1] for r in column_info if r[5]>0]
        if len(keys)!=1 or not IDENT.fullmatch(keys[0]):
            return sql,"no_single_primary_key"
        pk=keys[0]
    except (sqlite3.Error,OSError):
        return sql,"physical_key_error"
    finally:
        if conn:conn.close()
    # Original suffix begins at the top-level FROM (prechecked no subquery).
    m=re.search(r"\bFROM\b",original,re.I)
    if not m:
        return sql,"no_from"
    suffix=original[m.start():]
    observed=_read_only_fanout(Path(database),suffix,alias,pk)
    if observed is None or observed[0]<=observed[1]:
        return sql,"no_observed_join_fanout"
    fully_qualified=f'"{alias}"."{pk}"'
    if is_count:
        candidate=COUNT_PROJECTION.sub(
            lambda x: "SELECT COUNT(DISTINCT "+fully_qualified+")"+
                       (x.group(1) or "")+" FROM ",original,count=1)
    else:
        # Source-restricted: expression must be conditional 1/0; projected
        # numerator and denominator both counted at TARGET entity grain.
        condition=ratio.group(1).strip()
        # Defensive: require conditional expression references target alias;
        # otherwise per-target mapping may change the original question.
        if not re.search(r"\b"+re.escape(alias)+r"\.",condition,re.I):
            return sql,"numerator_not_target_scoped"
        candidate=PERCENT_PROJECTION.sub(
            "SELECT CAST(COUNT(DISTINCT CASE WHEN "+condition+" THEN "+
            fully_qualified+" END) AS FLOAT) * 100.0 / "+
            "COUNT(DISTINCT "+fully_qualified+") FROM ",
            original,count=1)
    if candidate==original:return sql,"no_change"
    return candidate,"target_entity_pk_grain_with_physical_fanout"
