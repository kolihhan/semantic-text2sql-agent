"""Conservative, zero-model-call COUNT(JOIN) entity-grain correction.

This is an opt-in research candidate, NOT production inference. No model calls,
labels, publisher SQL or reference results used. It deliberately abstains on
most SQL forms instead of guessing entity grain. New BIRD TRAIN physical DBs
must independently demonstrate strictly paired execution-match improvement.
"""
from __future__ import annotations
from dataclasses import dataclass
import re
from pathlib import Path
import sqlite3

from .execution import execute_readonly, SQLExecutionError, _open_readonly_connection
from .verifier import verify_sql_preflight

_IDENT=r"[A-Za-z_][A-Za-z_0-9]*"
_QUERY=re.compile(
    rf"^\s*SELECT\s+COUNT\s*\(\s*\*\s*\)\s+FROM\s+"
    rf"(?P<parent>{_IDENT})\s+(?:AS\s+)?(?P<pa>{_IDENT})\s+"
    rf"JOIN\s+(?P<child>{_IDENT})\s+(?:AS\s+)?(?P<ca>{_IDENT})\s+"
    rf"ON\s+(?P<l_alias>{_IDENT})\.(?P<l_col>{_IDENT})\s*=\s*"
    rf"(?P<r_alias>{_IDENT})\.(?P<r_col>{_IDENT})"
    rf"(?P<tail>\s+WHERE\s+.+?)?\s*;?\s*$",
    re.I | re.S
)

@dataclass(frozen=True)
class GrainProposal:
    changed: bool
    sql: str
    reason: str
    original_count: int | None = None
    distinct_count: int | None = None

def _entity_forms(table: str) -> set[str]:
    token=table.casefold().replace("_"," ").strip()
    if " " in token:
        # No multiword / compound entity inference from table names.
        return set()
    forms={token}
    if token.endswith("ies") and len(token)>4:
        forms.add(token[:-3]+"y")
    elif token.endswith("sses"):
        forms.add(token[:-2]) # businesses -> business
    elif token.endswith("s") and not token.endswith("ss") and len(token)>3:
        forms.add(token[:-1])
    else:
        forms.add(token+"s")
    return {x for x in forms if len(x)>2}

def _question_names_parent_entity(question: str, parent: str) -> bool:
    if not re.search(r"\bhow\s+many\b",question,re.I):
        return False
    noun_forms=_entity_forms(parent)
    if not noun_forms:
        return False
    # Only the explicit object near "how many", not any incidental mentions.
    return any(re.search(r"\bhow\s+many\s+(?:[\w'-]+\s+){0,4}"
                         +re.escape(noun)+r"\b",question,re.I)
               for noun in noun_forms)

def _safe_count(db: Path, sql: str) -> int | None:
    try:
        result=execute_readonly(db,sql,max_rows=2)
    except (SQLExecutionError,ValueError):
        return None
    if result.truncated or len(result.rows)!=1 or len(result.rows[0])!=1:
        return None
    value=result.rows[0][0]
    if isinstance(value,bool) or not isinstance(value,int):
        return None
    return value

def propose_join_entity_grain(db: str | Path, sql: str, question: str) -> GrainProposal:
    """Rewrite COUNT(*) into COUNT(DISTINCT parent.pk) only when all gates pass.

    Gates: exact one SELECT/COUNT(*)/single INNER JOIN, simple PK=FK equality,
    no subqueries/additional joins/group/sort, parent entity lexically requested
    near "how many", one physical declared primary key, true JOIN fanout, and
    both old and proposed SQL pass read-only physical SQLite execution.
    Never modifies a query when original count is already an entity count.
    """
    database=Path(db)
    def skip(reason):
        return GrainProposal(False,sql,reason)
    m=_QUERY.fullmatch(sql)
    if not m:
        return skip("unsupported_sql_form")
    g=m.groupdict()
    pa,ca=g["pa"],g["ca"]
    if pa.casefold()==ca.casefold():
        return skip("duplicate_alias")
    parent,child=g["parent"],g["child"]
    tail=g["tail"] or ""
    if re.search(r"\b(JOIN|SELECT|GROUP|HAVING|UNION|INTERSECT|EXCEPT|ORDER|LIMIT)\b",tail,re.I):
        return skip("complex_join_query")
    if not _question_names_parent_entity(question,parent):
        return skip("question_entity_not_explicit")
    l=(g["l_alias"].casefold(),g["l_col"])
    r=(g["r_alias"].casefold(),g["r_col"])
    valid_aliases={pa.casefold(),ca.casefold()}
    if {l[0],r[0]}!=valid_aliases:
        return skip("join_alias_mismatch")
    pk_column=l[1] if l[0]==pa.casefold() else r[1]
    child_key=r[1] if l[0]==pa.casefold() else l[1]
    try:
        connection, _blocked = _open_readonly_connection(database)
        try:
            # Tables and columns are strictly [A-Za-z_0-9] by parser.
            info=connection.execute(f"PRAGMA table_info({parent})").fetchall()
            child_info=connection.execute(f"PRAGMA table_info({child})").fetchall()
            primary=[row[1] for row in info if row[5] > 0]
            child_names={row[1] for row in child_info}
        finally:
            connection.close()
    except (sqlite3.Error,ValueError):
        return skip("physical_schema_unavailable")
    if (len(primary)!=1 or primary[0].casefold()!=pk_column.casefold()
        or child_key.casefold() not in {x.casefold() for x in child_names}):
        return skip("not_parent_primary_key_join")
    # Replace only the unique top-level COUNT(*), preserve the WHERE clause.
    replacement=f"COUNT(DISTINCT {pa}.{pk_column})"
    proposed=re.sub(r"\bCOUNT\s*\(\s*\*\s*\)",replacement,sql,count=1,flags=re.I)
    if proposed==sql or not verify_sql_preflight(database,proposed).ok:
        return skip("invalid_proposed_sql")
    original_count=_safe_count(database,sql)
    distinct_count=_safe_count(database,proposed)
    if original_count is None or distinct_count is None:
        return skip("physical_execution_invalid")
    if not (original_count>distinct_count>=0):
        return skip("no_verified_join_fanout")
    return GrainProposal(True,proposed,"question_targets_distinct_parent_entities",
                         original_count,distinct_count)
