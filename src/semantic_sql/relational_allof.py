"""Conservative relational ALL-of counterexample, BIRD research only.

When a question explicitly requires BOTH distinct child labels, SQL aggregating
parents over an IN (label1,label2) has UNION semantics, not INTERSECTION.
Require real SQLite FK/parent PK, a simple exact SQL AST shape, both literals
quoted and grounded verbatim in the question, AND a physical counterexample.
Do not rewrite ambiguous, compound or unrelated SQL. No benchmark gold/LLM.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3

# Explicitly limit to a single SELECT, two-table FK join and ONE child label
# predicate. An unsupported shape is never a reason to mutate the query.
_IDENT = r"[A-Za-z_][A-Za-z_0-9]*"
_SRC = rf"(?P<parent>{_IDENT})\s+(?:AS\s+)?(?P<pa>{_IDENT})"
_CHILD = rf"(?P<child>{_IDENT})\s+(?:AS\s+)?(?P<ca>{_IDENT})"
_SQL = re.compile(
    rf"^\s*SELECT\s+COUNT\s*\(\s*DISTINCT\s+(?P=pa)\.(?P<pk>{_IDENT})\s*\)"
    rf"\s+FROM\s+{_SRC}\s+JOIN\s+{_CHILD}"
    rf"\s+ON\s+(?P<left>{_IDENT}\.{_IDENT})\s*=\s*(?P<right>{_IDENT}\.{_IDENT})"
    rf"\s+WHERE\s+(?P<filter_alias>{_IDENT})\.(?P<label>{_IDENT})\s+IN\s*"
    rf"\(\s*'(?P<first>[^']{{1,60}})'\s*,\s*'(?P<second>[^']{{1,60}})'\s*\)"
    rf"\s*;?\s*$",
    re.I,
)

# The parent alias backreference must be checked after table alias binding,
# so selection uses a simple independent primary-projection prefix.
_PREFIX = re.compile(rf"^\s*SELECT\s+COUNT\s*\(\s*DISTINCT\s+(?P<pa>{_IDENT})\.(?P<pk>{_IDENT})\s*\)",re.I)
_FROM = re.compile(
    rf"^\s*FROM\s+{_SRC}\s+JOIN\s+{_CHILD}"
    rf"\s+ON\s+(?P<left>{_IDENT}\.{_IDENT})\s*=\s*(?P<right>{_IDENT}\.{_IDENT})"
    rf"\s+WHERE\s+(?P<fa>{_IDENT})\.(?P<label>{_IDENT})\s+IN\s*"
    rf"\(\s*'(?P<v1>[^']{{1,60}})'\s*,\s*'(?P<v2>[^']{{1,60}})'\s*\)"
    rf"\s*;?\s*$",
    re.I
)


@dataclass(frozen=True)
class Counterexample:
    original: str
    alternative: str
    question_obligation: str
    parent: str
    child: str
    union_count: int
    intersection_count: int


def _q(identifier: str) -> str:
    if re.fullmatch(_IDENT,identifier) is None:
        raise ValueError("not a simple SQL identifier")
    return '"' + identifier + '"'


def _read_only_connection(database: Path) -> sqlite3.Connection:
    if not database.is_file():
        raise FileNotFoundError(database)
    con=sqlite3.connect(database.resolve().as_uri()+"?mode=ro",uri=True,timeout=2)
    con.execute("PRAGMA query_only = ON")
    con.set_progress_handler(lambda: 1, 500_000)
    return con


def find_allof_counterexample(database: str | Path, question: str, sql: str) -> Counterexample | None:
    """Return an *observed* and fully grounded UNION-vs-intersection witness."""
    if not re.search(r"\bboth\b", question, flags=re.I):
        return None
    if re.search(r"(?is)\b(with|union|except|intersect|group\s+by|having|limit|order\s+by)\b", sql):
        return None
    prefix=_PREFIX.match(sql)
    if prefix is None:
        return None
    rest=sql[prefix.end():]
    m=_FROM.fullmatch(rest)
    if m is None:
        return None
    vals=(m["v1"],m["v2"])
    if vals[0].casefold()==vals[1].casefold() or not all(re.search(r"(?i)(?<!\w)"+re.escape(v)+r"(?!\w)",question) for v in vals):
        return None
    if not (prefix["pa"].casefold()==m["pa"].casefold()
            and m["fa"].casefold()==m["ca"].casefold()):
        return None
    if (m["parent"].casefold()==m["child"].casefold()
            or prefix["pa"].casefold()==m["ca"].casefold()):
        return None
    left,right=m["left"].casefold(),m["right"].casefold()
    expected={
        prefix["pa"].casefold()+"."+prefix["pk"].casefold(),
        m["ca"].casefold()+"."}
    if left==right or prefix["pa"].casefold()+"."+prefix["pk"].casefold() not in (left,right):
        return None
    child_col=(right if left==prefix["pa"].casefold()+"."+prefix["pk"].casefold() else left)
    if not child_col.startswith(m["ca"].casefold()+"."):
        return None
    child_fk=child_col.split(".")[1]
    database=Path(database)
    con=_read_only_connection(database)
    try:
        tables={name.casefold():name for (name,) in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
        parent=tables.get(m["parent"].casefold())
        child=tables.get(m["child"].casefold())
        if not parent or not child:
            return None
        keys=[(r[1],r[5]) for r in con.execute("PRAGMA table_info("+_q(parent)+")")]
        parent_pk=[name for name,seq in keys if seq>0]
        if len(parent_pk)!=1 or parent_pk[0].casefold()!=prefix["pk"].casefold():
            return None
        fks=con.execute("PRAGMA foreign_key_list("+_q(child)+")").fetchall()
        matching=[r for r in fks if r[2].casefold()==parent.casefold()
                  and r[3].casefold()==child_fk
                  and r[4].casefold()==prefix["pk"].casefold()]
        if len(matching)!=1:
            return None
        columns={row[1].casefold() for row in con.execute("PRAGMA table_info("+_q(child)+")")}
        if m["label"].casefold() not in columns:
            return None
        pa,ca=m["pa"],m["ca"]
        pkey=_q(pa)+"."+_q(prefix["pk"])
        cattr=_q(ca)+"."+_q(m["label"])
        # Build SQL from verified identifiers, not user-entered arbitrary SQL.
        fragment=(" FROM "+_q(parent)+" AS "+_q(pa)+" JOIN "+_q(child)+" AS "+_q(ca)
                  +" ON "+pkey+" = "+_q(ca)+"."+_q(child_fk)
                  +" WHERE "+cattr+" IN (?, ?)")
        union_sql="SELECT COUNT(DISTINCT "+pkey+")"+fragment
        intersect_sql=("SELECT COUNT(*) FROM (SELECT "+pkey+fragment
                       +" GROUP BY "+pkey+" HAVING COUNT(DISTINCT "+cattr+") = 2)")
        intersection=con.execute(intersect_sql,vals).fetchone()[0]
        union=con.execute(union_sql,vals).fetchone()[0]
        if union<=intersection or intersection<0:
            return None
        # Values already exist in the original literal pair; safe escaping.
        quoted=["'"+v.replace("'","''")+"'" for v in vals]
        literal_fragment=fragment.replace("?, ?",", ".join(quoted))
        alternative=("SELECT COUNT(*) FROM (SELECT "+pkey+literal_fragment
                     +" GROUP BY "+pkey+" HAVING COUNT(DISTINCT "+cattr+") = 2)")
        return Counterexample(
            original=sql,alternative=alternative,
            question_obligation="both labels must apply to the SAME parent entity",
            parent=parent,child=child,
            union_count=int(union),intersection_count=int(intersection),
        )
    except (sqlite3.Error,ValueError):
        return None
    finally:
        con.close()
