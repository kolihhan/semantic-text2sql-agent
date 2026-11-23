"""Exploratory BIRD global-cardinality witness, no model, no benchmark labels.

Many valid SQLite SQLs are semantically wrong when a *global number of
entities* is asked but SQL groups by that entity and emits one count per group.
This checker accepts only a narrow, physically proven semantic contradiction:
SELECT COUNT(DISTINCT entity_key) ... GROUP BY entity_key HAVING ...
returns at least two independent scalar '1' rows, whereas the question asks
for one overall number. Grouped key equality and all original HAVING filters
are preserved; unsupported/ambiguous syntax abstains.

IMPORTANT: physical differential evidence is not gold-answer equivalence.
This is research-only; no rewriting production SQL without independent BIRD EX.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3

_IDENTIFIER = r"[A-Za-z_][A-Za-z_0-9]*"
_KEY = rf"{_IDENTIFIER}(?:\.{_IDENTIFIER})?"
_QUERY = re.compile(
    rf"^\s*SELECT\s+COUNT\s*\(\s*DISTINCT\s+(?P<key>{_KEY})\s*\)\s*"
    rf"(?:AS\s+{_IDENTIFIER}\s+)?"
    rf"FROM\b(?P<source>.+?)"
    rf"\bGROUP\s+BY\s+(?P<group>{_KEY})\s+HAVING\b(?P<having>.+?)\s*;?\s*$",
    flags=re.I | re.S,
)
_FORBIDDEN = re.compile(
    r"\b(?:SELECT|WITH|UNION|EXCEPT|INTERSECT|OVER|WINDOW|LIMIT|ORDER\s+BY|RETURNING)\b",
    flags=re.I,
)
_COUNT_REQUEST = re.compile(r"\b(?:how many|number of|total number of)\b", re.I)
_MULTI_OUTPUT = re.compile(
    r"\b(?:for each|per (?:each|every|customer|group|year|month)|"
    r"grouped by|breakdown|list (?:them|each)|indicate each|"
    r"show each|which (?:specific|individual))\b", re.I
)


@dataclass(frozen=True)
class GroupedScalarWitness:
    original_sql: str
    proposed_sql: str
    count_per_group: tuple[int, ...]
    question_contract: str


def _open_readonly(database: str | Path) -> sqlite3.Connection:
    path=Path(database).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    conn=sqlite3.connect(path.as_uri()+"?mode=ro",uri=True,timeout=2)
    conn.execute("PRAGMA query_only=ON")
    budget=[0]
    def limit():
        budget[0]+=1
        return 1 if budget[0]>300 else 0
    conn.set_progress_handler(limit, 10000)
    return conn


def find_global_cardinality_contradiction(
    database: str | Path, question: str, sql: str
) -> GroupedScalarWitness | None:
    if (not _COUNT_REQUEST.search(question)
            or _MULTI_OUTPUT.search(question)
            or not isinstance(sql, str)):
        return None
    match=_QUERY.fullmatch(sql.strip())
    if match is None:
        return None
    key=match["key"].casefold()
    group=match["group"].casefold()
    if key!=group:
        return None
    if (_FORBIDDEN.search(match["source"])
            or _FORBIDDEN.search(match["having"])):
        return None
    # A nontrivial aggregate-level HAVING selects qualifying entity groups.
    # A tautological HAVING COUNT(DISTINCT same_key)>0 is insufficient evidence.
    if re.search(r"\b(?:SUM|AVG|MIN|MAX)\s*\(",match["having"],re.I) is None:
        return None
    # Avoid lexically or semantically ambiguous multiple group expressions.
    if re.search(r",|\bGROUP\s+BY\b|\bHAVING\b", match["having"], re.I):
        # Commas in SQL functions are valid, but this narrow version abstains.
        return None
    old=sql.strip().rstrip(";").strip()
    proposal="SELECT COUNT(*) AS number_of_entities FROM ("+old+") AS qualified_entities"
    try:
        conn=_open_readonly(database)
    except (sqlite3.Error,OSError,ValueError):
        return None
    try:
        # Fetch only a bounded witness, rather than full expensive answer.
        observed=conn.execute(old).fetchmany(3)
        if (len(observed)<2
                or any(len(row)!=1 or type(row[0]) is not int or row[0]!=1
                       for row in observed)):
            return None
        candidate=conn.execute(proposal).fetchone()
        if (candidate is None or len(candidate)!=1
                or type(candidate[0]) is not int or candidate[0]<2):
            return None
    except (sqlite3.Error,ValueError):
        return None
    finally:
        conn.close()
    return GroupedScalarWitness(
        original_sql=sql,proposed_sql=proposal,
        count_per_group=tuple(row[0] for row in observed),
        question_contract="one global entity count, not one count per grouped entity",
    )
