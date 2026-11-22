"""Research-only joint structural selector; no SQL generation or EX labels.

The obligation must be extracted from question+schema *without candidate SQL*.
This selector merely tests whether a candidate's AST covers a known, physically
grounded entity, relation and count-grain triple. A matched triple is NOT proof
of semantic correctness: extra filters or alternate legitimate SQL can remain.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .sql_semantics import ColumnRef, NormalizedJoin, parse_sql_semantics

import sqlglot
from sqlglot import exp


@dataclass(frozen=True)
class JointObligation:
    entity_column: ColumnRef
    join_edge: NormalizedJoin
    count_grain: ColumnRef
    distinct: bool
    confidence: str = "low"


@dataclass(frozen=True)
class JointDecision:
    status: Literal["KEEP_ORIGINAL", "SELECT_REPAIR", "ABSTAIN"]
    sql: str
    reason: str


def _edge_key(join: NormalizedJoin) -> frozenset[ColumnRef]:
    return frozenset((join.left, join.right))


def _covers_joint_obligation(sql: str, req: JointObligation) -> bool | None:
    """True = AST covers all obligations; False = clear mismatch; None = unknown."""
    try:
        parsed = parse_sql_semantics(sql)
    except (ImportError, ModuleNotFoundError):
        return None
    if parsed.status != "ok" or parsed.semantics is None:
        return None
    sem = parsed.semantics
    if sem.has_derived_sources:
        return None  # Cannot reason about CTE/subquery scope yet.
    if len(sem.joins) != 1 or len(sem.aggregations) != 1:
        return None  # Multiple relational paths or aggregates can be equivalent.
    agg = sem.aggregations[0]
    return (
        req.entity_column in sem.projections
        and _edge_key(sem.joins[0]) == _edge_key(req.join_edge)
        and agg.function == "COUNT"
        and agg.target == req.count_grain
        and agg.distinct is req.distinct
    )




def _preserves_unchecked_semantics(original: str, candidate: str) -> bool:
    """Never let a repair alter filters, grouping, ranking, or source tables.

    This is intentionally conservative. Only entity projection, aggregate
    expression and ON equality are variable; unsupported cases abstain.
    """
    try:
        a, b = (sqlglot.parse_one(s, read="sqlite") for s in (original, candidate))
    except Exception:
        return False
    if not isinstance(a, exp.Select) or not isinstance(b, exp.Select):
        return False
    for key in ("where", "having", "group", "order", "limit", "offset", "qualify"):
        x, y = a.args.get(key), b.args.get(key)
        if (x.sql(dialect="sqlite", normalize=True) if x is not None else None) != (
            y.sql(dialect="sqlite", normalize=True) if y is not None else None
        ):
            return False
    # Named source tables must be identical, not just a matching join edge.
    names = lambda tree: tuple(sorted(t.name.casefold() for t in tree.find_all(exp.Table)))
    return names(a) == names(b)

def select_joint_repair(
    original_sql: str, candidate_sqls: list[str], obligation: JointObligation,
) -> JointDecision:
    """Keep original unless a proven structural mismatch has one unambiguous repair.

    Not for deployment: evaluating only three requirements does not certify
    unexamined filters/projection semantics; requires independent DB-level EX.
    """
    abstain = lambda reason: JointDecision("ABSTAIN", original_sql, reason)
    if obligation.confidence != "high":
        return abstain("untrusted_question_schema_obligation")
    original = _covers_joint_obligation(original_sql, obligation)
    if original is None:
        return abstain("original_structure_unknown")
    if original:
        return JointDecision("KEEP_ORIGINAL", original_sql, "original_covers_joint_obligation")
    # Avoid selecting despite other structurally unknown/possibly correct SQLs.
    candidates = list(dict.fromkeys(s.strip() for s in candidate_sqls if s.strip()))
    statuses = [_covers_joint_obligation(sql, obligation) for sql in candidates]
    if any(status is None for status in statuses):
        return abstain("ambiguous_candidate_structure")
    supported = [sql for sql, status in zip(candidates, statuses) if status]
    if len(supported) != 1:
        return abstain("zero_or_multiple_joint_compatible_candidates")
    # Even among structurally valid candidates, unrelated clauses may have
    # changed. Reject rather than choosing a conveniently matching fragment.
    if not _preserves_unchecked_semantics(original_sql, supported[0]):
        return abstain("unverified_predicate_or_source_change")
    return JointDecision("SELECT_REPAIR", supported[0], "unique_joint_compatible_candidate")
