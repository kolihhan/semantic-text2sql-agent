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
    return JointDecision("SELECT_REPAIR", supported[0], "unique_joint_compatible_candidate")
