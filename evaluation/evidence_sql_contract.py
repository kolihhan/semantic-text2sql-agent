"""Frozen, gold-blind BIRD External evidence -> candidate SQL constraint screen.

Same fixed Qwen3.5-9B. Never infer correctness from publisher reference SQL.
The literal rule is intentionally narrow: only an explicit evidence assignment
to a quoted nonnumeric value >=4 characters, never from the question alone.
The arithmetic rule is diagnostic only; do not auto-repair.
Rules must be frozen BEFORE running any original real model VAL64 comparison.
"""
from __future__ import annotations
from dataclasses import dataclass
import re

_EVIDENCE_MARKER="External evidence:\n"
_EXPLICIT_EQUALS=re.compile(r"(?<![<>=!])=\s*'((?:''|[^'])*)'",re.I)
_SQL_LITERALS=re.compile(r"'((?:''|[^'])*)'")
_EXPLICIT_DIVIDE=re.compile(r"\bdivid(?:e|ed|ing)\s*\(",re.I)
_RATIO_SQL=re.compile(r"(?<!/)/(?!/)")
_AVG_SQL=re.compile(r"\bAVG\s*\(",re.I)


def _normalize_literal(raw: str) -> str:
    return " ".join(raw.replace("''","'").strip().casefold().split())


def evidence_text(prompt: str) -> str:
    return prompt.split(_EVIDENCE_MARKER,1)[1] if _EVIDENCE_MARKER in prompt else ""


def explicit_evidence_values(prompt: str) -> tuple[str,...]:
    values=set()
    for raw in _EXPLICIT_EQUALS.findall(evidence_text(prompt)):
        v=_normalize_literal(raw)
        if len(v)<4 or re.search(r"\d",v) or v in {"null","none","true","false"}:
            continue
        values.add(v)
    return tuple(sorted(values))


def candidate_literals(sql: str) -> frozenset[str]:
    # SQLite quote escaping is represented by two apostrophes; SQL comments
    # or comments with fake literals are excluded to avoid false satisfaction.
    masked=re.sub(r"/\*.*?\*/"," ",sql,flags=re.S)
    masked=re.sub(r"--[^\r\n]*"," ",masked)
    return frozenset(_normalize_literal(s).strip("%") for s in _SQL_LITERALS.findall(masked))


def missing_explicit_evidence_values(prompt: str, sql: str) -> tuple[str,...]:
    evidence=explicit_evidence_values(prompt)
    observed=candidate_literals(sql)
    return tuple(v for v in evidence if v not in observed)


def evidence_division_contract(prompt: str, sql: str) -> int:
    """+1 satisfied, -1 missing expected arithmetic, 0 unsupported/abstain."""
    e=evidence_text(prompt)
    if not _EXPLICIT_DIVIDE.search(e):
        return 0
    if _RATIO_SQL.search(sql):
        return 1
    if ("average" in e.lower() or "avg" in e.lower()) and _AVG_SQL.search(sql):
        return 1
    return -1


@dataclass(frozen=True)
class EvidenceSignal:
    explicit_value_requirements: int
    missing_values: tuple[str,...]
    division_check: int

    @property
    def violation(self) -> bool:
        return bool(self.missing_values or self.division_check==-1)

    @property
    def covered(self) -> bool:
        return bool(self.explicit_value_requirements or self.division_check!=0)


def evidence_signal(prompt: str, sql: str) -> EvidenceSignal:
    return EvidenceSignal(
        explicit_value_requirements=len(explicit_evidence_values(prompt)),
        missing_values=missing_explicit_evidence_values(prompt,sql),
        division_check=evidence_division_contract(prompt,sql),
    )


def select_two_original_qwen_candidates(prompt: str, direct_sql: str,
                                         guarded_sql: str,
                                         direct_exec: bool, guarded_exec: bool) -> dict:
    """Diagnostic only: no new Qwen calls; never alters candidate text."""
    a=evidence_signal(prompt,direct_sql)
    b=evidence_signal(prompt,guarded_sql)
    can_switch=bool(direct_exec and guarded_exec and a.covered
                    and a.violation and b.covered and not b.violation)
    return {
        "switch_to_guarded":can_switch,
        "direct_covered":a.covered,"guarded_covered":b.covered,
        "direct_violates":a.violation,"guarded_violates":b.violation,
        "direct_missing_evidence_values":len(a.missing_values),
        "guarded_missing_evidence_values":len(b.missing_values),
        "direct_division_check":a.division_check,
        "guarded_division_check":b.division_check,
    }
