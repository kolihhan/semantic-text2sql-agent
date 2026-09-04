from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class SQLCandidate:
    sql: str
    attempt: int = 0


@dataclass(frozen=True)
class VerificationIssue:
    code: str
    message: str


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    issues: tuple[VerificationIssue, ...] = ()


@dataclass(frozen=True)
class StageRecord:
    name: str
    summary: str


@dataclass(frozen=True)
class AgentResult:
    status: Literal["ok", "verification_failed", "execution_failed"]
    question: str
    candidate: SQLCandidate | None = None
    verification: VerificationResult | None = None
    rows: tuple[tuple[Any, ...], ...] = ()
    columns: tuple[str, ...] = ()
    truncated: bool = False
    message: str = ""
    stages: tuple[StageRecord, ...] = field(default_factory=tuple)
