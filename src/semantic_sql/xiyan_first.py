"""Optional XiYan-first Text-to-SQL runner with Qwen guarded failover.

Only deterministic preflight, actual SQLite execution failure, or truncation
causes failover. Neither gold SQL nor generated-answer equality is consulted.
Experimental opt-in: the default SemanticSQLService is unchanged.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .catalog import DatabaseCatalog
from .contracts import AgentResult, SQLCandidate, StageRecord
from .direct import _strip_fence
from .inference import run_guarded
from .providers import ModelProvider


def xiyan_native_prompt(
    *, question: str, schema_context: str, evidence: str | None,
) -> tuple[str, str]:
    # Preserve the independently measured XiYan native prompt exactly.
    user = (
        "你是一名SQLite专家，现在需要阅读并理解下面的【数据库schema】描述，"
        "以及可能用到的【参考信息】，并运用SQLite知识生成sql语句回答【用户问题】。\n"
        f"【用户问题】\n{question}\n\n"
        f"【数据库schema】\n{schema_context}\n\n"
        f"【参考信息】\n{evidence or ''}\n\n"
        f"【用户问题】\n{question}\n\n"
        "只返回一条只读SQLite SQL，不要解释。\n"
        + chr(96) * 3 + "sql"
    )
    return "", user


class XiYanFirstSQLService:
    def __init__(
        self, database: str | Path, *,
        xiyan_provider: ModelProvider,
        fallback_provider: ModelProvider,
        max_rows: int = 100,
        fallback_repairs: int = 2,
    ) -> None:
        if max_rows <= 0 or fallback_repairs < 0:
            raise ValueError("max_rows must be positive and fallback_repairs nonnegative")
        self.database = Path(database)
        self.schema = DatabaseCatalog.from_sqlite(self.database).schema_text()
        self.xiyan_provider = xiyan_provider
        self.fallback_provider = fallback_provider
        self.max_rows = max_rows
        self.fallback_repairs = fallback_repairs

    def ask(
        self, question: str, *,
        evidence: str | None = None,
        schema_context: str | None = None,
    ) -> AgentResult:
        schema = self.schema if schema_context is None else schema_context
        reason = "primary_failed"
        try:
            system, user = xiyan_native_prompt(
                question=question, schema_context=schema, evidence=evidence,
            )
            generated = self.xiyan_provider.complete_text(system=system, user=user)
            primary = run_guarded(
                database=self.database,
                provider=self.xiyan_provider,
                question=question,
                schema_context=schema,
                evidence=evidence,
                initial_candidate=SQLCandidate(_strip_fence(generated)),
                max_repairs=0,
                max_rows=self.max_rows,
            )
            if primary.status == "ok" and not primary.truncated:
                return replace(primary, stages=(
                    StageRecord("route", "xiyan_primary_valid"), *primary.stages,
                ))
            reason = "primary_truncated" if primary.truncated else primary.status
        except Exception as exc:
            # Model transport/candidate extraction failure is not a correctness verdict.
            # Fail over safely, recording only the exception type, not possibly
            # sensitive DB contents from exception text.
            reason = "primary_exception:" + type(exc).__name__
        fallback = run_guarded(
            database=self.database,
            provider=self.fallback_provider,
            question=question,
            schema_context=schema,
            evidence=evidence,
            max_repairs=self.fallback_repairs,
            max_rows=self.max_rows,
        )
        return replace(fallback, stages=(
            StageRecord("route", "qwen_fallback:" + reason), *fallback.stages,
        ))
