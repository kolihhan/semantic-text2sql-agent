from __future__ import annotations

from pathlib import Path

from .catalog import DatabaseCatalog
from .contracts import AgentResult, SQLCandidate
from .inference import run_guarded
from .providers import ModelProvider


class SemanticSQLService:
    def __init__(
        self,
        database: str | Path,
        *,
        provider: ModelProvider,
        max_repairs: int = 2,
        max_rows: int = 100,
        semantic_revision: bool = False,
        mode: str = "auto",
    ) -> None:
        self.database = Path(database)
        self.catalog = DatabaseCatalog.from_sqlite(self.database)
        self.provider = provider
        self.max_repairs = max_repairs
        self.max_rows = max_rows
        if mode not in {"auto", "structured", "guarded"}:
            raise ValueError(f"Unknown SQL engine mode: {mode}")
        if mode == "structured" and not callable(getattr(provider, "complete_messages", None)):
            raise ValueError("Structured mode requires a structured-capable model provider")
        if mode == "structured" and semantic_revision:
            raise ValueError("Structured mode and experimental semantic revision are mutually exclusive")
        self.mode = mode
        self.semantic_revision = semantic_revision

    def ask(
        self,
        question: str,
        *,
        evidence: str | None = None,
        initial_candidate: SQLCandidate | None = None,
    ) -> AgentResult:
        structured = (self.mode == "structured" or (
            self.mode == "auto"
            and callable(getattr(self.provider, "complete_messages", None))
            and not self.semantic_revision
            and initial_candidate is None
        ))
        if structured:
            if initial_candidate is not None:
                raise ValueError("Structured mode cannot accept an injected raw SQL candidate")
            from .structured import run_structured
            return run_structured(database=self.database, provider=self.provider,
                                  question=question, schema_context=self.catalog.native_ddl(),
                                  evidence=evidence, max_rows=self.max_rows)
        return run_guarded(
            database=self.database,
            provider=self.provider,
            question=question,
            schema_context=self.catalog.native_ddl(),
            evidence=evidence,
            initial_candidate=initial_candidate,
            max_repairs=self.max_repairs,
            max_rows=self.max_rows,
            semantic_revision=self.semantic_revision,
        )
