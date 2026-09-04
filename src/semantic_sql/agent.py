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
    ) -> None:
        self.database = Path(database)
        self.catalog = DatabaseCatalog.from_sqlite(self.database)
        self.provider = provider
        self.max_repairs = max_repairs
        self.max_rows = max_rows

    def ask(
        self,
        question: str,
        *,
        evidence: str | None = None,
        initial_candidate: SQLCandidate | None = None,
    ) -> AgentResult:
        return run_guarded(
            database=self.database,
            provider=self.provider,
            question=question,
            schema_context=self.catalog.schema_text(),
            evidence=evidence,
            initial_candidate=initial_candidate,
            max_repairs=self.max_repairs,
            max_rows=self.max_rows,
        )
