from __future__ import annotations

from pathlib import Path

from .catalog import DatabaseCatalog
from .contracts import AgentResult, SQLCandidate
from .inference import run_guarded
from .literal_evidence import append_literal_evidence
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
        literal_grounding: bool = False,
    ) -> None:
        self.database = Path(database)
        self.catalog = DatabaseCatalog.from_sqlite(self.database)
        self.provider = provider
        self.max_repairs = max_repairs
        self.max_rows = max_rows
        self.semantic_revision = semantic_revision
        self.literal_grounding = literal_grounding

    def ask(
        self,
        question: str,
        *,
        evidence: str | None = None,
        initial_candidate: SQLCandidate | None = None,
    ) -> AgentResult:
        schema_context = self.catalog.schema_text()
        if self.literal_grounding:
            schema_context = append_literal_evidence(
                self.catalog, question, schema_context=schema_context
            )
        return run_guarded(
            database=self.database,
            provider=self.provider,
            question=question,
            schema_context=schema_context,
            evidence=evidence,
            initial_candidate=initial_candidate,
            max_repairs=self.max_repairs,
            max_rows=self.max_rows,
            semantic_revision=self.semantic_revision,
        )
