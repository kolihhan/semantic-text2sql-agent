from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .contracts import AgentResult


class QueryService(Protocol):
    def ask(self, question: str, *, evidence: str | None = None) -> AgentResult: ...


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    evidence: str | None = None


def _demo_service() -> QueryService:
    from .agent import SemanticSQLService
    from .providers import DemoProvider

    root = Path(__file__).resolve().parents[2]
    return SemanticSQLService(root / "demo" / "demo.db", provider=DemoProvider())


def create_app(*, service: QueryService | None = None) -> FastAPI:
    app = FastAPI(title="Semantic Text-to-SQL Reliability", version="0.1.0")
    state: dict[str, QueryService | None] = {"service": service}

    def current_service() -> QueryService:
        if state["service"] is None:
            state["service"] = _demo_service()
        return state["service"]

    @app.get("/health")
    def health() -> dict[str, bool]:
        return {"ok": True, "service_loaded": state["service"] is not None}

    @app.post("/query")
    def query(request: QueryRequest):
        return asdict(current_service().ask(request.question, evidence=request.evidence))

    return app


app = create_app()
