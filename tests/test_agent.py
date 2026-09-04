from pathlib import Path

from semantic_sql.agent import SemanticSQLService
from semantic_sql.contracts import SQLCandidate
from semantic_sql.providers import DemoProvider


ROOT = Path(__file__).parents[1]
DB = ROOT / "demo" / "demo.db"


class SequencedProvider:
    def __init__(self, *sqls):
        self.sqls = iter(sqls)
        self.text_calls = 0

    def complete_text(self, *, system: str, user: str) -> str:
        self.text_calls += 1
        return next(self.sqls)


def test_service_executes_verified_direct_candidate():
    result = SemanticSQLService(DB, provider=DemoProvider()).ask("Show total sales from Czech Republic in 2024")
    assert result.status == "ok"
    assert result.verification and result.verification.ok
    assert result.rows == ((370.0,),)
    assert [stage.name for stage in result.stages] == ["sql", "verify", "execute"]


def test_service_repairs_before_execution():
    provider = SequencedProvider("SELECT missing FROM orders", "SELECT COUNT(*) FROM orders")
    result = SemanticSQLService(DB, provider=provider, max_repairs=1).ask("How many orders?")
    assert result.status == "ok"
    assert result.candidate == SQLCandidate("SELECT COUNT(*) FROM orders", attempt=1)
    assert [stage.name for stage in result.stages] == ["sql", "verify", "repair", "verify", "execute"]
    assert provider.text_calls == 2


def test_service_refuses_when_repair_budget_is_zero():
    provider = SequencedProvider("SELECT missing FROM orders")
    result = SemanticSQLService(DB, provider=provider, max_repairs=0).ask("How many orders?")
    assert result.status == "verification_failed"
    assert result.rows == ()
    assert provider.text_calls == 1


def test_service_accepts_frozen_initial_candidate_without_generation():
    provider = SequencedProvider()
    candidate = SQLCandidate("SELECT COUNT(*) FROM orders")
    result = SemanticSQLService(DB, provider=provider).ask("How many orders?", initial_candidate=candidate)
    assert result.status == "ok"
    assert result.candidate == candidate
    assert provider.text_calls == 0
