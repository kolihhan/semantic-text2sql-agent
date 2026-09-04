from pathlib import Path

from semantic_sql.agent import SemanticSQLService
from semantic_sql.contracts import SQLCandidate
from semantic_sql.inference import _build_guarded_graph, run_guarded


ROOT = Path(__file__).parents[1]
DB = ROOT / "demo" / "demo.db"


class DirectProvider:
    def __init__(self, *sqls: str) -> None:
        self.sqls = iter(sqls)
        self.json_calls = 0
        self.text_calls = 0
        self.prompts: list[tuple[str, str]] = []

    def complete_text(self, *, system: str, user: str) -> str:
        self.text_calls += 1
        self.prompts.append((system, user))
        return next(self.sqls)


def test_current_graph_contains_only_guarded_direct_nodes() -> None:
    graph = _build_guarded_graph().get_graph()
    assert set(graph.nodes) == {"__start__", "generate", "verify", "repair", "execute", "refuse", "__end__"}
    assert {(edge.source, edge.target) for edge in graph.edges} == {
        ("__start__", "generate"),
        ("generate", "verify"),
        ("verify", "execute"),
        ("verify", "repair"),
        ("verify", "refuse"),
        ("repair", "verify"),
        ("execute", "__end__"),
        ("refuse", "__end__"),
    }


def test_service_uses_schema_without_evidence_or_json_planner() -> None:
    provider = DirectProvider("SELECT SUM(orders.total_amount) FROM orders")
    result = SemanticSQLService(DB, provider=provider).ask("Show total sales")
    assert result.status == "ok"
    assert provider.json_calls == 0
    assert provider.text_calls == 1
    assert "orders" in provider.prompts[0][1]
    assert "evidence" not in provider.prompts[0][1].lower()


def test_optional_evidence_reaches_generation() -> None:
    provider = DirectProvider("SELECT SUM(orders.total_amount) FROM orders")
    SemanticSQLService(DB, provider=provider).ask("Show total sales", evidence="source hint")
    assert "source hint" in provider.prompts[0][1]


def test_frozen_candidate_skips_initial_generation() -> None:
    provider = DirectProvider()
    candidate = SQLCandidate("SELECT SUM(orders.total_amount) FROM orders")
    result = SemanticSQLService(DB, provider=provider).ask("Show total sales", initial_candidate=candidate)
    assert result.status == "ok"
    assert result.candidate == candidate
    assert provider.text_calls == 0


def test_repair_precedes_execution_and_zero_budget_refuses(tmp_path: Path, monkeypatch) -> None:
    bad = "SELECT s.id, County Name FROM schools"
    good = 'SELECT "County Name" FROM schools'
    database = tmp_path / "db.sqlite"
    import sqlite3

    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE schools (id INTEGER, "County Name" TEXT)')
        connection.execute('INSERT INTO schools VALUES (1, "Alpha")')

    provider = DirectProvider(bad, good)
    execution_order: list[str] = []
    import semantic_sql.inference as inference

    original = inference.execute_readonly
    monkeypatch.setattr(
        inference,
        "execute_readonly",
        lambda *args, **kwargs: (execution_order.append(args[1]), original(*args, **kwargs))[1],
    )
    result = run_guarded(
        database=database,
        provider=provider,
        question="Return county",
        schema_context='schools(id, "County Name")',
        max_repairs=1,
        max_rows=10,
    )
    assert result.status == "ok"
    assert execution_order == [good]

    refusing = DirectProvider(bad)
    result = run_guarded(
        database=database,
        provider=refusing,
        question="Return county",
        schema_context='schools(id, "County Name")',
        max_repairs=0,
        max_rows=10,
    )
    assert result.status == "verification_failed"
    assert refusing.text_calls == 1


def test_guarded_repair_prompt_requests_internal_step_by_step_reasoning_but_sql_only_output() -> None:
    from types import SimpleNamespace
    from semantic_sql.contracts import SQLCandidate, VerificationIssue, VerificationResult
    from semantic_sql.inference import _repair

    provider = DirectProvider('SELECT COUNT(*) FROM items')
    state = {
        'question': 'How many items?',
        'schema_context': 'items(id)',
        'candidate': SQLCandidate('SELECT missing FROM items'),
        'verification': VerificationResult(False, (VerificationIssue('sqlite_compile_error', 'no such column: missing'),)),
        'stages': (),
    }
    runtime = SimpleNamespace(context=SimpleNamespace(provider=provider))
    repaired = _repair(state, runtime)

    system, user = provider.prompts[-1]
    prompt = (system + '\n' + user).lower()
    assert 'reason step-by-step internally' in prompt
    assert 'user intent' in prompt
    assert 'schema' in prompt
    assert 'verifier' in prompt
    assert 'smallest correction' in prompt
    assert 'return corrected sql only' in prompt
    assert repaired['candidate'].sql == 'SELECT COUNT(*) FROM items'
