"""Proof-of-wiring for independent Qwen plan IN the verifier (not SQL generation).

Fixtures validate control flow/cost and SQL semantics, never benchmark lift.
"""
from __future__ import annotations
import json
import sqlite3
import pytest
from semantic_sql.inference import run_guarded
from semantic_sql.agent import SemanticSQLService

BAD = "SELECT SUM(amount) FROM orders"
GOOD = "SELECT COUNT(*) FROM orders"
QUESTION = "How many orders are there?"

def plan(**overrides):
    data = dict(
        confidence="high", projection=[],
        aggregation={"function": "COUNT", "target": None, "entity_grain": "orders", "distinct": False},
        group_by=[], ranking=None, filters=[], join_entities=["orders"],
    )
    data.update(overrides)
    return json.dumps(data)

class Model:
    def __init__(self, *responses):
        self.replies=list(responses)
        self.calls=[]
    def complete_text(self, *, system, user):
        self.calls.append((system, user))
        if not self.replies:
            raise AssertionError("unbudgeted model call")
        return self.replies.pop(0)

@pytest.fixture
def db(tmp_path):
    path=tmp_path/"orders.sqlite"
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, amount INTEGER)")
        con.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)")
        con.executemany("INSERT INTO orders(amount) VALUES (?)", [(10,), (20,)])
    return path

def execute(db, model, *, budget=2):
    return run_guarded(
        database=db, provider=model, question=QUESTION,
        schema_context="orders(id, amount)", max_repairs=budget,
        semantic_plan_verifier=True,
    )

def test_executable_but_wrong_cannot_pass_semantic_verifier(db):
    model=Model(BAD, plan(), GOOD)
    result=execute(db, model)
    assert result.status == "ok"
    assert result.rows == ((2,),) and result.candidate.sql == GOOD
    assert len(model.calls)==3
    assert "candidate sql" not in model.calls[1][1].casefold()
    assert BAD not in model.calls[1][1]  # independent plan, no candidate!
    stages=[x.name for x in result.stages]
    assert stages.count("semantic_plan_verify")==2
    assert stages.count("semantic_plan_repair")==1
    assert stages.count("semantic_plan_contract")==1
    assert stages.count("semantic_router")==1
    assert any("CONFLICT" in x.summary for x in result.stages)
    assert any("CONTRACT_CONSISTENT" in x.summary for x in result.stages)

def test_plan_reused_and_still_wrong_is_refused_even_if_executable(db):
    model=Model(BAD, plan(), BAD)
    result=execute(db, model)
    assert result.status=="verification_failed"
    assert len(model.calls)==3  # no second semantic repair
    assert result.rows == ()
    assert all(x.name != "execute" for x in result.stages)

def test_model_confidence_low_is_not_semantic_pass(db):
    model=Model(GOOD, plan(confidence="low"))
    result=execute(db, model)
    assert result.status=="semantic_unverified" and result.rows==((2,),)
    assert "semantic plan" in result.message.lower()
    assert len(model.calls)==2

def test_hallucinated_schema_ref_cannot_block_good_sql(db):
    model=Model(GOOD, plan(join_entities=["invented_table"]))
    result=execute(db, model)
    assert result.status=="semantic_unverified"
    assert result.rows==((2,),)
    assert len(model.calls)==2

def test_valid_sql_matching_contract_no_semantic_repair(db):
    model=Model(GOOD, plan())
    result=execute(db, model)
    assert result.status=="ok" and result.rows==((2,),)
    assert len(model.calls)==2  # direct SQL + independent plan
    assert not any(x.name in {"repair", "semantic_plan_repair"} for x in result.stages)

def test_one_zero_budget_plan_conflict_blocks_execution(db):
    model=Model(BAD, plan())
    result=execute(db, model, budget=0)
    assert result.status=="verification_failed" and result.rows==()
    assert len(model.calls)==2
    assert all(x.name!="execute" for x in result.stages)

def test_original_default_remains_one_call(db):
    model=Model(BAD)
    service=SemanticSQLService(database=db, provider=model)
    result=service.ask(QUESTION)
    assert result.status=="ok" and result.rows==((30,),)
    assert len(model.calls)==1

def test_opt_in_service_wires_plan_verifier(db):
    model=Model(BAD, plan(), GOOD)
    service=SemanticSQLService(database=db, provider=model, semantic_plan_verifier=True)
    result=service.ask(QUESTION)
    assert result.status=="ok" and result.rows==((2,),)

def test_mutually_exclusive_legacy_review_and_typed_plan(db):
    with pytest.raises(ValueError):
        run_guarded(
            database=db, provider=Model(), question=QUESTION, schema_context="orders",
            semantic_plan_verifier=True, semantic_revision=True,
        )

def test_no_fk_metadata_must_not_force_invalid_join_error(db):
    # Correct joins need not be represented in SQLite PRAGMA foreign_key_list.
    from semantic_sql.semantic_contract import parse_semantic_contract
    from semantic_sql.catalog import DatabaseCatalog
    from semantic_sql.typed_plan_gate import verify_plan_against_sql
    extraction=parse_semantic_contract(plan())
    verdict=verify_plan_against_sql(extraction=extraction,sql=GOOD,
                                    catalog=DatabaseCatalog.from_sqlite(db))
    assert verdict.status=="consistent"

def test_aggregate_match_but_unverified_question_filter_not_semantic_pass(db):
    requirement=plan(filters=[{"field":"orders.id","op":"=","value":1}])
    model=Model(GOOD, requirement)
    result=execute(db, model)
    assert len(model.calls)==2
    assert result.status=="semantic_unverified"
    assert result.rows==((2,),)
    assert "UNVERIFIED" in [x.summary for x in result.stages if x.name=="semantic_plan_verify"][0]

def test_unchecked_multi_entity_relation_not_semantic_pass(db):
    requirement=plan(join_entities=["orders","customers"])
    model=Model(GOOD, requirement)
    result=execute(db, model)
    assert result.status=="semantic_unverified"
    assert result.rows==((2,),)
    assert len(model.calls)==2

def test_actual_graph_has_separate_langgraph_nodes_and_cycles(db):
    from semantic_sql.inference import _build_guarded_graph
    base=_build_guarded_graph().get_graph()
    assert "semantic_plan_contract" not in base.nodes
    g=_build_guarded_graph(semantic_plan_verifier=True).get_graph()
    names=set(g.nodes)
    assert {"generate","verify","repair","semantic_router",
            "semantic_plan_contract","semantic_plan_verify",
            "semantic_plan_repair","semantic_plan_refuse","execute"} <= names
    edges={(e.source,e.target) for e in g.edges}
    assert ("semantic_router","semantic_plan_contract") in edges
    assert ("semantic_router","execute") in edges
    assert ("semantic_plan_contract","semantic_plan_verify") in edges
    assert ("semantic_plan_repair","verify") in edges
    assert ("semantic_plan_verify","semantic_plan_refuse") in edges


def test_selective_low_risk_skips_model_plan_but_never_claims_semantic_pass(db):
    model=Model("SELECT amount FROM orders")
    result=run_guarded(
        database=db, provider=model, question="List amounts.",
        schema_context="orders(id, amount)", semantic_plan_verifier=True,
        semantic_plan_policy="selective",
    )
    assert len(model.calls)==1
    assert result.status=="semantic_unverified"
    assert result.rows==((10,), (20,))
    assert [x.name for x in result.stages].count("semantic_router")==1
    assert not any(x.name=="semantic_plan_contract" for x in result.stages)


def test_selective_high_risk_invokes_independent_plan_once(db):
    model=Model(GOOD,plan())
    result=run_guarded(
        database=db, provider=model, question=QUESTION,
        schema_context="orders(id, amount)", semantic_plan_verifier=True,
        semantic_plan_policy="selective",
    )
    assert len(model.calls)==2
    assert result.status=="ok" and result.rows==((2,),)
    assert [x.name for x in result.stages].count("semantic_plan_contract")==1


def test_syntax_failure_at_zero_budget_never_spends_plan_call(db):
    model=Model("SELECT missing FROM orders")
    result=run_guarded(
        database=db, provider=model, question=QUESTION,
        schema_context="orders(id, amount)", max_repairs=0,
        semantic_plan_verifier=True,
    )
    assert result.status=="verification_failed"
    assert len(model.calls)==1
    assert not any(x.name.startswith("semantic_") for x in result.stages)


def test_syntax_repair_then_semantic_route_uses_independent_plan(db):
    model=Model("SELECT missing FROM orders", GOOD, plan())
    result=run_guarded(
        database=db, provider=model, question=QUESTION,
        schema_context="orders(id, amount)", semantic_plan_verifier=True,
    )
    assert result.status=="ok" and result.rows==((2,),)
    assert len(model.calls)==3
    assert [x.name for x in result.stages].count("repair")==1
    assert [x.name for x in result.stages].count("semantic_plan_contract")==1


def test_invalid_semantic_policy_fails_before_any_model_call(db):
    p=Model()
    with pytest.raises(ValueError):
        run_guarded(database=db,provider=p,question=QUESTION,
                    schema_context="orders(id, amount)",
                    semantic_plan_verifier=True,semantic_plan_policy="wat")
    assert p.calls==[]

def test_unexpected_ast_mixed_qualifier_error_abstains_not_crashes(db, monkeypatch):
    from semantic_sql import typed_plan_gate

    def raise_ambiguous_schema(*, extraction, sql, catalog):
        raise TypeError("'<' not supported between instances of 'NoneType' and 'str'")

    monkeypatch.setattr(typed_plan_gate, "verify_plan_against_sql", raise_ambiguous_schema)
    model = Model(GOOD, plan())
    result = execute(db, model)
    assert len(model.calls) == 2
    assert result.status == "semantic_unverified"
    assert result.rows == ((2,),)
    assert any("plan_runtime_inconclusive:TypeError" in s.summary for s in result.stages)
