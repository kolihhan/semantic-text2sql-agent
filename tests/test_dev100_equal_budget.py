from __future__ import annotations

import sqlite3

from experiments.dev100_equal_budget import (
    DIGEST, EXTRA, build_graph, choose, plan, resolved, semantic_witness,
)
from semantic_sql.catalog import DatabaseCatalog


class FixedProvider:
    def __init__(self, sql):
        self.sql = sql
        self.calls = 0

    def complete_text(self, *, system, user):
        self.calls += 1
        return self.sql


def db_catalog(tmp_path):
    root = tmp_path / "databases"
    folder = root / "test"
    folder.mkdir(parents=True)
    db = folder / "test.sqlite"
    with sqlite3.connect(db) as con:
        con.executescript("""
            CREATE TABLE client(id INTEGER PRIMARY KEY, name TEXT);
            CREATE TABLE visit(id INTEGER PRIMARY KEY, client_id INT,
                FOREIGN KEY(client_id) REFERENCES client(id));
            INSERT INTO client VALUES (1,'John'),(2,'Jane');
            INSERT INTO visit VALUES (1,1),(2,1),(3,2);
        """)
    return root, db, DatabaseCatalog.from_sqlite(db)


def test_source_only_witnesses_and_gold_blind_gate(tmp_path):
    _, db, cat = db_catalog(tmp_path)
    w = semantic_witness('How many clients visited?', 
        'SELECT COUNT(*) FROM client JOIN visit ON client.id=visit.client_id', cat)
    assert w["kind"] == "JOIN_MULTIPLICITY"
    assert resolved("SELECT COUNT(DISTINCT client.id) FROM client JOIN visit ON client.id=visit.client_id", w)
    assert not resolved("SELECT COUNT(*) FROM client JOIN visit ON client.id=visit.client_id", w)
    quoted = semantic_witness('Find client "John"', 'SELECT name FROM client', cat)
    assert quoted["kind"] == "NAMED_ENTITY_FILTER"
    assert choose(db, "SELECT name FROM client", "DROP TABLE client", quoted)[1] != "selected_gold_blind"
    assert choose(db, "SELECT name FROM client", "SELECT name FROM client WHERE name='John'", quoted)[1] == "selected_gold_blind"


def test_langgraph_state_routes_semantic_and_fallback(tmp_path):
    _, db, _ = db_catalog(tmp_path)
    provider = FixedProvider("SELECT COUNT(DISTINCT client.id) FROM client JOIN visit ON client.id=visit.client_id")
    s = build_graph().invoke({
        "question":"How many clients visited?",
        "schema":"client(id,name), visit(id,client_id)",
        "evidence":"",
        "sql":"SELECT COUNT(*) FROM client JOIN visit ON client.id=visit.client_id",
        "witness":{"kind":"JOIN_MULTIPLICITY","detail":"join fanout"}
    }, context={"provider":provider,"database":db})
    assert s["why"] == "selected_gold_blind"
    assert provider.calls == 1
    p2=FixedProvider("SELECT * FROM nonexistent")
    r=build_graph().invoke({**s,"sql":"SELECT * FROM nowhere",
        "witness":{"kind":"COMPILE_REPAIR","detail":"sqlite compile failed"}},
        context={"provider":p2,"database":db})
    assert r["why"] == "invalid_readonly_or_compile"
    assert r["final_sql"] == "SELECT * FROM nowhere"
    assert p2.calls == 1


def test_exact_budget_precommitted_from_only_source(tmp_path):
    root, db, _ = db_catalog(tmp_path)
    rows = [{"case_id":str(i),"database_id":"test",
             "question":f'Find "Person{i}"',"evidence":""} for i in range(100)]
    old = {c["case_id"]:{"initial_sql":"SELECT name FROM client",
                        "question":c["question"],"database_id":"test"} for c in rows}
    p = plan({"cases":rows,"cohort_sha256":"fixed","model_digest":DIGEST}, old, root)
    assert sum(c["selected"] for c in p["cases"]) == EXTRA
    assert all(c["witness"]["kind"]=="NAMED_ENTITY_FILTER" for c in p["cases"])
    assert p["equal_budget_total_calls"]==127
    assert [c["case_id"] for c in p["cases"] if c["selected"]] == [
        c["case_id"] for c in plan({"cases":rows,"cohort_sha256":"fixed",
            "model_digest":DIGEST},old,root)["cases"] if c["selected"]]
