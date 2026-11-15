"""No SQL gold or model is needed to check that an IR compiles safely."""
from pathlib import Path
import sqlite3

from evaluation.typed_query_ir import compile_typed_query
from semantic_sql.catalog import DatabaseCatalog


def _catalog(tmp_path):
    db = tmp_path / "toy.db"
    with sqlite3.connect(db) as con:
        con.executescript("""
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY, customer_id INTEGER,
            status TEXT, amount REAL,
            FOREIGN KEY(customer_id) REFERENCES customers(id)
        );
        INSERT INTO customers VALUES (1, 'A'), (2, 'B');
        INSERT INTO orders VALUES (1, 1, 'paid', 10), (2, 1, 'paid', 20), (3, 2, 'open', 30);
        """)
    return db, DatabaseCatalog.from_sqlite(db)


def _grouped():
    return {
        "from": "orders",
        "joins": [{"table": "customers", "on": ["orders.customer_id", "customers.id"]}],
        "select": [
            {"op": "column", "column": "customers.name"},
            {"op": "count", "column": "orders.id", "as": "n"},
        ],
        "where": [{"column": "orders.status", "op": "=", "value": "paid"}],
        "grain": ["customers.name"],
        "group_by": ["customers.name"],
        "order_by": [{"alias": "n", "direction": "DESC"}],
        "limit": 5,
    }


def test_compiles_grouped_join_and_executes_against_real_sqlite(tmp_path):
    db, cat = _catalog(tmp_path)
    sql = compile_typed_query(_grouped(), cat)
    assert sql is not None
    with sqlite3.connect(db) as con:
        assert con.execute(sql).fetchall() == [("A", 2)]


def test_compiles_single_table_projection(tmp_path):
    db, cat = _catalog(tmp_path)
    plan = {"from": "customers", "select": [{"op": "column", "column": "customers.name"}],
            "grain": [], "where": [], "joins": [], "group_by": [], "order_by": [], "limit": None}
    sql = compile_typed_query(plan, cat)
    assert sql is not None
    with sqlite3.connect(db) as con:
        assert con.execute(sql).fetchall() == [("A",), ("B",)]


def test_abstains_on_unverified_joins_or_missing_schema(tmp_path):
    _, cat = _catalog(tmp_path)
    p = _grouped()
    p["joins"] = [{"table": "customers", "on": ["orders.id", "customers.id"]}]
    assert compile_typed_query(p, cat) is None
    p = _grouped()
    p["select"][1]["column"] = "orders.missing"
    assert compile_typed_query(p, cat) is None
    p = _grouped()
    p["joins"] = []
    assert compile_typed_query(p, cat) is None


def test_abstains_on_ambiguous_grain_or_unhandled_structure(tmp_path):
    _, cat = _catalog(tmp_path)
    p = _grouped()
    p["grain"] = ["orders.id"]
    assert compile_typed_query(p, cat) is None
    p = _grouped()
    p["having"] = "COUNT(*) > 1"
    assert compile_typed_query(p, cat) is None
    p = _grouped()
    p["select"].append({"op": "sum", "column": "orders.amount"})
    p["group_by"] = []
    assert compile_typed_query(p, cat) is None


def test_quotes_literals_without_allowing_injected_statements(tmp_path):
    db, cat = _catalog(tmp_path)
    p = {"from": "orders", "select": [{"op": "column", "column": "orders.id"}],
         "joins": [], "where": [{"column": "orders.status", "op": "=",
                                  "value": "paid'; DROP TABLE orders; --"}],
         "grain": [], "group_by": [], "order_by": [], "limit": None}
    sql = compile_typed_query(p, cat)
    assert sql is not None
    assert "''; DROP TABLE" in sql
    with sqlite3.connect(db) as con:
        assert con.execute(sql).fetchall() == []
        assert con.execute("SELECT count(*) FROM orders").fetchone() == (3,)


def test_abstains_on_invalid_types_and_invalid_alias(tmp_path):
    _, cat = _catalog(tmp_path)
    p = _grouped()
    p["select"][1]["as"] = "n;DROP"
    assert compile_typed_query(p, cat) is None
    p = _grouped()
    p["limit"] = -1
    assert compile_typed_query(p, cat) is None
    p = _grouped()
    p["where"][0]["value"] = ["paid"]
    assert compile_typed_query(p, cat) is None


def test_same_model_provider_can_only_emit_compiled_ir_not_raw_sql(tmp_path):
    import json
    from evaluation.typed_query_ir import draft_typed_query_sql
    _, cat = _catalog(tmp_path)

    class Provider:
        response = json.dumps(_grouped())
        def complete_text(self, *, system, user):
            assert "JSON" in system and "physical schema" in user
            assert "paid" in user
            return self.response

    p = Provider()
    sql = draft_typed_query_sql("Count paid orders per customer", cat.schema_text(),
                                cat, p, evidence="paid")
    assert sql is not None and sql.startswith("SELECT ")
    p.response = "SELECT * FROM orders"
    assert draft_typed_query_sql("q", cat.schema_text(), cat, p) is None
    p.response = '{"from": "orders", "select": [{"op": "raw_sql", "column": "DROP TABLE orders"}]}'
    assert draft_typed_query_sql("q", cat.schema_text(), cat, p) is None
