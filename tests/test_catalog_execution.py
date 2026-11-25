from pathlib import Path
import sqlite3

import pytest

from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.execution import SQLExecutionError, execute_readonly
from evaluation.schema_context import select_schema_context


def make_db(tmp_path: Path) -> Path:
    db = tmp_path / "demo.db"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, country_code TEXT);
        CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price REAL);
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            product_id INTEGER,
            total_amount REAL,
            created_at TEXT,
            FOREIGN KEY(customer_id) REFERENCES customers(id),
            FOREIGN KEY(product_id) REFERENCES products(id)
        );
        INSERT INTO customers VALUES (1, 'Alice', 'CZE'), (2, 'Bob', 'MYS');
        INSERT INTO products VALUES (1, 'Widget', 10.0);
        INSERT INTO orders VALUES (1, 1, 1, 100.0, '2024-05-01'), (2, 2, 1, 50.0, '2023-01-01');
        """
    )
    con.close()
    return db


def test_catalog_introspects_tables_columns_foreign_keys_and_values(tmp_path):
    catalog = DatabaseCatalog.from_sqlite(make_db(tmp_path))
    assert set(catalog.tables) == {"customers", "products", "orders"}
    assert "country_code" in catalog.tables["customers"].columns
    assert ("orders", "customer_id", "customers", "id") in catalog.foreign_keys
    assert "CZE" in catalog.sample_values("customers", "country_code")


def test_readonly_executor_returns_columns_rows_and_enforces_limit(tmp_path):
    db = make_db(tmp_path)
    result = execute_readonly(db, "SELECT name FROM customers ORDER BY id", max_rows=1)
    assert result.columns == ("name",)
    assert result.rows == (("Alice",),)
    assert result.truncated is True


def test_readonly_executor_rejects_mutation(tmp_path):
    db = make_db(tmp_path)
    with pytest.raises(SQLExecutionError, match="read-only"):
        execute_readonly(db, "DELETE FROM customers")


def test_schema_selector_matches_columns_and_keeps_fk_neighbors(tmp_path):
    catalog = DatabaseCatalog.from_sqlite(make_db(tmp_path))
    context, diagnostics = select_schema_context(catalog, tmp_path, "demo", "show customer total amount")
    assert diagnostics["selector"] == "deterministic_lexical_v1"
    assert diagnostics["fallback_full_schema"] is False
    assert {"orders", "customers", "products"} <= set(diagnostics["selected_tables"])
    assert "FK orders.customer_id -> customers.id" in context


def test_schema_selector_falls_back_when_question_is_uncertain(tmp_path):
    catalog = DatabaseCatalog.from_sqlite(make_db(tmp_path))
    context, diagnostics = select_schema_context(catalog, tmp_path, "demo", "please answer this")
    assert diagnostics["fallback_full_schema"] is True
    assert set(diagnostics["selected_tables"]) == set(catalog.tables)
    assert "products" in context and "customers" in context


def test_native_ddl_selected_tables_and_descriptions(tmp_path):
    db = make_db(tmp_path)
    catalog = DatabaseCatalog.from_sqlite(db)
    before = db.read_bytes()
    ddl = catalog.native_ddl({"orders"})
    assert "CREATE TABLE orders" in ddl
    assert "total_amount REAL" in ddl
    assert "REFERENCES customers(id)" in ddl
    assert "CREATE TABLE customers" not in ddl
    with pytest.raises(ValueError, match="Native DDL unavailable"):
        catalog.native_ddl({"missing"})
    desc = tmp_path / "demo" / "database_description"
    desc.mkdir(parents=True)
    (desc / "orders.csv").write_text("original_column_name,column_description\ntotal_amount,Order total\n")
    original, baseline = select_schema_context(catalog, tmp_path, "demo", "show customer total amount")
    treatment, diagnostic = select_schema_context(catalog, tmp_path, "demo", "show customer total amount", native_ddl=True)
    assert diagnostic["selected_tables"] == baseline["selected_tables"]
    assert diagnostic["schema_format"] == "native_ddl"
    assert "CREATE TABLE orders" in treatment and "Order total" in treatment
    assert "FK orders.customer_id -> customers.id" in original
    assert db.read_bytes() == before


def test_service_passes_native_ddl_to_generation_and_existing_repair(tmp_path):
    from semantic_sql.agent import SemanticSQLService

    prompts = []
    class Provider:
        def complete_text(self, *, system, user):
            prompts.append(user)
            return "SELECT missing FROM orders" if len(prompts) == 1 else "SELECT COUNT(*) FROM orders"

    result = SemanticSQLService(make_db(tmp_path), provider=Provider()).ask("How many orders?")
    assert result.status == "ok"
    assert result.rows == ((2,),)
    assert len(prompts) == 2
    assert all("CREATE TABLE orders" in p and "REFERENCES customers(id)" in p for p in prompts)
