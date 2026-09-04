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
