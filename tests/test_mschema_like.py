import sqlite3

import pytest

from evaluation.mschema_like import to_mschema_like
from semantic_sql.catalog import DatabaseCatalog


@pytest.fixture
def catalog(tmp_path):
    db = tmp_path / "mini.sqlite"
    with sqlite3.connect(db) as con:
        con.executescript("""
            CREATE TABLE customer(id INTEGER PRIMARY KEY, customer_name TEXT);
            CREATE TABLE orders(id INTEGER PRIMARY KEY, customer_id INTEGER,
                total INTEGER, FOREIGN KEY(customer_id) REFERENCES customer(id));
            INSERT INTO customer VALUES (1, 'Alice'), (2, 'Bob');
            INSERT INTO orders VALUES (100, 1, 25);
        """)
    return DatabaseCatalog.from_sqlite(db)


def test_mschema_without_samples_preserves_keys_and_types(catalog):
    schema = to_mschema_like(
        catalog, db_id="mini", selected_tables=["customer", "orders"],
        question="What customer name?", lexical_context=(
            "Physical SQLite schema:\ncustomer(id, customer_name)\n"
            "orders(id, customer_id, total)\nColumn descriptions:\n"
            "- customer.customer_name: customer legal name"
        ), with_examples=False,
    )
    assert "〖DB_ID〗 mini" in schema
    assert "〖Schema〗" in schema
    assert "# Table: customer" in schema
    assert "(id:INTEGER, Primary Key)" in schema
    assert "(customer_name:TEXT, customer legal name)" in schema
    assert "〖Foreign keys〗" in schema
    assert "orders.customer_id=customer.id" in schema
    assert "Examples:" not in schema


def test_mschema_optional_examples_only_on_matching_columns(catalog):
    schema = to_mschema_like(
        catalog, db_id="mini", selected_tables=["customer", "orders"],
        question="Which customer name?", lexical_context="",
        with_examples=True,
    )
    assert "Examples: [Alice, Bob]" in schema
    assert "(total:INTEGER, Examples:" not in schema


def test_reject_unknown_or_duplicate_table(catalog):
    for bad in ([], ["customer", "customer"], ["not_real"]):
        with pytest.raises(ValueError):
            to_mschema_like(
                catalog, db_id="mini", selected_tables=bad,
                question="names", lexical_context="", with_examples=False,
            )
