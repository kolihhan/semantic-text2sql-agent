"""Physical SQLite proof that COUNT(*) JOIN fanout can miscount entities.

These tests are synthetic correctness/abstention tests, not BIRD EX evidence.
"""
import sqlite3
from semantic_sql.grain_guard import propose_join_entity_grain


def sample_db(tmp_path):
    db=tmp_path/"physical.sqlite"
    conn=sqlite3.connect(db)
    conn.executescript("""
    CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT);
    CREATE TABLE orders(id INTEGER PRIMARY KEY, customer_id INTEGER,
                        price INTEGER, FOREIGN KEY(customer_id) REFERENCES customers(id));
    INSERT INTO customers VALUES(1,'Jane'),(2,'Mike'),(3,'Alice');
    INSERT INTO orders VALUES(10,1,20),(11,1,35),(12,2,25),(13,3,5);
    """)
    conn.commit();conn.close()
    return db


def test_rewrite_count_join_fanout_only_when_counting_distinct_parent_entities(tmp_path):
    db=sample_db(tmp_path)
    sql=("SELECT COUNT(*) FROM customers c JOIN orders o ON c.id = o.customer_id "
         "WHERE o.price >= 20;")
    q="How many customers placed orders worth at least 20?"
    r=propose_join_entity_grain(db,sql,q)
    assert r.changed and r.original_count==3 and r.distinct_count==2
    assert r.sql.startswith("SELECT COUNT(DISTINCT c.id)")


def test_no_rewrite_when_question_counts_join_children(tmp_path):
    db=sample_db(tmp_path)
    sql=("SELECT COUNT(*) FROM customers c JOIN orders o ON c.id = o.customer_id "
         "WHERE o.price >= 20;")
    r=propose_join_entity_grain(db,sql,"How many orders worth at least 20 were placed by customers?")
    assert not r.changed and r.sql==sql
    assert r.reason=="question_entity_not_explicit"


def test_abstain_when_no_duplicates_or_ambiguous_parent_noun(tmp_path):
    db=sample_db(tmp_path)
    sql="SELECT COUNT(*) FROM customers c JOIN orders o ON c.id = o.customer_id WHERE o.price > 100;"
    assert propose_join_entity_grain(db,sql,"How many customers ordered over 100?").reason=="no_verified_join_fanout"
    assert not propose_join_entity_grain(db,sql,"List the customers.").changed


def test_abstain_without_primary_key_or_with_nested_group(tmp_path):
    db=sample_db(tmp_path)
    sql="SELECT COUNT(*) FROM customers c JOIN orders o ON c.name = o.customer_id;"
    assert not propose_join_entity_grain(db,sql,"How many customers?").changed
    sql="SELECT COUNT(*) FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.name;"
    assert not propose_join_entity_grain(db,sql,"How many customers?").changed


def test_abstain_on_non_readonly_and_complex_sql(tmp_path):
    db=sample_db(tmp_path)
    for sql in [
        "DELETE FROM customers",
        "SELECT COUNT(*) FROM customers c JOIN orders o ON c.id = o.customer_id; DROP TABLE customers;",
        "SELECT COUNT(*) FROM customers c JOIN orders o ON c.id = o.customer_id JOIN customers d ON d.id=o.customer_id;",
        "SELECT COUNT(DISTINCT c.id) FROM customers c JOIN orders o ON c.id=o.customer_id;",
    ]:
        assert not propose_join_entity_grain(db,sql,"How many customers have orders?").changed


def test_business_plural_singular_alias_case(tmp_path):
    assert "business" in __import__("semantic_sql.grain_guard",fromlist=["_entity_forms"])._entity_forms("businesses")
    assert "company" in __import__("semantic_sql.grain_guard",fromlist=["_entity_forms"])._entity_forms("companies")
