from pathlib import Path
import sqlite3

from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.execution import execute_readonly
from evaluation.unique_fk_join_repair import repair_one_unique_fk_join


def fixture(tmp_path:Path, *, ambiguous:bool=False, declared:bool=True):
    db=tmp_path/"orders.sqlite"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            "CREATE TABLE users(user_id INTEGER PRIMARY KEY, name TEXT);"
            "CREATE TABLE orders(order_id INTEGER PRIMARY KEY, user_id INTEGER,"
            " approved_by INTEGER,"
            + ("FOREIGN KEY(user_id) REFERENCES users(user_id)" if declared else "CHECK(user_id > 0)")
            + (", FOREIGN KEY(approved_by) REFERENCES users(user_id)" if ambiguous else "")
            + ");"
            "INSERT INTO users VALUES (1,'Alice'),(2,'Bob');"
            "INSERT INTO orders(order_id,user_id,approved_by) VALUES(11,1,2),(12,1,2),(13,2,1);"
        )
    return db,DatabaseCatalog.from_sqlite(db)


def test_physical_one_endpoint_repair_changes_actual_execution(tmp_path):
    db,catalog=fixture(tmp_path)
    original=("SELECT u.name FROM users AS u JOIN orders AS o "
              "ON u.user_id = o.order_id WHERE o.order_id = 11")
    out=repair_one_unique_fk_join(original,catalog)
    assert out.changed and out.reason=="one_endpoint_matches_unique_physical_fk"
    assert out.physical_fk=="u.user_id = o.user_id"
    assert "WHERE o.order_id = 11" in out.sql
    assert execute_readonly(db,original).rows==()
    assert execute_readonly(db,out.sql).rows==(("Alice",),)


def test_existing_correct_query_never_rewritten(tmp_path):
    _,cat=fixture(tmp_path)
    query="SELECT u.name FROM users u JOIN orders o ON u.user_id = o.user_id"
    out=repair_one_unique_fk_join(query,cat)
    assert not out.changed and out.sql==query


def test_ambiguous_published_relationship_refuses(tmp_path):
    _,cat=fixture(tmp_path,ambiguous=True)
    query="SELECT u.name FROM users u JOIN orders o ON u.user_id = o.order_id"
    out=repair_one_unique_fk_join(query,cat)
    assert not out.changed


def test_without_physical_fk_no_sql_semantic_guess(tmp_path):
    _,cat=fixture(tmp_path,declared=False)
    query="SELECT u.name FROM users u JOIN orders o ON u.user_id = o.order_id"
    assert not repair_one_unique_fk_join(query,cat).changed


def test_incomplete_source_anchor_and_compound_join_refused(tmp_path):
    _,cat=fixture(tmp_path)
    one=("SELECT u.name FROM users u JOIN orders o "
         "ON u.name = o.order_id")
    assert not repair_one_unique_fk_join(one,cat).changed
    compound=("SELECT u.name FROM users u JOIN orders o ON "
              "u.user_id = o.order_id AND o.order_id > 0")
    assert not repair_one_unique_fk_join(compound,cat).changed


def test_nested_or_union_never_regex_rewritten(tmp_path):
    _,cat=fixture(tmp_path)
    nested=("SELECT * FROM users u JOIN orders o "
            "ON u.user_id = o.order_id WHERE EXISTS "
            "(SELECT 1 FROM orders WHERE order_id = 11)")
    assert not repair_one_unique_fk_join(nested,cat).changed
    union=("SELECT u.name FROM users u JOIN orders o ON u.user_id = o.order_id "
           "UNION SELECT name FROM users")
    assert not repair_one_unique_fk_join(union,cat).changed


def test_preserve_left_join_and_trailing_clause(tmp_path):
    _,cat=fixture(tmp_path)
    query=("SELECT u.name FROM users u LEFT JOIN orders o ON "
           "u.user_id = o.order_id ORDER BY u.name;")
    out=repair_one_unique_fk_join(query,cat)
    assert out.changed
    assert out.sql.endswith("ORDER BY u.name;")
    assert "LEFT JOIN orders o ON u.user_id = o.user_id" in out.sql
