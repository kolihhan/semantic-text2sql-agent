from __future__ import annotations

import sqlite3

import pytest

from semantic_sql.grouped_global_cardinality import find_global_cardinality_contradiction

SQL = (
    "SELECT COUNT(DISTINCT o.orderNumber) FROM orders o "
    "JOIN orderdetails od ON o.orderNumber = od.orderNumber "
    "WHERE o.status = 'Cancelled' "
    "GROUP BY o.orderNumber "
    "HAVING SUM(od.quantityOrdered * od.priceEach) > 4000"
)
QUESTION="How many different orders with a total price greater than 4000 are cancelled?"


@pytest.fixture
def db(tmp_path):
    path=tmp_path/"orders.sqlite"
    with sqlite3.connect(path) as c:
        c.executescript("""
        CREATE TABLE orders(orderNumber INTEGER PRIMARY KEY, status TEXT);
        CREATE TABLE orderdetails(id INTEGER PRIMARY KEY,
           orderNumber INTEGER REFERENCES orders(orderNumber),
           quantityOrdered INTEGER,priceEach INTEGER);
        INSERT INTO orders VALUES(1,'Cancelled'),(2,'Cancelled'),(3,'Shipped');
        INSERT INTO orderdetails VALUES
          (1,1,20,250),(2,1,1,3),(3,2,30,200),(4,3,30,200);
        """)
    return path


def test_global_grouped_count_has_physical_witness_and_preserves_having(db):
    witness=find_global_cardinality_contradiction(db,QUESTION,SQL)
    assert witness is not None
    assert witness.count_per_group==(1,1)
    with sqlite3.connect(db) as c:
        assert c.execute(SQL).fetchall()==[(1,),(1,)]
        assert c.execute(witness.proposed_sql).fetchall()==[(2,)]
        assert c.execute("SELECT COUNT(*) FROM orders").fetchone()[0]==3
    assert "HAVING SUM(od.quantityOrdered * od.priceEach) > 4000" in witness.proposed_sql


@pytest.mark.parametrize("question",[
    "For each cancelled order, how many different items were ordered?",
    "List them: How many different orders with a total greater than 4000?",
    "How many different orders per customer were cancelled?",
    "Which orders were cancelled?",
])
def test_question_does_not_require_one_global_scalar(db,question):
    assert find_global_cardinality_contradiction(db,question,SQL) is None


@pytest.mark.parametrize("sql",[
    SQL.replace("COUNT(DISTINCT o.orderNumber)","COUNT(*)"),
    SQL.replace("GROUP BY o.orderNumber","GROUP BY o.status"),
    SQL.replace("GROUP BY o.orderNumber","GROUP BY o.orderNumber, o.status"),
    SQL.replace("HAVING SUM(od.quantityOrdered * od.priceEach) > 4000","HAVING COUNT(DISTINCT o.orderNumber)>0"),
    SQL + " ORDER BY o.orderNumber",
    "WITH q AS ("+SQL+") SELECT * FROM q",
    SQL.replace("HAVING SUM(od.quantityOrdered * od.priceEach) > 4000",
                "HAVING SUM(od.quantityOrdered * od.priceEach) > 999999999"),
])
def test_complex_or_unsupported_or_no_multiple_groups_abstains(db,sql):
    assert find_global_cardinality_contradiction(db,QUESTION,sql) is None


def test_original_global_count_is_never_rewritten(db):
    q=SQL.replace("GROUP BY o.orderNumber HAVING SUM(od.quantityOrdered * od.priceEach) > 4000","")
    assert find_global_cardinality_contradiction(db,QUESTION,q) is None
