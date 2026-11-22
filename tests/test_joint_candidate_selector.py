"""Candidate-selection diagnostics only; these synthetic SQLs are not BIRD EX evidence."""
from semantic_sql.joint_candidate_selector import JointObligation, select_joint_repair
from semantic_sql.sql_semantics import ColumnRef, NormalizedJoin


REQUIRED = JointObligation(
    entity_column=ColumnRef("customer", "name"),
    join_edge=NormalizedJoin(ColumnRef("orders", "customer_id"), ColumnRef("customer", "id")),
    count_grain=ColumnRef("orders", "id"),
    distinct=True,
    confidence="high",
)
ORIGINAL = (
    "SELECT customer.name, COUNT(DISTINCT orders.id) FROM customer "
    "JOIN orders ON customer.id = orders.seller_id GROUP BY customer.name"
)
CORRECT_SHAPE = (
    "SELECT customer.name, COUNT(DISTINCT orders.id) FROM customer "
    "JOIN orders ON customer.id = orders.customer_id GROUP BY customer.name"
)
WRONG_GRAIN = (
    "SELECT customer.name, COUNT(DISTINCT orders.customer_id) FROM customer "
    "JOIN orders ON customer.id = orders.customer_id GROUP BY customer.name"
)
WRONG_ENTITY = (
    "SELECT customer.id, COUNT(DISTINCT orders.id) FROM customer "
    "JOIN orders ON customer.id = orders.customer_id GROUP BY customer.id"
)


def test_joint_requirements_choose_only_complete_candidate():
    out = select_joint_repair(ORIGINAL, [WRONG_ENTITY, WRONG_GRAIN, CORRECT_SHAPE], REQUIRED)
    assert out.status == "SELECT_REPAIR"
    assert out.sql == CORRECT_SHAPE


def test_preserves_original_when_it_already_covers_all_three():
    out = select_joint_repair(CORRECT_SHAPE, [ORIGINAL], REQUIRED)
    assert out.status == "KEEP_ORIGINAL"
    assert out.sql == CORRECT_SHAPE


def test_tie_between_different_supported_sqls_abstains():
    same_semantics_other_sql = CORRECT_SHAPE + " ORDER BY customer.name"
    out = select_joint_repair(ORIGINAL, [CORRECT_SHAPE, same_semantics_other_sql], REQUIRED)
    assert out.status == "ABSTAIN"
    assert out.sql == ORIGINAL


def test_unsupported_derived_query_abstains_not_false_negatives():
    derived = "SELECT customer.name, COUNT(DISTINCT orders.id) FROM customer JOIN orders ON customer.id = orders.customer_id WHERE customer.id IN (SELECT id FROM customer) GROUP BY customer.name"
    out = select_joint_repair(ORIGINAL, [CORRECT_SHAPE, derived], REQUIRED)
    assert out.status == "ABSTAIN"


def test_low_confidence_and_unknown_original_do_not_rewrite():
    from dataclasses import replace
    low = replace(REQUIRED, confidence="medium")
    assert select_joint_repair(ORIGINAL, [CORRECT_SHAPE], low).status == "ABSTAIN"
    assert select_joint_repair("SELECT FROM", [CORRECT_SHAPE], REQUIRED).status == "ABSTAIN"


def test_physical_sqlite_join_repair_and_unrelated_filter_regression():
    """Execute real SQLite rows; compatible AST alone cannot validate extra predicates."""
    import sqlite3

    original = (
        "SELECT customer.name, COUNT(DISTINCT orders.id) FROM customer "
        "JOIN orders ON customer.id = orders.seller_id "
        "WHERE orders.amount >= 50 GROUP BY customer.name"
    )
    correct = original.replace("seller_id", "customer_id")
    bad_filter = correct.replace("amount >= 50", "amount < 50")
    with sqlite3.connect(":memory:") as db:
        db.executescript("""
        CREATE TABLE customer (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INT, seller_id INT, amount INT);
        INSERT INTO customer VALUES (1, 'Alice'), (2, 'Bob');
        INSERT INTO orders VALUES (1,1,2,80), (2,1,2,70), (3,2,1,60), (4,2,1,20);
        """)
        assert db.execute(original).fetchall() == [('Alice', 1), ('Bob', 2)]
        assert db.execute(correct).fetchall() == [('Alice', 2), ('Bob', 1)]
        assert db.execute(bad_filter).fetchall() != db.execute(correct).fetchall()
    assert select_joint_repair(original, [correct], REQUIRED).status == "SELECT_REPAIR"
    # A repair may satisfy entity+relation+grain while changing a correct filter.
    assert select_joint_repair(original, [bad_filter], REQUIRED).status == "ABSTAIN"
