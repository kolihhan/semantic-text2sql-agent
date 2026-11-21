"""Synthetic, gold-free checks for the frozen physical evidence detector."""
import sqlite3

from scripts.blind_physical_semantic_signals import (
    SIGNALS, _aliases, _metrics, _wilson, detect_one,
)


def _db():
    c = sqlite3.connect(":memory:")
    c.executescript("""
      CREATE TABLE schools(id INTEGER PRIMARY KEY, city TEXT, state TEXT);
      INSERT INTO schools VALUES (1, 'Little Lever Urban District', 'UK');
      CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT);
      CREATE TABLE orders(
        id INTEGER PRIMARY KEY, customer_id INTEGER, name TEXT,
        FOREIGN KEY(customer_id) REFERENCES customers(id)
      );
      INSERT INTO customers VALUES (1, 'Alice'),(2,'Bob');
      INSERT INTO orders VALUES (1,1,'x'),(2,1,'y'),(3,2,'z');
    """)
    c.execute("PRAGMA query_only=ON")
    return c


def test_physical_case_only_and_exact_negative_control():
    conn = _db()
    flagged = detect_one("Which city is named?",
       "SELECT id FROM schools s WHERE s.city = 'little lever urban district'", conn)
    assert "VALUE_CASE_ONLY_PHYSICAL_MATCH" in flagged["flags"]
    assert all(v["strength"] == "PHYSICAL_OBSERVATION" for v in flagged["observations"])
    exact = detect_one("Which city is named?",
       "SELECT id FROM schools s WHERE s.city = 'Little Lever Urban District'", conn)
    assert "VALUE_CASE_ONLY_PHYSICAL_MATCH" not in exact["flags"]
    assert "VALUE_ABSENT_FROM_COLUMN" not in exact["flags"]


def test_value_absence_is_not_semantic_proof():
    r = detect_one("Which city is missing?",
        "SELECT id FROM schools WHERE city = 'Unseen City'", _db())
    assert "VALUE_ABSENT_FROM_COLUMN" in r["flags"]
    assert not any(o["strength"] == "SEMANTIC_PROOF" for o in r["observations"])


def test_unique_count_candidate_and_normal_negative_control():
    unique = detect_one("How many distinct customers placed orders?",
        "SELECT COUNT(*) FROM orders", _db())
    assert "EXPLICIT_UNIQUE_COUNT_WITHOUT_DISTINCT" in unique["flags"]
    correct = detect_one("How many distinct customers placed orders?",
        "SELECT COUNT(DISTINCT customer_id) FROM orders", _db())
    assert "EXPLICIT_UNIQUE_COUNT_WITHOUT_DISTINCT" not in correct["flags"]


def test_ratio_and_topk_signals_do_not_need_model_or_gold():
    r = detect_one("What is the ratio of A to B? Give top 2 entries.",
       "SELECT name FROM customers", _db())
    assert "RATIO_WITHOUT_COMPUTATION" in r["flags"]
    assert "EXPLICIT_TOPK_WITHOUT_LIMIT" in r["flags"]


def test_declared_fk_mismatch_requires_known_fk():
    good = detect_one("Show orders and customers",
       "SELECT o.id FROM orders o JOIN customers c ON o.customer_id = c.id", _db())
    assert "DECLARED_FK_JOIN_KEY_MISMATCH" not in good["flags"]
    bad = detect_one("Show orders and customers",
       "SELECT o.id FROM orders o JOIN customers c ON o.name = c.name", _db())
    assert "DECLARED_FK_JOIN_KEY_MISMATCH" in bad["flags"]


def test_unqualified_ambiguous_probe_abstains():
    r = detect_one("Find a customer with name Alex",
       "SELECT o.id FROM orders o JOIN customers c ON o.customer_id=c.id WHERE name='Alex'", _db())
    assert "VALUE_ABSENT_FROM_COLUMN" not in r["flags"]
    assert "EQ_OPERAND_UNRESOLVED" in r["abstentions"]


def test_casefold_physical_probe_ignores_sql_non_string_literal():
    r = detect_one("How many records?",
       "SELECT COUNT(*) FROM schools WHERE id=123", _db())
    assert "VALUE_CASE_ONLY_PHYSICAL_MATCH" not in r["flags"]


def test_alias_parser_has_not_consumed_join_keyword():
    a = _aliases("SELECT * FROM orders o JOIN customers c ON o.customer_id=c.id",
                 {"orders": "orders", "customers": "customers"})
    assert a["o"] == "orders"
    assert a["c"] == "customers"


def test_original_ex_wrong_enrichment_not_causal_accuracy():
    rows = [
        {"case_id":"1","db_id":"one","execution_success":True,"strict_ex":False,"flags":[SIGNALS[0]]},
        {"case_id":"2","db_id":"one","execution_success":True,"strict_ex":True,"flags":[]},
        {"case_id":"3","db_id":"two","execution_success":False,"strict_ex":False,"flags":[SIGNALS[0]]},
    ]
    m = _metrics(rows, SIGNALS[0])
    assert m["flagged_n"] == 1 and m["flagged_wrong"] == 1
    assert m["n_executable"] == 2
    assert m["wrong_sql_coverage"] == 1.0


def test_empty_wilson_and_zero_division():
    assert _wilson(0, 0) is None
    assert _metrics([], SIGNALS[0])["error_enrichment_ppv"] is None
