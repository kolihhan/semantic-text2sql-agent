"""Synthetic Question-Schema-SQL count-unit gate tests (no publisher gold)."""
import sqlite3

from semantic_sql.question_schema_sql_unit_gate import decide


def make_conn():
    conn=sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript("""
    CREATE TABLE Patient(ID INTEGER PRIMARY KEY, Description TEXT);
    CREATE TABLE Laboratory(
      lab_id INTEGER PRIMARY KEY,
      ID INTEGER NOT NULL REFERENCES Patient(ID),
      CRP TEXT
    );
    CREATE TABLE customers(CustomerID INTEGER PRIMARY KEY, Currency TEXT);
    CREATE TABLE transactions(
      TransactionID INTEGER PRIMARY KEY,
      CustomerID INTEGER REFERENCES customers(CustomerID),
      Amount REAL
    );
    CREATE TABLE independent_readings(
      ReadingID INTEGER PRIMARY KEY, PatientID INTEGER
    );
    INSERT INTO Patient VALUES(1,NULL),(2,NULL);
    INSERT INTO Laboratory VALUES(11,1,'+'),(12,1,'+'),(13,2,'+');
    INSERT INTO customers VALUES(1,'EUR'),(2,'USD');
    INSERT INTO transactions VALUES(20,1,5.0),(21,1,7.0),(22,2,8.0);
    INSERT INTO independent_readings VALUES(100,1),(101,1);
    """)
    return conn


def test_conflict_requires_actual_one_to_many_fk_and_changed_population():
    con=make_conn()
    bad="SELECT COUNT(*) FROM Patient p JOIN Laboratory l ON p.ID=l.ID WHERE l.CRP='+'"
    r=decide("How many patients have a positive CRP laboratory result?",bad,con)
    assert r["decision"]=="CONFLICT", r
    assert r["reason"]=="ONE_TO_MANY_FK_PROVEN_DIFFERENT_ANSWER_UNIT"
    assert r["changed_result_boolean"]
    con.close()


def test_explicit_target_pk_distinct_is_unit_consistent_not_proven_correct():
    con=make_conn()
    r=decide("How many patients have a positive CRP result?",
       "SELECT COUNT(DISTINCT p.ID) FROM Patient p JOIN Laboratory l ON p.ID=l.ID WHERE l.CRP='+'",con)
    assert r["decision"]=="CONSISTENT"
    assert r["whole_sql_correctness_not_inferred"] is True
    con.close()


def test_count_transactions_preserved_even_when_join_has_customer_pk():
    con=make_conn()
    r=decide("How many transactions were made by EUR customers?",
       "SELECT COUNT(*) FROM transactions t JOIN customers c ON t.CustomerID=c.CustomerID WHERE c.Currency='EUR'",con)
    assert r["decision"]=="CONSISTENT"
    assert r["reason"]=="COUNT_CHILD_ROWS_JOINED_ONLY_TO_PK_PARENTS"
    con.close()


def test_no_fk_must_abstain_even_if_physical_multiplicity_exists():
    con=make_conn()
    r=decide("How many patients have independent readings?",
        "SELECT COUNT(*) FROM Patient p JOIN independent_readings r ON r.PatientID=p.ID",con)
    assert r["decision"]=="ABSTAIN"
    con.close()


def test_grouped_derived_and_ratio_ambiguities_all_abstain():
    con=make_conn()
    queries=[
        ("How many patients?","SELECT COUNT(*) FROM Patient GROUP BY Description"),
        ("What percentage of customers use EUR?","SELECT 100.0 * SUM(CASE WHEN Currency='EUR' THEN 1 END)/COUNT(*) FROM customers"),
        ("How many more patients than tests?","SELECT 3"),
        ("How many patients?","WITH x AS (SELECT * FROM Patient) SELECT COUNT(*) FROM x"),
        ("How many patients?","SELECT COUNT(*) FROM Patient UNION SELECT COUNT(*) FROM Laboratory"),
        ("How many people?","SELECT COUNT(*) FROM Laboratory"),
    ]
    for q,sql in queries:
        assert decide(q,sql,con)["decision"]=="ABSTAIN",sql
    con.close()


def test_count_directly_from_target_has_consistent_unit_only():
    con=make_conn()
    r=decide("How many customers are there?",
             "SELECT COUNT(*) FROM customers WHERE Currency='EUR'",con)
    assert r["decision"]=="CONSISTENT"
    con.close()


def test_no_actual_multiplicity_abstains_not_overconfident_conflict():
    con=make_conn()
    r=decide("How many patients have a result?",
       "SELECT COUNT(*) FROM Patient p JOIN Laboratory l ON p.ID=l.ID WHERE p.ID=2",con)
    assert r["decision"]=="ABSTAIN"
    assert r["reason"]=="ONE_TO_MANY_FK_BUT_NO_PHYSICAL_DIFFERENCE"
    con.close()


def test_implicit_sqlite_fk_target_is_abstain_not_runtime_crash():
    con=make_conn()
    con.executescript("""
    CREATE TABLE Parent2(parent_id INTEGER PRIMARY KEY);
    CREATE TABLE Child2(child_id INTEGER PRIMARY KEY,
                        parent_id INTEGER REFERENCES Parent2);
    INSERT INTO Parent2 VALUES(1);
    INSERT INTO Child2 VALUES(1,1),(2,1);
    """)
    out=decide("How many parent2 entries have children?",
               "SELECT COUNT(*) FROM Parent2 p JOIN Child2 c ON p.parent_id=c.parent_id",con)
    assert out["decision"]=="ABSTAIN",out
    con.close()
