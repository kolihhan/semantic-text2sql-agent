import sqlite3
from scripts.predicate_contract_probe import inspect_predicate

def test_contract_negative_controls(tmp_path):
    db = tmp_path / "fixture.db"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE people(name TEXT, city TEXT)")
        c.execute("INSERT INTO people VALUES (?, ?)", ("O'Neil", "Paris"))
    cases = [
        ("Find people named 'O''Neil'", "SELECT name FROM people WHERE name = 'O''Neil'", "ABSTAIN"),
        ('Find "Rome" residents', "SELECT name FROM people WHERE city = 'Rome'", "OBSERVED"),
        ('Find "Paris" residents', "SELECT name FROM people WHERE city = 'Paris'", "OBSERVED"),
        ('Find "Paris" residents', "SELECT name FROM people WHERE city = 'Paris' AND name='x'", "ABSTAIN"),
        ('Find "Paris" residents', "SELECT name FROM people p JOIN people q ON p.name=q.name WHERE p.city='Paris'", "ABSTAIN"),
        ('Find "Paris" residents', "SELECT name FROM people WHERE city='Berlin'", "ABSTAIN"),
    ]
    for question, sql, expected in cases:
        assert inspect_predicate(question, sql, db)["status"] == expected
    assert inspect_predicate('Find "Rome" residents', "SELECT name FROM people WHERE city='Rome'", db)["observations"][0]["exact_exists"] is False
    assert inspect_predicate('Find "Paris" residents', "SELECT name FROM people WHERE city='Paris'", db)["observations"][0]["exact_exists"] is True

def test_fail_closed_and_no_mutation(tmp_path):
    db = tmp_path / "readonly.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE people(name TEXT, city TEXT)")
        conn.execute("INSERT INTO people VALUES ('Alice', 'Paris')")
    original = db.read_bytes()
    cases = [
        ('Find "Paris" residents', "SELECT name FROM missing WHERE city='Paris'", "ABSTAIN"),
        ('Find "Paris" residents', "SELECT name FROM people WHERE nonexistent='Paris'", "ABSTAIN"),
        ('Find "Paris" residents', "SELECT name FROM people WHERE city='Paris' OR name='Alice'", "ABSTAIN"),
        ('Find "Paris" residents', "SELECT name FROM people WHERE city='Paris' AND name='Alice'", "ABSTAIN"),
        ('Find "Paris" residents', "SELECT name FROM people WHERE city='Paris' UNION SELECT name FROM people", "ABSTAIN"),
        ('Find "Paris" residents', "SELECT name FROM people WHERE city='Paris' AND 1=1", "ABSTAIN"),
        ('Find Paris residents', "SELECT name FROM people WHERE city='Paris'", "ABSTAIN"),
        ('Find "Paris" residents', "SELECT name FROM people WHERE city=42", "ABSTAIN"),
        ('Find "Paris" residents', "SELECT name FROM people WHERE city='Paris'", "OBSERVED"),
    ]
    for question, sql, status in cases:
        before_sql = sql
        output = inspect_predicate(question, sql, db)
        assert output["status"] == status, (sql, output)
        assert sql == before_sql
        assert all(x.get("semantic_correctness") == "UNKNOWN" for x in output["observations"])
    assert db.read_bytes() == original

def test_missing_database_does_not_create_file(tmp_path):
    db = tmp_path / "nonexistent.db"
    result = inspect_predicate('Find "Paris" residents', "SELECT name FROM people WHERE city='Paris'", db)
    assert result["status"] == "ABSTAIN"
    assert not db.exists()


def test_single_table_unqualified_column_and_multi_statement_rejection(tmp_path):
    """Unqualified names resolve in a single-table scope; extra statements must abstain."""
    db = tmp_path / "scope.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE people(name TEXT, city TEXT)")
        conn.execute("INSERT INTO people VALUES ('Alice', 'Paris')")
    question = 'Find "Paris" residents'
    assert inspect_predicate(question, "SELECT name FROM people WHERE city='Paris'", db)["status"] == "OBSERVED"
    cases = [
        "SELECT name FROM people WHERE people.city='Paris'; SELECT 1",
        "SELECT name FROM people WHERE people.city='Paris' -- trailing comment",
    ]
    for sql in cases:
        assert inspect_predicate(question, sql, db)["status"] == "ABSTAIN", sql
    assert inspect_predicate(question, "SELECT name FROM people WHERE people.city='Paris'", db)["status"] == "OBSERVED"
    assert inspect_predicate(question, "SELECT name FROM people WHERE people.city='Paris';", db)["status"] == "OBSERVED"
    assert inspect_predicate(question, "SELECT name FROM people WHERE people.city='Paris';;", db)["status"] == "ABSTAIN"
