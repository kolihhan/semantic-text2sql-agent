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
