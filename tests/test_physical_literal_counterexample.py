"""Pure synthetic SQLite tests: never adapt rule based on real validation SQL/EX."""
import sqlite3

import pytest

from evaluation.physical_literal_counterexample import propose_literal_counterexample


@pytest.fixture
def sample_db(tmp_path):
    p = tmp_path / "s.sqlite"
    with sqlite3.connect(p) as con:
        con.executescript("""
        CREATE TABLE language(id INTEGER PRIMARY KEY, language_name TEXT);
        CREATE TABLE book(id INTEGER PRIMARY KEY, title TEXT, lang_id INT REFERENCES language(id));
        INSERT INTO language VALUES(1,'British-English'),(2,'British English!'),(3,'Spanish');
        INSERT INTO book VALUES(1,'one',1),(2,'two',2),(3,'three',3);
        CREATE TABLE categories(id INTEGER PRIMARY KEY, name TEXT);
        INSERT INTO categories VALUES (1, 'Fine Art'), (2, 'Modern Art'),(3,'Fine-Art');
        CREATE TABLE orders(id INTEGER PRIMARY KEY, category_id INTEGER, note TEXT);
        INSERT INTO orders VALUES (1,1,'British-English'), (2,2,'Spanish');
        """)
    return p


def prompt(q):
    return f"Question: {q}\nSchema context:\nFake(no effect)"


def test_unique_physical_counterexample_and_join_alias(sample_db):
    sql = ("SELECT title FROM book JOIN language l ON book.lang_id=l.id "
           "WHERE l.language_name='SPANISH!'")
    p = propose_literal_counterexample(sample_db, prompt("Find SPANISH! books"), sql)
    assert p.changed
    assert p.reason == "unique_physical_normalized_equality_counterexample"
    assert p.base_table == "language" and p.column == "language_name"
    assert "l.language_name='Spanish'" in p.sql


def test_ambiguous_normalized_value_abstains(sample_db):
    sql = "SELECT id FROM language WHERE language_name = 'British English'"
    p = propose_literal_counterexample(sample_db, prompt("British English"), sql)
    assert not p.changed and p.reason == "ambiguous_or_missing"


def test_exact_existing_value_never_rewritten(sample_db):
    sql = "SELECT id FROM language WHERE language_name = 'Spanish'"
    p = propose_literal_counterexample(sample_db, prompt("Spanish"), sql)
    assert not p.changed and p.reason == "no_or_multiple_unique_counterexamples"


def test_value_not_present_in_question_or_evidence_abstains(sample_db):
    sql = "SELECT id FROM language WHERE language_name = 'SPANISH!'"
    p = propose_literal_counterexample(sample_db, prompt("Find a book"), sql)
    assert not p.changed and p.reason == "no_explicit_question_literal_equality"


def test_ambiguous_multi_filter_counterexample_abstains(sample_db):
    sql = ("SELECT * FROM language l JOIN categories c ON 1=1 "
           "WHERE l.language_name='Spanish!' AND c.name='Modern-Art'")
    p = propose_literal_counterexample(sample_db, prompt("Spanish! Modern-Art"), sql)
    assert not p.changed and p.reason == "no_or_multiple_unique_counterexamples"


def test_or_and_nested_subquery_fail_closed(sample_db):
    p = propose_literal_counterexample(
        sample_db, prompt("Spanish!"),
        "SELECT * FROM language WHERE language_name='Spanish!' OR id=2")
    assert not p.changed and p.reason == "fail_closed_ValueError"
    p = propose_literal_counterexample(
        sample_db, prompt("Spanish!"),
        "SELECT * FROM language WHERE id IN (SELECT id FROM language WHERE language_name='Spanish!')")
    assert not p.changed and p.reason == "fail_closed_ValueError"


def test_multi_statement_never_executed(sample_db):
    p = propose_literal_counterexample(
        sample_db, prompt("Spanish!"),
        "SELECT * FROM language WHERE language_name='Spanish!'; DROP TABLE language;")
    assert not p.changed and p.reason == "fail_closed_ValueError"
    with sqlite3.connect(sample_db) as con:
        assert con.execute("SELECT COUNT(*) FROM language").fetchone()[0] == 3


def test_quoted_columns_and_literal_escape(tmp_path):
    db = tmp_path / "quoted.sqlite"
    with sqlite3.connect(db) as con:
        con.execute('CREATE TABLE "Movie Tags" ("Tag Name" TEXT)')
        con.execute('INSERT INTO "Movie Tags" VALUES (?)', ("Director's-Cut",))
    sql = ("SELECT \"Tag Name\" FROM \"Movie Tags\" AS mt "
           "WHERE mt.\"Tag Name\"='Director''s Cut'")
    p = propose_literal_counterexample(db, prompt("Director's Cut"), sql)
    assert p.changed and "Director''s-Cut" in p.sql


def test_case_and_punctuation_unique_but_not_fuzzy(sample_db):
    sql = "SELECT id FROM language WHERE language_name='SPANISH!'"
    p = propose_literal_counterexample(sample_db, prompt("Spanish!"), sql)
    assert p.changed and "'Spanish'" in p.sql
    # A synonym, typo or added word must NOT become a fabricated correction.
    p = propose_literal_counterexample(
        sample_db, prompt("Spanish Names"),
        "SELECT id FROM language WHERE language_name='Spanish Names'")
    assert not p.changed


def test_ambiguous_unqualified_join_column_abstains(sample_db):
    sql = ("SELECT * FROM language l JOIN orders o ON 1=1 "
           "WHERE id='SPANISH!'")
    p = propose_literal_counterexample(sample_db, prompt("Spanish!"), sql)
    assert not p.changed


def test_external_evidence_literal_supported(sample_db):
    user = ("Question: Which books?\nSchema context:\nbook(id,title)\n"
            "External evidence:\nlanguage_name = 'Spanish!'")
    sql = "SELECT id FROM language WHERE language_name='SPANISH!'"
    p = propose_literal_counterexample(sample_db, user, sql)
    assert p.changed


def test_missing_database_abstains(tmp_path):
    p = propose_literal_counterexample(
        tmp_path / "missing.sqlite", prompt("Spanish!"),
        "SELECT * FROM language WHERE language_name='Spanish!'")
    assert not p.changed
