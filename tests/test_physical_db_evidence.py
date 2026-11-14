"""Physical grounding must be reproducible, bounded and gold-blind."""
import sqlite3

from evaluation.physical_db_evidence import verified_db_facts
from semantic_sql.catalog import DatabaseCatalog


def test_only_verified_keys_and_exact_value_hits(tmp_path):
    path = tmp_path / "probe.sqlite"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE people (PersonID INTEGER PRIMARY KEY, FullName TEXT, Region TEXT);
        CREATE TABLE payments (
          PaymentID INTEGER PRIMARY KEY, PersonID INTEGER, Amount REAL,
          FOREIGN KEY(PersonID) REFERENCES people(PersonID)
        );
        INSERT INTO people VALUES
          (1, 'Harlan', 'West'), (2, 'Rita', 'East');
        INSERT INTO payments VALUES (10,1,60.0), (11,1,20.0);
        """
    )
    con.close()
    catalog = DatabaseCatalog.from_sqlite(path)
    report = verified_db_facts(path,catalog,{"people","payments"},'How much did "Harlan" pay?')
    assert "people.PersonID" in report.text
    assert "payments.PersonID" in report.text
    assert "people.FullName = 'Harlan'" in report.text
    assert "not a proven 1:1" in report.text
    assert report.value_matches == 1
    assert report.probes <= 24
    assert "Rita" not in report.text


def test_empty_or_unverified_evidence_never_invents_facts(tmp_path):
    path = tmp_path / "probe.sqlite"
    con = sqlite3.connect(path)
    con.executescript("CREATE TABLE x (id INT, title TEXT); INSERT INTO x VALUES (1, 'A');")
    con.close()
    c = DatabaseCatalog.from_sqlite(path)
    d = verified_db_facts(path, c, {"x"},'Find "Not Here".')
    assert "Not Here" not in d.text
    assert d.value_matches == 0
    assert d.probes <= 24


def test_quoted_table_column_names_do_not_become_sql_injection(tmp_path):
    path = tmp_path / "probe.sqlite"
    con = sqlite3.connect(path)
    con.executescript('CREATE TABLE "say""hi" ("named""col" TEXT PRIMARY KEY);')
    con.execute('INSERT INTO "say""hi" VALUES (?)', ("O'Reilly",))
    con.commit()
    con.close()
    c=DatabaseCatalog.from_sqlite(path)
    d=verified_db_facts(path, c, {'say"hi'},'Find "O\'Reilly".')
    assert d.value_matches == 1
    assert c.tables


def test_scan_budget_is_hard_limit(tmp_path):
    path = tmp_path/"probe.sqlite"
    con = sqlite3.connect(path)
    con.executescript("CREATE TABLE t (a TEXT,b TEXT,c TEXT);")
    con.close()
    c=DatabaseCatalog.from_sqlite(path)
    d=verified_db_facts(path,c,{"t"},'Compare "alpha" and "beta".',max_scans=1)
    assert d.probes <= 1
