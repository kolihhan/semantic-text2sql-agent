"""Source-first entity evidence: independent of model and benchmark gold."""
import sqlite3

from evaluation.source_first_value_anchor import observed_value_anchors
from semantic_sql.catalog import DatabaseCatalog


def _db(tmp_path):
    path = tmp_path / "synthetic.sqlite"
    with sqlite3.connect(path) as c:
        c.executescript("""
        CREATE TABLE cards(id INTEGER, name TEXT, setCode TEXT);
        CREATE TABLE sets(code TEXT, name TEXT);
        CREATE TABLE set_translations(setCode TEXT, language TEXT, translation TEXT);
        INSERT INTO cards VALUES(1,'Unique Widget','UW');
        INSERT INTO sets VALUES('UW','The Collection');
        INSERT INTO set_translations VALUES('UW','French','La Collection');
        """)
    return path


def test_observes_quoted_entity_in_table_outside_lexical_selection(tmp_path):
    path=_db(tmp_path)
    catalog=DatabaseCatalog.from_sqlite(path)
    r=observed_value_anchors(
        path,catalog,'What is the French set containing "Unique Widget"?',
        selected_tables={"sets", "set_translations"})
    assert ("cards","name","Unique Widget") in [
        (h.table,h.column,h.value) for h in r.hits]
    assert "cards.name" in r.outside_schema
    assert r.probes<=100
    assert r.ambiguous is False
    assert r.empty_probes_are_not_negative_evidence is True


def test_ambiguous_alias_cannot_be_claimed_as_unique(tmp_path):
    path=_db(tmp_path)
    with sqlite3.connect(path) as c:
        c.execute("UPDATE sets SET name='Unique Widget'")
    r=observed_value_anchors(
        path,DatabaseCatalog.from_sqlite(path),'"Unique Widget"',
        selected_tables={"sets"})
    assert r.ambiguous is True
    assert "cards.name" in r.outside_schema


def test_no_proof_if_value_missing_or_budget_exhausted(tmp_path):
    path=_db(tmp_path)
    c=DatabaseCatalog.from_sqlite(path)
    a=observed_value_anchors(path,c,'Find "No Such Phrase"',selected_tables=set())
    b=observed_value_anchors(path,c,'Find "Unique Widget"',selected_tables=set(),max_probes=0)
    assert not a.hits and not a.ambiguous
    assert not b.hits and b.probes==0
    assert not b.outside_schema


def test_quoted_identifiers_and_values_are_safe(tmp_path):
    path=tmp_path/"odd.sqlite"
    with sqlite3.connect(path) as c:
        c.execute('CREATE TABLE "odd""table" ("odd""column" TEXT)')
        c.execute('INSERT INTO "odd""table" VALUES (?)', ("A'b",))
    # Use a controlled catalog: the existing catalog PRAGMA quoting has a known gap.
    from semantic_sql.catalog import TableInfo
    c=DatabaseCatalog(path,{'odd"table':TableInfo('odd"table',('odd"column',))},())
    r=observed_value_anchors(path,c,'Find "A\'b"',selected_tables=set())
    assert r.hits[0].table=='odd"table'
    assert r.hits[0].column=='odd"column'



def test_quoted_literals_ignore_possessives_and_numeric_ids():
    from evaluation.source_first_value_anchor import _literals
    assert _literals("What's the driver's name?") == ()
    assert _literals("Which school's campus is larger?") == ()
    assert _literals('Find "174" but also "Blue Land".') == ("Blue Land",)
    assert _literals("Search 'Unique Widget' and 'TR004'") == ("Unique Widget", "TR004")
