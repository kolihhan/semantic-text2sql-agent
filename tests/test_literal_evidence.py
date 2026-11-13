"""Opt-in exact-value evidence must not mutate or silently change baseline."""
import sqlite3

from semantic_sql.agent import SemanticSQLService
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.literal_evidence import append_literal_evidence, exact_literal_evidence, quoted_literals


def db_fixture(tmp_path):
    db = tmp_path / "example.db"
    with sqlite3.connect(db) as con:
        con.executescript("""
        CREATE TABLE events (id INTEGER PRIMARY KEY, title TEXT);
        CREATE TABLE attendees (person TEXT, event_id INTEGER REFERENCES events(id));
        INSERT INTO events VALUES (1, 'October Meeting'), (2, 'November Meeting');
        INSERT INTO attendees VALUES ('Mina', 1), ('Jo', 2);
        """)
    return db


def test_quoted_entities_found_with_exact_readonly_sql(tmp_path):
    catalog = DatabaseCatalog.from_sqlite(db_fixture(tmp_path))
    assert quoted_literals('Who came to "October Meeting"?') == ("October Meeting",)
    assert exact_literal_evidence(catalog, "Who came to 'October Meeting'?") == (
        {"literal": "October Meeting", "table": "events", "column": "title"},
    )
    schema = catalog.schema_text()
    enhanced = append_literal_evidence(catalog, 'Who came to "October Meeting"?',
                                       schema_context=schema)
    assert enhanced.startswith(schema + "\n")
    assert '"events"."title" contains "October Meeting"' in enhanced
    assert "attendees" in enhanced


def test_does_not_guess_unknown_entity_or_emit_negative_claims(tmp_path):
    catalog = DatabaseCatalog.from_sqlite(db_fixture(tmp_path))
    schema = catalog.schema_text()
    assert append_literal_evidence(catalog, "Who came to October Meeting?", schema_context=schema) == schema
    assert append_literal_evidence(catalog, 'Who came to "March Meeting"?', schema_context=schema) == schema
    assert exact_literal_evidence(catalog, 'Who came to "October Meeting"?', max_probes=0) == ()


def test_sql_injection_is_only_parameter_bound(tmp_path):
    catalog = DatabaseCatalog.from_sqlite(db_fixture(tmp_path))
    assert exact_literal_evidence(catalog, """Who came to "October Meeting'); DROP TABLE events;--"?""") == ()
    assert len(catalog.tables) == 2
    with sqlite3.connect(catalog.database) as db:
        assert db.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 2


def test_optin_reaches_generator_and_repair_but_default_preserved(tmp_path):
    db = db_fixture(tmp_path)
    question = 'Who came to "October Meeting"?'

    class Provider:
        def __init__(self):
            self.messages = []

        def complete_text(self, *, system, user):
            self.messages.append(user)
            return ("SELECT attendees.person FROM attendees JOIN events "
                    "ON attendees.event_id = events.id WHERE events.title = 'October Meeting'")

    before = Provider()
    service = SemanticSQLService(db, provider=before)
    assert service.ask(question).status == "ok"
    assert len(before.messages) == 1
    assert "Verified exact quoted-value hits" not in before.messages[0]

    after = Provider()
    service = SemanticSQLService(db, provider=after, literal_grounding=True)
    result = service.ask(question)
    assert result.status == "ok"
    assert result.rows == (("Mina",),)
    assert "Verified exact quoted-value hits" in after.messages[0]


def test_quoted_literal_budget(tmp_path):
    catalog = DatabaseCatalog.from_sqlite(db_fixture(tmp_path))
    assert len(quoted_literals(' "a" "b" "c" "d" "e" ')) == 4
    assert exact_literal_evidence(catalog, '"October Meeting"', max_vm_steps_per_probe=1) == ()
