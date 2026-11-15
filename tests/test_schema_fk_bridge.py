from pathlib import Path
import sqlite3

from evaluation.schema_fk_bridge import augment_fk_paths, shortest_path
from semantic_sql.catalog import DatabaseCatalog


def sample_db(tmp_path):
    db = tmp_path / "join_chain.sqlite"
    with sqlite3.connect(db) as con:
        con.executescript(
            "CREATE TABLE Users(id INTEGER PRIMARY KEY, name TEXT);"
            "CREATE TABLE Enrollments(id INTEGER PRIMARY KEY, owner_ref INTEGER,"
            " FOREIGN KEY(owner_ref) REFERENCES Users(id));"
            "CREATE TABLE Classes(id INTEGER PRIMARY KEY, prev_ref INTEGER,"
            " FOREIGN KEY(prev_ref) REFERENCES Enrollments(id));"
            "CREATE TABLE Assignments(id INTEGER PRIMARY KEY, prev_ref INTEGER,"
            " FOREIGN KEY(prev_ref) REFERENCES Classes(id));"
            "CREATE TABLE Instructors(id INTEGER PRIMARY KEY, prev_ref INTEGER,"
            " FOREIGN KEY(prev_ref) REFERENCES Assignments(id));"
        )
    return db


def test_bridge_is_physical_schema_only(tmp_path):
    db = sample_db(tmp_path)
    catalog = DatabaseCatalog.from_sqlite(db)
    path = shortest_path(catalog, "Users", "Instructors")
    assert path is not None and len(path[1]) == 4
    baseline, added, info = augment_fk_paths(
        catalog, Path(tmp_path), "join_chain",
        "Which users have instructors?"
    )
    assert added.startswith(baseline)
    assert info["intervention"] == "schema_fk_routes"
    assert info["missing_bridge_table_count"] >= 1
    assert "Classes(" in added
    assert "Enrolments(" not in added
    assert "Assignments.prev_ref = Classes.id" in added


def test_no_relation_does_not_invent_foreign_keys(tmp_path):
    db = sample_db(tmp_path)
    catalog = DatabaseCatalog.from_sqlite(db)
    baseline, added, diag = augment_fk_paths(
        catalog, Path(tmp_path), "join_chain",
        "How many students have a birthday?"
    )
    assert baseline == added
    assert not diag["routes"]


def test_anchor_requires_actual_table_name_not_generic_id_or_name(tmp_path):
    db = sample_db(tmp_path)
    catalog = DatabaseCatalog.from_sqlite(db)
    before, after, diag = augment_fk_paths(
        catalog, Path(tmp_path), "join_chain",
        "Show the ID and name for all records."
    )
    assert before == after
    assert diag["anchors"] == []


def test_no_fake_path_to_missing_or_disconnected_table(tmp_path):
    db = sample_db(tmp_path)
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE MissingRelation(id INTEGER PRIMARY KEY)")
    catalog = DatabaseCatalog.from_sqlite(db)
    assert shortest_path(catalog, "Users", "MissingRelation") is None
    assert shortest_path(catalog, "NotATable", "Users") is None
