"""Relational metadata must be true for arbitrary SQLite schemas, not BIRD cases."""
import sqlite3

import pytest

from evaluation.relational_contract import inspect_constraints


def _db(tmp_path, ddl):
    path=tmp_path/"test.sqlite"
    with sqlite3.connect(path) as con:
        con.execute("PRAGMA foreign_keys=ON")
        con.executescript(ddl)
    return path


def test_one_to_many_and_optional_fk_and_unique_keys(tmp_path):
    db=_db(tmp_path, """
        CREATE TABLE authors(
            author_id INTEGER PRIMARY KEY,
            email TEXT UNIQUE,
            display_name TEXT
        );
        CREATE TABLE books(
            id INTEGER PRIMARY KEY,
            author_id INTEGER,
            title TEXT,
            FOREIGN KEY (author_id) REFERENCES authors(author_id)
        );
    """)
    m=inspect_constraints(db)
    assert any(k.table=="authors" and k.columns==("author_id",)
               and k.kind=="PRIMARY_KEY" for k in m.keys)
    assert any(k.table=="authors" and k.columns==("email",)
               and k.kind=="UNIQUE_KEY" for k in m.keys)
    assert any(r.source=="books" and r.target=="authors"
               and r.target_key_declared and r.nullable for r in m.relations)
    assert "display_name" not in m.render().split("UNIQUE_KEY")[-1]


def test_rejects_inferred_links_and_partial_unique(tmp_path):
    db=_db(tmp_path, """
       CREATE TABLE a (id TEXT, label TEXT);
       CREATE TABLE b (id TEXT, a_id TEXT, label TEXT);
       CREATE UNIQUE INDEX only_special ON b(label) WHERE a_id IS NOT NULL;
       CREATE UNIQUE INDEX expression_unique ON a(lower(label));
    """)
    m=inspect_constraints(db)
    assert m.relations==()
    assert m.keys==()
    assert m.ignored_partial_or_expression_indexes==2
    assert m.render()==""


def test_composite_fk_and_composite_primary_key(tmp_path):
    db=_db(tmp_path, """
        CREATE TABLE parent(
          workspace TEXT,
          entry INTEGER,
          content TEXT,
          PRIMARY KEY(workspace,entry)
        );
        CREATE TABLE child(
          id INTEGER PRIMARY KEY,
          workspace TEXT NOT NULL,
          parent_entry INTEGER NOT NULL,
          FOREIGN KEY(workspace,parent_entry) REFERENCES parent(workspace,entry)
        );
    """)
    m=inspect_constraints(db)
    match=[r for r in m.relations if r.source=="child"]
    assert len(match)==1
    assert match[0].source_columns==("workspace","parent_entry")
    assert match[0].target_columns==("workspace","entry")
    assert match[0].target_key_declared and not match[0].nullable
    assert '"workspace", "entry"' in m.render()


def test_selected_tables_restrict_context_without_fabricating_fk(tmp_path):
    db=_db(tmp_path, """
        CREATE TABLE parent (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE child (id INTEGER, parent_id INTEGER,
              FOREIGN KEY (parent_id) REFERENCES parent(id));
        CREATE TABLE distractor (child_id INTEGER, name TEXT);
    """)
    m=inspect_constraints(db,selected_tables={"child"})
    assert m.relations==()
    assert m.render()==""
    with pytest.raises(ValueError,match="absent"):
        inspect_constraints(db,selected_tables={"not_a_table"})


def test_quoted_sqlite_identifiers_and_nonkey_names_do_not_inject(tmp_path):
    db=_db(tmp_path, """
        CREATE TABLE "weird""name"("k""id" TEXT PRIMARY KEY, normal TEXT);
        CREATE TABLE "reference"("a" TEXT REFERENCES "weird""name"("k""id"));
    """)
    m=inspect_constraints(db)
    assert any(k.table=='weird"name' for k in m.keys)
    assert any(r.target=='weird"name' and r.target_key_declared for r in m.relations)
    assert 'weird""name' in m.render()
