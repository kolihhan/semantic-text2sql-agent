from __future__ import annotations

import sqlite3

from evaluation.implicit_fk_grounding import resolve_implicit_fk_targets
from evaluation.schema_context import select_schema_context
from semantic_sql.catalog import DatabaseCatalog


def _catalog(tmp_path, ddl):
    path=tmp_path/"foreign_keys.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript(ddl)
    return DatabaseCatalog.from_sqlite(path)


def test_implicit_simple_pk_is_actual_target_not_literal_none(tmp_path):
    cat=_catalog(tmp_path, """
        CREATE TABLE parent(id INTEGER PRIMARY KEY, val TEXT);
        CREATE TABLE child(cid INTEGER PRIMARY KEY, parent_id INTEGER,
                           FOREIGN KEY(parent_id) REFERENCES parent);
    """)
    assert ("child","parent_id","parent",None) in cat.foreign_keys
    fixed, diag=resolve_implicit_fk_targets(cat)
    assert ("child","parent_id","parent","id") in fixed.foreign_keys
    assert diag.original_missing==diag.inferred==1 and diag.unresolved==0
    original,_=select_schema_context(cat,tmp_path,"demo","parent child")
    corrected,_=select_schema_context(fixed,tmp_path,"demo","parent child")
    assert "parent.None" in original
    assert "FK child.parent_id -> parent.id" in corrected
    assert cat.foreign_keys != fixed.foreign_keys


def test_composite_implicit_key_uses_pragmas_order_not_alphabetical(tmp_path):
    cat=_catalog(tmp_path,"""
        CREATE TABLE parent(b TEXT, a TEXT, PRIMARY KEY (b,a));
        CREATE TABLE child(x TEXT,y TEXT, FOREIGN KEY(x,y) REFERENCES parent);
    """)
    fixed,diag=resolve_implicit_fk_targets(cat)
    assert ("child","x","parent","b") in fixed.foreign_keys
    assert ("child","y","parent","a") in fixed.foreign_keys
    assert diag.inferred==2 and diag.unresolved==0


def test_existing_explicit_fk_unchanged(tmp_path):
    cat=_catalog(tmp_path,"""
        CREATE TABLE parent(id INTEGER PRIMARY KEY, k TEXT UNIQUE);
        CREATE TABLE child(ref TEXT, FOREIGN KEY(ref) REFERENCES parent(k));
    """)
    fixed,diag=resolve_implicit_fk_targets(cat)
    assert fixed.foreign_keys==cat.foreign_keys
    assert diag.original_explicit==1 and diag.inferred==0


def test_implicit_without_explicit_primary_key_is_not_guessed(tmp_path):
    cat=_catalog(tmp_path,"""
        CREATE TABLE parent(name TEXT, value INTEGER);
        CREATE TABLE child(x TEXT, FOREIGN KEY(x) REFERENCES parent);
    """)
    fixed,diag=resolve_implicit_fk_targets(cat)
    assert not fixed.foreign_keys
    assert diag.unresolved==1 and diag.inferred==0
