"""CPU source-only role-scoping gates. No Qwen, gold SQL or GPU."""
from __future__ import annotations

import sqlite3
from evaluation.role_scoped_grounding import (
    lexical_role_match, role_scoped_candidates, evidence_string,
    source_only_role_evidence,
)


def test_race_role_rejects_surname_same_value():
    evidence=[
        {"value":"White","table":"person","column":"last_name"},
        {"value":"White","table":"demographic","column":"race_name"},
    ]
    got=role_scoped_candidates("What share of people have race White?",evidence)
    assert len(got)==1
    assert got[0]["column"]=="race_name"
    assert got[0]["basis"]=="column"
    assert "last_name" not in evidence_string(got)


def test_recipe_title_role_rejects_ingredient_same_string():
    evidence=[
        {"value":"Apricot","table":"recipe_ingredients","column":"ingredient"},
        {"value":"Apricot","table":"recipes","column":"title"},
    ]
    got=role_scoped_candidates("Find recipe title 'Apricot'.",evidence)
    assert [(g["table"],g["column"]) for g in got]==[("recipes","title")]


def test_professor_role_does_not_ground_student_name():
    evidence=[
        {"value":"Ogdon","table":"prof","column":"first_name"},
        {"value":"Ogdon","table":"student","column":"first_name"},
    ]
    got=role_scoped_candidates(
        "How many assistants did Professor Ogdon teach?",evidence)
    assert [(x["table"],x["column"]) for x in got]==[("prof","first_name")]


def test_fails_closed_for_untyped_value_and_ambiguous_roles():
    assert role_scoped_candidates("Show all rows with White",[
        {"value":"White","table":"student","column":"last_name"}])==[]
    assert role_scoped_candidates("How many have race White?",[
        {"value":"White","table":"persons","column":"race"},
        {"value":"White","table":"customers","column":"race"},
    ])==[]


def test_read_only_original_physical_sqlite_and_no_gold(tmp_path):
    path=tmp_path/"source.sqlite"
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE person(last_name TEXT, race TEXT);
            CREATE TABLE ingredients(name TEXT);
            INSERT INTO person VALUES('White','White');
            INSERT INTO ingredients VALUES('White');
        """)
    good=source_only_role_evidence(path,"How many are of race White?")
    assert len(good)==1 and good[0]["column"]=="race"
    bad=source_only_role_evidence(path,"Find ingredients of recipe 'White'")
    assert not bad
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM person").fetchone()[0]==1
