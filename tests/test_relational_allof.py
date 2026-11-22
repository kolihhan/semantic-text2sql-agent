from __future__ import annotations

import sqlite3

import pytest

from semantic_sql.relational_allof import find_allof_counterexample
from semantic_sql.execution import execute_readonly


SQL = ("SELECT COUNT(DISTINCT u.id) FROM users AS u JOIN badges AS b "
       "ON u.id = b.user_id WHERE b.role IN ('Teacher', 'Supporter')")


@pytest.fixture
def db(tmp_path):
    path=tmp_path/"roles.sqlite"
    with sqlite3.connect(path) as con:
        con.executescript("""
        PRAGMA foreign_keys=ON;
        CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE badges(id INTEGER PRIMARY KEY, user_id INTEGER,
                            role TEXT, FOREIGN KEY(user_id) REFERENCES users(id));
        INSERT INTO users VALUES(1,'one'),(2,'two'),(3,'three');
        INSERT INTO badges VALUES(1,1,'Teacher'),(2,1,'Supporter'),
                                 (3,2,'Teacher'),(4,3,'Supporter');
        """)
    return path


def test_physical_relational_counterexample_produces_real_answer_difference(db):
    before=execute_readonly(db,SQL).rows
    witness=find_allof_counterexample(
        db,"How many users have both Teacher and Supporter roles?",SQL)
    assert before==((3,),)
    assert witness is not None
    assert (witness.union_count,witness.intersection_count)==(3,1)
    assert execute_readonly(db,witness.alternative).rows==((1,),)
    assert "HAVING COUNT(DISTINCT" in witness.alternative
    assert witness.parent=="users" and witness.child=="badges"
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT COUNT(*) FROM badges").fetchone()[0]==4


def test_ambiguous_question_never_triggers(db):
    for question in (
        "How many users have either Teacher or Supporter roles?",
        "How many users have Teacher and Supporter roles?",
        "How many users have both Teacher and Auditor roles?",
    ):
        assert find_allof_counterexample(db,question,SQL) is None


def test_already_correct_relational_division_not_rewritten(db):
    true_sql=(
        "SELECT COUNT(*) FROM (SELECT u.id FROM users AS u JOIN badges AS b "
        "ON u.id = b.user_id WHERE b.role IN ('Teacher', 'Supporter') "
        "GROUP BY u.id HAVING COUNT(DISTINCT b.role) = 2)"
    )
    assert execute_readonly(db,true_sql).rows==((1,),)
    assert find_allof_counterexample(
        db,"How many users have both Teacher and Supporter roles?",true_sql) is None


def test_no_physical_union_vs_intersection_difference_abstains(tmp_path):
    path=tmp_path/"same.sqlite"
    with sqlite3.connect(path) as con:
        con.executescript("""
        CREATE TABLE users(id INTEGER PRIMARY KEY);
        CREATE TABLE badges(id INTEGER PRIMARY KEY,
          user_id INTEGER REFERENCES users(id),role TEXT);
        INSERT INTO users VALUES(1);
        INSERT INTO badges VALUES(1,1,'Teacher'),(2,1,'Supporter');
        """)
    assert find_allof_counterexample(
        path,"How many users have both Teacher and Supporter roles?",SQL) is None


def test_missing_physical_relation_abstains(tmp_path):
    path=tmp_path/"fk_absent.sqlite"
    with sqlite3.connect(path) as con:
        con.executescript("""
        CREATE TABLE users(id INTEGER PRIMARY KEY);
        CREATE TABLE badges(id INTEGER PRIMARY KEY,user_id INTEGER,role TEXT);
        INSERT INTO users VALUES(1),(2);
        INSERT INTO badges VALUES(1,1,'Teacher'),(2,1,'Supporter'),
                                 (3,2,'Teacher');
        """)
    assert find_allof_counterexample(
        path,"How many users have both Teacher and Supporter roles?",SQL) is None


@pytest.mark.parametrize("sql", [
    SQL+" LIMIT 1",
    SQL+" GROUP BY u.id",
    SQL.replace("u.id = b.user_id","u.id=b.id"),
    SQL.replace("COUNT(DISTINCT u.id)","COUNT(*)"),
    SQL.replace("WHERE b.role IN","WHERE u.name IN"),
    SQL.replace("FROM users AS u JOIN badges AS b","FROM badges AS b JOIN users AS u"),
])
def test_unsafe_or_unsupported_sql_structure_abstains(db,sql):
    assert find_allof_counterexample(
        db,"How many users have both Teacher and Supporter roles?",sql) is None
