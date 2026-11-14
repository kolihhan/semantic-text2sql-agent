"""Read-only audits must never silently substitute semantic labels for physical evidence."""
import sqlite3

from evaluation.qwen_literal_grounding_audit import (
    extract_simple_equalities, check_literal, LiteralPredicate,
)


def test_extract_alias_and_unaliased_sql_only_where():
    schema={"drivers":("driverId","nationality"),"qualifying":("driverId","q2")}
    sql=(
        "SELECT drivers.nationality FROM drivers JOIN qualifying "
        "ON drivers.driverId=qualifying.driverId "
        "WHERE qualifying.q2 = '0:01:15' AND qualifying.driverId = 7"
    )
    items=extract_simple_equalities(sql,schema)
    assert items == (LiteralPredicate("qualifying","q2","0:01:15"),)


def test_extract_join_alias_and_sql_escaped_quotes():
    schema={"drivers":("surname",),"races":("year",)}
    sql=("SELECT * FROM drivers d JOIN races r ON d.surname = r.name "
         "WHERE d.surname = 'O''Brien' AND r.year=2007")
    items=extract_simple_equalities(sql,schema)
    assert items == (LiteralPredicate("drivers","surname","O'Brien"),)


def test_conservative_unsupported_dynamic_and_or_conditions():
    schema={"mytable":("name",)}
    assert extract_simple_equalities(
        "SELECT * FROM mytable t WHERE LOWER(t.name) = 'Alice'",schema) == ()
    assert extract_simple_equalities(
        "SELECT * FROM mytable t WHERE t.name IN ('Alice')",schema) == ()
    assert extract_simple_equalities(
        "SELECT * FROM mytable t WHERE t.name = 'Alice' OR t.name = 'Bob'",schema) == ()


def test_real_db_time_and_date_representation_without_write(tmp_path):
    path=tmp_path/"real.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE t (q2 TEXT, created TEXT)")
        db.executemany("INSERT INTO t VALUES (?, ?)",[
            ("1:15.018","1994-02-19"),("1:15.593","1994-02-19"),("2:12.119","1995-05-02")
        ])
    a=check_literal(path,LiteralPredicate("t","q2","0:01:15"))
    assert a.status == "time_precision_match"
    assert a.positive_count == 2
    assert a.replacement == "1:15.%" and a.operator == "LIKE"
    b=check_literal(path,LiteralPredicate("t","created","1994/2/19"))
    assert b.status == "date_format_match"
    assert b.replacement == "1994-02-19" and b.operator == "="
    c=check_literal(path,LiteralPredicate("t","created","1994-02-19"))
    assert c.status == "exact_present"


def test_no_false_evidence_for_missing_or_unsupported_values(tmp_path):
    path=tmp_path/"real.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE t (s TEXT)")
        db.execute("INSERT INTO t VALUES ('Mary')")
    a=check_literal(path,LiteralPredicate("t","s","Other"))
    assert a.status == "no_positive_evidence"
    b=check_literal(path,LiteralPredicate("t","s","mary"))
    assert b.status == "casefold_match"
    assert b.replacement == "Mary"
    assert a.positive_count==0


def test_ambiguous_dates_are_not_rewritten(tmp_path):
    path=tmp_path/"real.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE t (d TEXT)")
        db.executemany("INSERT INTO t VALUES (?)",[("02/03/1994",),("03/02/1994",)])
    d=check_literal(path,LiteralPredicate("t","d","03/02/1994"))
    assert d.status == "exact_present"


def test_schema_identifier_injection_is_safely_escaped(tmp_path):
    path=tmp_path/"real.sqlite"
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE "t""odd" ("col""q" TEXT)')
        db.execute('INSERT INTO "t""odd" VALUES (?)',("A'b",))
    r=check_literal(path,LiteralPredicate('t"odd','col"q',"A'b"))
    assert r.status=="exact_present"
