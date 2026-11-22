import sqlite3
from pathlib import Path

from experiments.bird_qwen9b_physical_answer_contract16 import (
    PROTOCOL, N, GATE, physical_answer_evidence,
)


def test_bird_physical_evidence_is_readonly_bounded_and_schema_grounded(tmp_path: Path):
    db=tmp_path/"bird.sqlite"
    con=sqlite3.connect(db)
    con.executescript("""
        PRAGMA foreign_keys=ON;
        CREATE TABLE parent(id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE child(id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parent(id), amount INTEGER);
        INSERT INTO parent VALUES(1,'one'),(2,'two');
        INSERT INTO child VALUES(10,1,3),(11,1,5),(12,2,7);
    """)
    con.commit()
    con.close()
    evidence=physical_answer_evidence(
        db, ["SELECT COUNT(*) FROM parent JOIN child ON parent.id=child.parent_id"]
    )
    assert "parent: rows=2" in evidence
    assert "child: rows=3" in evidence
    assert "parent_id->parent.id" in evidence
    assert "one" in evidence
    assert "NOT GOLD SQL" in evidence
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT COUNT(*) FROM child").fetchone()[0]==3


def test_bird_only_prospective_contract():
    assert "BIRD" in PROTOCOL and "SPIDER" not in PROTOCOL
    assert N==16 and GATE["min_selected_net"]==2


def test_unrecognized_sql_table_does_not_execute(tmp_path: Path):
    db=tmp_path/"db.sqlite"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE safe(x INTEGER)")
        con.execute("INSERT INTO safe VALUES (7)")
    x=physical_answer_evidence(db, ["SELECT * FROM does_not_exist"])
    assert "does_not_exist:" not in x
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT x FROM safe").fetchone()[0]==7
