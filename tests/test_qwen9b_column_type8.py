"""CPU-only gold-blind raw-column evidence tests; NOT new model EX results."""
import sqlite3
import json
from pathlib import Path
import pytest
from collections import Counter

from experiments import qwen9b_column_type8 as m


def local_db(tmp_path):
    db=tmp_path/"source.sqlite"
    con=sqlite3.connect(db)
    con.executescript('''
        CREATE TABLE employee(id INTEGER, salary TEXT, secret_unrelated TEXT);
        CREATE TABLE cars(id INTEGER, train_id INTEGER, len TEXT);
        INSERT INTO employee VALUES(1,'USD 2,500','NEVER EXPOSE THIS');
        INSERT INTO employee VALUES(2,'USD 5,400','NEVER EXPOSE THIS');
        INSERT INTO cars VALUES(1,1,'long');
        INSERT INTO cars VALUES(2,1,'short');
    ''')
    con.commit()
    con.close()
    return db


def test_authorizer_only_reads_physically_referenced_money_text(tmp_path):
    db=local_db(tmp_path)
    evidence=m._source_column_evidence(db,"SELECT AVG(salary) FROM employee","What is average salary?")
    columns={(x["table"],x["column"]) for x in evidence["columns"]}
    assert ("employee","salary") in columns
    assert not any(col=="secret_unrelated" for _,col in columns)
    salary=next(x for x in evidence["columns"] if x["column"]=="salary")
    assert salary["declared_type"]=="TEXT"
    assert all(v["sqlite_type"]=="text" for v in salary["distinct_raw_examples"])
    assert any("USD 2,500" in v["literal"] for v in salary["distinct_raw_examples"])
    assert "gold_or_publisher_sql_used" not in evidence


def test_authorizer_catches_categorical_vs_numeric_comparison(tmp_path):
    db=local_db(tmp_path)
    evidence=m._source_column_evidence(
        db,"SELECT COUNT(*) FROM cars WHERE len > 2",
        "How many cars have more than two long cars?")
    fields={x["column"]:x for x in evidence["columns"]}
    assert "len" in fields
    assert fields["len"]["declared_type"]=="TEXT"
    values={v["literal"] for v in fields["len"]["distinct_raw_examples"]}
    assert values=={"long","short"}


def test_raw_column_prompt_has_no_reference_sql_or_output_oracle(tmp_path):
    db=local_db(tmp_path)
    fields=m._source_column_evidence(db,"SELECT AVG(salary) FROM employee","What is average salary?")
    p=m._prompt("What is average salary?","Question: What is average salary?\nSchema context: salary TEXT",
                "SELECT AVG(salary) FROM employee",fields)
    body=json.loads(p)
    assert set(body)=={"question","original_question_and_schema","baseline_sql",
                       "physical_sqlite_referenced_column_evidence"}
    assert "gold_sql" not in p and "publisher_sql" not in p
    assert "secret_unrelated" not in p


def test_physical_column_evidence_fails_closed_on_bad_source(tmp_path):
    db=local_db(tmp_path)
    with pytest.raises(sqlite3.Error):
        m._source_column_evidence(db,"SELECT * FROM missing_table","How many?")


def test_preregistered_gate_and_source_exclusions():
    assert m.GATE["min_net_set_ex"]==2 and m.GATE["max_set_regressions"]==0
    assert m.N==8 and m.SHARDS==4 and len(m.SOURCE_SHA)==7
    assert m.SOURCE_COUNTS["prior_preview8"]==8
    assert len(m.RULE_SHA)==64
    assert m.PROTOCOL.endswith("COLUMN-EVIDENCE-V1")
    assert m.MODEL=="qwen3.5:9b-q4_K_M"
