from __future__ import annotations
from pathlib import Path
import sqlite3
import pytest
from scripts.prepare_birdtrain_qwen9b_sft_source import (
    normalized,validate_gold_sql,make_example,plan_cohorts,N_TRAIN_DBS,
    N_VALID_DBS,QUESTIONS_PER_DB,
)
from semantic_sql.catalog import DatabaseCatalog

def test_reference_only_compiler_without_mutating_database(tmp_path):
    db=tmp_path/"test.sqlite"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE source(id INTEGER PRIMARY KEY, value TEXT)")
        con.execute("INSERT INTO source VALUES(1, 'ok')")
    assert validate_gold_sql(db,"SELECT value FROM source") is None
    assert validate_gold_sql(db,"DELETE FROM source") == "NON_READONLY_REFERENCE"
    assert validate_gold_sql(db,"SELECT nonexistent FROM source") != None
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT COUNT(*) FROM source").fetchone()[0] == 1

def test_example_matches_original_direct_sql_contract(tmp_path):
    db=tmp_path/"bird"/"bird.sqlite"
    db.parent.mkdir(parents=True)
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE items(id INTEGER, price FLOAT)")
    catalog=DatabaseCatalog.from_sqlite(db)
    sample={"source_index":4,"db_id":"bird","question":"What is the highest price?",
            "evidence":"","SQL":"SELECT MAX(price) FROM items"}
    x=make_example(sample,catalog,tmp_path)
    assert x["messages"][1]["content"].startswith("Question: What is the highest price?\nSchema context:\n")
    assert x["messages"][2]["content"] == "SELECT MAX(price) FROM items"

def test_selection_is_question_and_database_disjoint_without_gold():
    # Synthetic sufficient 20 eligible DBs × 16 questions (no reference SQL).
    train=[{"db_id":f"train{d}","question":f"Question {d} {q}","evidence":""}
           for d in range(N_TRAIN_DBS+N_VALID_DBS)
           for q in range(QUESTIONS_PER_DB)]
    dev=[{"db_id":"devdb","question":"Prior heldout question"}]
    manifest={f"train{d}":{"size":100} for d in range(N_TRAIN_DBS+N_VALID_DBS)}
    result=plan_cohorts(train,dev,manifest)
    assert len(result["train"])==N_TRAIN_DBS*QUESTIONS_PER_DB
    assert len(result["validation"])==N_VALID_DBS*QUESTIONS_PER_DB
    assert set(x["db_id"] for x in result["train"]).isdisjoint(
        x["db_id"] for x in result["validation"]
    )
    assert normalized("Hello  WORLD")=="hello world"
