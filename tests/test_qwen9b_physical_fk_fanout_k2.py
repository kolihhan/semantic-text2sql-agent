"""Tests for same-model distinct-candidate generation and frozen physical FK facts."""
from pathlib import Path
import json
import sqlite3

import pytest

from semantic_sql.catalog import DatabaseCatalog
from experiments.qwen9b_physical_fk_fanout_k2 import (
    physical_fk_multiplicity, checked_rows, aggregate,
    MODEL_DIGEST, PROTOCOL, GATE, N, SHARDS
)


def db(tmp_path:Path)->Path:
    p=tmp_path/"source.sqlite"
    with sqlite3.connect(p) as con:
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("CREATE TABLE companies (id INTEGER PRIMARY KEY, name TEXT)")
        con.execute("CREATE TABLE employees (id INTEGER PRIMARY KEY, company_id INT,"
                    "FOREIGN KEY(company_id) REFERENCES companies(id))")
        con.execute("INSERT INTO companies VALUES (1,'A'),(2,'B')")
        con.executemany("INSERT INTO employees VALUES (?,?)",[(1,1),(2,1),(3,1),(4,2)])
    return p


def test_measured_physical_child_parent_fanout(tmp_path):
    p=db(tmp_path)
    cat=DatabaseCatalog.from_sqlite(p)
    schema="Physical SQLite schema:\ncompanies(id, name)\nemployees(id, company_id)\nFK employees.company_id -> companies.id"
    txt,n=physical_fk_multiplicity(p,cat,schema)
    assert n==1
    assert "linked child records=4" in txt
    assert "distinct parent-key values=2" in txt
    assert "multiple child records per parent-key=yes" in txt
    assert "FROM" not in txt
    assert not physical_fk_multiplicity(p,cat,"Physical SQLite schema:\ncompanies(id, name)")[0]


def test_frozen_source_must_match_original_prompt_sha(tmp_path):
    rows=[]
    for ix in range(N):
        base="Question: How many employees?\nSchema context:\nPhysical SQLite schema:\ncompanies(id)\nemployees(company_id)"
        treatment=base+"\nPhysical FK multiplicity observations: fake"
        rows.append({
            "source_index":ix,"db_id":f"db{ix//4}","question":"How many employees?",
            "original_user":base,"intervention_user":treatment,
            "original_prompt_sha256":"forged",
            "treatment_prompt_sha256":"forged",
            "physical_fk_edges_shown":1
        })
    p=tmp_path/"source.jsonl"
    blob="".join(json.dumps(x)+"\n" for x in rows).encode()
    p.write_bytes(blob)
    import hashlib
    m={"status":"SOURCE_QUESTION_FROZEN_BEFORE_MODEL_OR_EX",
       "protocol":PROTOCOL,"model_digest":MODEL_DIGEST,
       "source_original_train_sha256":__import__(
         "scripts.prepare_birdtrain_qwen9b_sft_source",fromlist=["TRAIN_SHA"]).TRAIN_SHA,
       "source_sha256":hashlib.sha256(blob).hexdigest(),
       "predeclared_gate":GATE,"n":N,"shards":SHARDS}
    meta=tmp_path/"freeze.json"
    meta.write_text(json.dumps(m))
    with pytest.raises(ValueError,match="prompt drift"):
        checked_rows(p,meta)


def test_prospective_aggregate_requires_four_original_model_workers(tmp_path):
    p=tmp_path/"cohort.jsonl";p.write_text("")
    m=tmp_path/"freeze.json";m.write_text("{}")
    with pytest.raises(ValueError):
        aggregate([],p,m)
