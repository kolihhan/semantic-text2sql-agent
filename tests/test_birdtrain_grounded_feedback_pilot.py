"""Synthetic/CPU-only guards for source-blind semantic-feedback pilot."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from evaluation import birdtrain_grounded_feedback_pilot as pilot


def test_questions_only_projection_does_not_touch_publisher_sql():
    row={"messages":[{"role":"system","content":"x"},
        {"role":"user","content":"Question: How many people?\nSchema context:\npeople(id)\nExternal evidence:\nOnly active"},
        {"role":"assistant","content":"NEVER_CONSUME_GOLD_SQL"}]}
    assert pilot.project_input(row)==("How many people?","Only active")


def test_freeze_source_blind_four_dbs_three_each(tmp_path,monkeypatch):
    rows=[]
    for db in ("one","two","three","four"):
        for i in range(16):
            ix=len(rows)
            rows.append({"source_index":ix,"source_db_id":db,"messages":[
                {"role":"system","content":"sql"},
                {"role":"user","content":f"Question: Q {db} {i}\nSchema context:\n{db}(id)"},
                {"role":"assistant","content":"SECRET_PUBLISHER_REFERENCE_SQL_SHOULD_NOT_LEAK"},
            ]})
    raw="".join(json.dumps(r,sort_keys=True)+"\n" for r in rows)
    source=tmp_path/"source.jsonl";source.write_text(raw)
    monkeypatch.setattr(pilot,"SOURCE_SHA",pilot.sha(raw))
    out=tmp_path/"frozen.json";pilot.freeze(source,out)
    data=json.loads(out.read_text())
    assert data["n"]==12 and data["n_physical_dbs"]==4
    assert data["reference_sql_excluded"] is True
    assert len({r["source_index"] for r in data["cases"]})==12
    assert "SECRET_PUBLISHER" not in out.read_text()
    with pytest.raises(FileExistsError):pilot.freeze(source,out)


def test_no_gold_scoring_before_blind_seal(tmp_path,monkeypatch):
    frozen=tmp_path/"frozen.json";blind=tmp_path/"blind.json"
    frozen.write_text(json.dumps({"protocol":pilot.PROTO,"phase":"SOURCE_FROZEN_NO_GOLD","cases":[]}))
    blind.write_text(json.dumps({"protocol":pilot.PROTO,"phase":"INCOMPLETE_BLIND","cases":[]}))
    def should_not_read(_):
        raise AssertionError("reference opened before sealed blind model decisions")
    monkeypatch.setattr(pilot,"source_rows",should_not_read)
    with pytest.raises(ValueError,match="unsealed"):
        pilot.score(frozen,blind,tmp_path/"never-read",tmp_path,tmp_path/"report.json")


def test_untrusted_sql_review_must_fallback_if_invalid(tmp_path):
    from semantic_sql.verifier import verify_sql_preflight
    db=tmp_path/"x.sqlite"
    with sqlite3.connect(db) as con:con.execute("CREATE TABLE t(id INTEGER)")
    original="SELECT COUNT(*) FROM t"
    assert verify_sql_preflight(db,original).ok
    assert pilot.validate_revision(original,"DELETE FROM t",db)==(original,"invalid_fallback")
    assert pilot.validate_revision(original,"SELECT COUNT(*) FROM t",db)==(original,"unchanged")
    assert pilot.validate_revision(original,"SELECT SUM(id) FROM t",db)==("SELECT SUM(id) FROM t","changed")


def test_prompt_hashes_distinct_and_neither_uses_database_examples():
    assert pilot.GENERIC_SYSTEM!=pilot.EVIDENCE_SYSTEM
    for forbidden in ("patient","molecule","simpson","works_cycles","1256","1525"):
        assert forbidden not in (pilot.GENERIC_SYSTEM+pilot.EVIDENCE_SYSTEM).casefold()
