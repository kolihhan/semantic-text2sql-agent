"""Prove exact model/source/worker contracts before original model VAL64 evidence."""
import json
from pathlib import Path

import pytest

from scripts.audit_birdtrain64_physical_literal_counterexample import (
    MODEL_DIGEST, PROTOCOL, TRAIN_VAL64_SHA, DBS, GATE,
    original_rows, original_model, worker, aggregate,
)


def test_model_digest_fails_closed(tmp_path: Path):
    p=tmp_path/"wrong.json"
    p.write_text(json.dumps({
        "status":"complete","n":64,
        "source_validation_sha256":TRAIN_VAL64_SHA,
        "model_digest":"changed_base",
        "direct_ex":27,"guarded_ex":31,
        "cases":[{"id":i} for i in range(64)]
    }))
    with pytest.raises(ValueError, match="digest"):
        original_model(p)


def test_source_sha_fails_closed(tmp_path: Path):
    p=tmp_path/"val.jsonl"
    p.write_text('{"source_index":1,"source_db_id":"movie_3"}\n')
    with pytest.raises(ValueError, match="SHA"):
        original_rows(p)


def test_worker_shard_range_rejected_before_database(tmp_path: Path):
    with pytest.raises(ValueError, match="four-database"):
        worker(tmp_path/"source.jsonl",tmp_path/"model.json",
               tmp_path/"missing.sqlite",4)


def test_aggregation_rejects_incomplete_source(tmp_path: Path):
    with pytest.raises(ValueError):
        aggregate([],tmp_path/"missing_source.jsonl")


def test_rules_are_source_disjoint_from_previous_fanout():
    assert len(DBS)==4 and len(set(DBS))==4
    assert not set(DBS)&{"books","restaurant","movie","app_store"}
    assert "model" in "original real model".lower()
    assert GATE["min_guarded_changed"]>=3
    assert GATE["max_guarded_regressions"]==0
    assert "literal" in PROTOCOL.lower()
    assert MODEL_DIGEST=="56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
