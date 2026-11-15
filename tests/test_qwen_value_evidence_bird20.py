"""Gold-blind Qwen value grounding proposal: no SQL model training."""
from __future__ import annotations

import sqlite3
from pathlib import Path
import pytest

from semantic_sql.catalog import DatabaseCatalog
from evaluation.qwen_value_evidence_bird20 import enhanced_context, value_hits, make_manifest, aggregate, PROTOCOL


def test_original_database_values_are_grounded_not_hallucinated(tmp_path):
    p = tmp_path / "example.sqlite"
    with sqlite3.connect(p) as db:
        db.execute("CREATE TABLE locations(name TEXT, city TEXT)")
        db.execute("INSERT INTO locations VALUES (?, ?)", ("Finches", "Casablanca"))
    catalog = DatabaseCatalog.from_sqlite(p)
    hits = value_hits(catalog, "Which locations are in Casablanca?", "")
    assert any(x["value"] == "Casablanca" and x["column"] == "city" for x in hits)
    prompt = enhanced_context("Physical SQLite schema:", hits)
    assert 'locations.city = "Casablanca"' in prompt
    assert "data only" in prompt
    assert value_hits(catalog, "Which locations are in an unknown city?", "") == []
    with pytest.raises(ValueError):
        enhanced_context("schema", [])


def test_reproducible_question_only_new_db_cohort_ignores_gold(tmp_path):
    source = []
    for dbid in range(10):
        name = f"new_{dbid}"
        directory = tmp_path / name
        directory.mkdir()
        path = directory / f"{name}.sqlite"
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE places(label TEXT)")
            conn.execute("INSERT INTO places VALUES (?)", (f"Town_{dbid}",))
        for j in range(2):
            source.append({"question_id": f"{name}:{j}", "db_id": name,
                           "question": f"List places in Town_{dbid}", "evidence": "",
                           "SQL": "SELECT GOLD_PLACEHOLDER"})
    old = {"cases": [{"case_id": f"old:{j}", "database_id": "old"} for j in range(100)]}
    previous = {"selected": [{"case_id": f"past:{j}", "db_id": "previous"} for j in range(20)]}
    a = make_manifest(source, old, previous, tmp_path)
    b = make_manifest(source, old, previous, tmp_path)
    assert a == b and a["count"] == 20 and a["db_count"] == 10
    assert all("SQL" not in x and x["value_hits"] for x in a["cases"])
    assert {x["db_id"] for x in a["cases"]}.isdisjoint({"old", "previous"})


def test_fail_closed_paired_exact_scoring_and_gate():
    shards = []
    for shard in range(5):
        cases = []
        for j in range(4):
            i = shard*4+j
            old = i < 8
            new = old or i in (8,9,10)
            if i == 0:
                new = False
            cases.append({
                "case_id": str(i), "db_id": f"db{i//2}", "reference_scorable": True,
                "baseline": {"ex":old,"bag":old,"latency_s":1},
                "evidence": {"ex":new,"bag":new,"latency_s":2},
            })
        shards.append({"shard":shard,"protocol":PROTOCOL,"model_digest":"test",
                       "cohort_sha256":"fixed","cases":cases})
    report = aggregate(shards, expected_digest="test")
    assert (report["baseline_ex"], report["evidence_ex"]) == (8,10)
    assert (report["rescues"], report["regressions"]) == (3,1)
    assert report["promotion_allowed"] is False
    assert report["positive_screen_only"] is False
    with pytest.raises(ValueError, match="five shards"):
        aggregate(shards[:4], expected_digest="test")
