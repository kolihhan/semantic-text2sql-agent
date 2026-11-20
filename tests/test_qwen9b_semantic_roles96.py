"""Source and architecture guardrails for unchanged Qwen3.5-9B SQL generator AB."""
from __future__ import annotations

import inspect
import sqlite3
from pathlib import Path

import pytest

from experiments.qwen9b_semantic_roles96 import (
    DIGEST, MODEL, GATE, N, N_DB, PER_DB, SEMANTIC_SYSTEM,
    SECOND_96_SHA, source_freeze, execute
)
from semantic_sql.direct import _SYSTEM


def test_exact_same_base_and_generator_budget():
    assert MODEL == "qwen3.5:9b-q4_K_M"
    assert DIGEST == "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
    assert N_DB == 12 and PER_DB == 8 and N == 96
    assert "second96" in inspect.signature(source_freeze).parameters
    assert len(SECOND_96_SHA) == 64
    assert GATE["require_full_reference_n"] == 96
    assert SEMANTIC_SYSTEM.startswith(_SYSTEM)


def test_semantic_prompt_focuses_on_observed_61_failure_families_without_gold():
    for term in ("COUNT(DISTINCT", "denominator", "parent/child",
                 "output columns", "display name", "SQLite"):
        assert term in SEMANTIC_SYSTEM
    for forbidden in ("case 325", "case 169", "GAIA", "publisher answer"):
        assert forbidden not in SEMANTIC_SYSTEM


def test_original_physical_readonly_scorer_compares_actual_result_sets(tmp_path):
    db=tmp_path/"smoke.sqlite"
    conn=sqlite3.connect(db)
    try:
        conn.executescript(
            "CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT);"
            "CREATE TABLE messages(id INTEGER, user_id INTEGER);"
            "INSERT INTO users VALUES (1,'A'),(2,'B');"
            "INSERT INTO messages VALUES (1,1),(2,1),(3,2);"
        )
        conn.commit()
    finally:
        conn.close()
    want=execute(db,"SELECT COUNT(DISTINCT u.id) FROM users u JOIN messages m ON u.id=m.user_id")
    right=execute(db,"SELECT 2")
    wrong=execute(db,"SELECT COUNT(*) FROM users u JOIN messages m ON u.id=m.user_id")
    assert want["ok"] and right["ok"] and wrong["ok"]
    assert want["set_sha256"]==right["set_sha256"]
    assert wrong["set_sha256"]!=right["set_sha256"]
    assert not execute(db,"DROP TABLE users")["ok"]
