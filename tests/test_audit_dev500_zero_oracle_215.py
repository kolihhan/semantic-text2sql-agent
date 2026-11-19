"""No-GPU unit guards for the original 215-case forensic census."""
from __future__ import annotations

from collections import Counter

import pytest

pytest.importorskip("sqlglot", reason="CPU forensic script uses an optional SQL AST analysis dependency")

from scripts.audit_dev500_zero_oracle_215 import (
    STUDY, SOURCE_SHA, MODEL_DIGEST, FLAGGED_OPS,
    features, simple_intents,
)


def test_sql_ast_features_are_structural_only():
    a=features("SELECT DISTINCT m.name FROM molecule AS m JOIN atom a ON a.mid=m.id WHERE a.element='N' ORDER BY m.name LIMIT 10")
    assert a is not None
    assert a["projection_width"]==1
    assert {"molecule","atom"}<=set(a["tables"])
    assert {"distinct","join","where","order","limit"}<=set(a["ops"])
    b=features("SELECT COUNT(*) FROM molecule")
    assert b["projection_width"]==1
    assert "count" in b["ops"]
    assert a["coarse_shape_sha256"]!=b["coarse_shape_sha256"]


def test_invalid_sql_ast_abstains_not_impute_real_sqlite_failure():
    assert features("I refuse this task.") is None
    assert features("") is None


def test_question_intent_regex_is_only_diagnostic():
    assert "count" in simple_intents("How many molecules contain nitrogen?")
    assert "ratio_percent" in simple_intents("Calculate the percentage of molecules.")
    assert "ranking" in simple_intents("What is the highest scoring player?")
    assert "relationship" in simple_intents("Which atoms are connected to atoms?")
    assert len(MODEL_DIGEST)==len(SOURCE_SHA)==64
    assert "ORIGINAL-QWEN9B-BIRD-DEV500" in STUDY
    assert set(FLAGGED_OPS)>={"join","group","where","order","count","division"}


def test_original_full_500_frontier_requires_215_expected_zero_oracle_by_contract():
    from scripts import audit_dev500_zero_oracle_215 as m
    import inspect
    s=inspect.getsource(m.load_original)
    assert "!= 500" in s
    assert "!= 4000" in s
    assert "!= 233" in s
    assert "!= 285" in s
    s=inspect.getsource(m.audit)
    assert "!= 215" in s
    assert "!= 97" in s
    assert "!= 90" in s
    assert "!= 28" in s
    assert "publisher_unscorable_certified" in s
