"""The Qwen-only lexical ablation is a controlled *generator* test, not a selector."""
from pathlib import Path

from evaluation.qwen_schema_grounding_pair import (
    MODEL_DIGEST, aggregate, contexts, legacy_tokens,
)
from evaluation.schema_context import _tokens
from semantic_sql.catalog import DatabaseCatalog, TableInfo


def test_legacy_comparator_reproduces_pre_fix_bug():
    assert legacy_tokens("PayFrequency") == {"payfrequency"}
    assert {"payfrequency", "pay", "frequency"} <= _tokens("PayFrequency")


def test_contexts_match_original_and_change_only_grounding(tmp_path: Path):
    catalog = DatabaseCatalog(
        database=tmp_path/"unused.sqlite",
        tables={
            "Payroll": TableInfo("Payroll", ("PayFrequency", "EmployeeID")),
            "Geography": TableInfo("Geography", ("Latitude", "Longitude")),
        },
        foreign_keys=(),
    )
    old, new = contexts(catalog, tmp_path, "sample", "What is the pay frequency?")
    assert old[1]["fallback_full_schema"] is True
    assert old[1]["selected_tables"] == ["Geography", "Payroll"]
    assert new[1]["fallback_full_schema"] is False
    assert new[1]["selected_tables"] == ["Payroll"]
    assert old[0] != new[0]


def _shard(i: int):
    cases = []
    for j in range(4):
        idx = 4*i+j
        changed = idx < 8
        left = idx not in (0, 1)
        right = idx not in (1, 2)
        cases.append({
            "case_id": str(idx),
            "group": "changed" if changed else "control",
            "reference_scorable": idx != 3,
            "arms": {
                "legacy": {"ex": left, "bag": left, "sql": "SELECT 1",
                           "both_empty": False},
                "split_identifiers": {"ex": right, "bag": right,
                                      "sql": "SELECT 2" if changed else "SELECT 1",
                                      "both_empty": False},
            },
        })
    return {
        "shard": i, "protocol": "qwen-schema-tokenizer-paired16-v1",
        "cohort_sha256": "one-cohort", "model_digest": MODEL_DIGEST,
        "cases": cases,
    }


def test_aggregate_oracle_not_a_real_selector():
    report = aggregate([_shard(i) for i in range(4)])
    assert report["overall"]["scorable"] == 15
    assert report["overall"]["rescues"] == 1
    assert report["overall"]["regressions"] == 1
    assert report["overall"]["oracle_ex"] == 14
    assert report["groups"]["changed"]["sql_changed"] == 8
    assert report["groups"]["control"]["sql_changed"] == 0
    assert report["oracle_not_a_runtime_selector"] is True


def test_aggregate_rejects_incomplete_cohort():
    import pytest
    with pytest.raises(ValueError, match="four unique"):
        aggregate([_shard(i) for i in range(3)])
