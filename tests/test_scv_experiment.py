from __future__ import annotations

import sqlite3
from pathlib import Path

from experiments.xiyan_scv_frozen100 import aggregate_case_rows, score_query


def test_score_query_uses_explicit_frozen_execution_match_label(tmp_path: Path) -> None:
    db = tmp_path / "score.sqlite"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE items (id INTEGER)")
        con.executemany("INSERT INTO items VALUES (?)", [(1,), (2,)])

    correct = score_query(db, "SELECT id FROM items", "SELECT id FROM items")
    wrong = score_query(db, "SELECT id FROM items WHERE id = 1", "SELECT id FROM items")

    assert correct == {
        "execution_success": True,
        "frozen_execution_match": True,
        "truncated": False,
    }
    assert wrong["execution_success"] is True
    assert wrong["frozen_execution_match"] is False
    assert "official_ex" not in wrong


def test_aggregate_case_rows_reports_paired_transitions_and_diagnostics() -> None:
    rows = [
        {
            "case_id": "1",
            "baseline_execution_success": True,
            "baseline_frozen_execution_match": True,
            "scv_execution_success": True,
            "scv_frozen_execution_match": True,
            "sql_changed": False,
            "scv_status": "SCV_PASS",
            "scv_violation_codes": [],
            "scv_latency_s": 1.0,
        },
        {
            "case_id": "2",
            "baseline_execution_success": True,
            "baseline_frozen_execution_match": False,
            "scv_execution_success": True,
            "scv_frozen_execution_match": True,
            "sql_changed": True,
            "scv_status": "SCV_REPAIR_PASS",
            "scv_violation_codes": ["PROJECTION_MISMATCH"],
            "scv_latency_s": 3.0,
        },
        {
            "case_id": "3",
            "baseline_execution_success": False,
            "baseline_frozen_execution_match": False,
            "scv_execution_success": False,
            "scv_frozen_execution_match": False,
            "sql_changed": False,
            "scv_status": "SCV_SKIPPED_CONTRACT",
            "scv_violation_codes": [],
            "scv_latency_s": 2.0,
        },
        {
            "case_id": "4",
            "baseline_execution_success": True,
            "baseline_frozen_execution_match": True,
            "scv_execution_success": True,
            "scv_frozen_execution_match": True,
            "sql_changed": True,
            "scv_status": "SCV_VIOLATION",
            "scv_violation_codes": ["JOIN_EDGE_INVALID", "JOIN_EDGE_INVALID"],
            "scv_latency_s": 4.0,
        },
    ]

    summary = aggregate_case_rows(rows)
    assert summary["sample_size"] == 4
    assert summary["baseline"]["frozen_execution_match"] == 0.5
    assert summary["scv"]["frozen_execution_match"] == 0.75
    assert summary["transitions"] == {
        "wrong_to_correct": 1,
        "correct_to_wrong": 0,
        "net_correct_delta": 1,
    }
    assert summary["scv"]["changed_sql"] == 2
    assert summary["scv_status_counts"]["SCV_PASS"] == 1
    assert summary["scv_status_counts"]["SCV_SKIPPED_CONTRACT"] == 1
    assert summary["violation_counts"] == {
        "JOIN_EDGE_INVALID": 2,
        "PROJECTION_MISMATCH": 1,
    }
    assert summary["median_scv_latency_s"] == 2.5
