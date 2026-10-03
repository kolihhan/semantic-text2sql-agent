from __future__ import annotations

import json
from pathlib import Path

from experiments.merge_dgsl_results import merge_payloads
from experiments.xiyan_dgsl_frozen100 import aggregate_case_rows, load_baseline_sql


def _row(case_id: str, *, baseline: bool, dgsl: bool) -> dict[str, object]:
    base_metrics = {
        "table_recall": 0.8,
        "table_precision": 0.6,
        "column_recall": 0.7,
        "column_precision": 0.4,
        "fk_bridge_recall": 0.5,
        "value_grounding_recall": 0.0,
    }
    dgsl_metrics = {
        "table_recall": 1.0,
        "table_precision": 0.7,
        "column_recall": 0.9,
        "column_precision": 0.5,
        "fk_bridge_recall": 1.0,
        "value_grounding_recall": 1.0,
    }
    return {
        "case_id": case_id,
        "baseline_frozen_execution_match": baseline,
        "dgsl_frozen_execution_match": dgsl,
        "baseline_grounding": base_metrics,
        "dgsl_grounding": dgsl_metrics,
        "decomposition_latency_s": 0.2,
        "dgsl_generation_latency_s": 0.5,
    }


def test_load_baseline_sql_reuses_xiyan_candidates(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {"case_id": "93", "xiyan_sql": "SELECT 1"},
                    {"case_id": "1017", "xiyan_sql": "SELECT 2"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert load_baseline_sql(path) == {
        "93": "SELECT 1",
        "1017": "SELECT 2",
    }


def test_aggregate_case_rows_reports_transitions_and_gate() -> None:
    summary = aggregate_case_rows(
        [
            _row("1", baseline=True, dgsl=True),
            _row("2", baseline=False, dgsl=True),
            _row("3", baseline=True, dgsl=False),
        ],
        min_correct=2,
    )

    assert summary["baseline_correct"] == 2
    assert summary["dgsl_correct"] == 2
    assert summary["wrong_to_correct"] == 1
    assert summary["correct_to_wrong"] == 1
    assert summary["gate_met"] is True
    assert summary["grounding_metrics_macro"]["table_recall"]["dgsl"] == 1.0


def test_merge_payloads_rejects_duplicates_and_merges_unique_shards(
    tmp_path: Path,
) -> None:
    shard0 = tmp_path / "shard0.json"
    shard1 = tmp_path / "shard1.json"
    shard0.write_text(
        json.dumps({"model": "xiyan", "cases": [_row("1", baseline=True, dgsl=True)]}),
        encoding="utf-8",
    )
    shard1.write_text(
        json.dumps({"model": "xiyan", "cases": [_row("2", baseline=False, dgsl=True)]}),
        encoding="utf-8",
    )

    merged = merge_payloads([shard0, shard1], min_correct=2)

    assert merged["sample_size"] == 2
    assert merged["aggregate"]["baseline_correct"] == 1
    assert merged["aggregate"]["dgsl_correct"] == 2
    assert merged["aggregate"]["gate_met"] is True
