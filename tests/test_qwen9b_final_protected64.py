"""FINAL preregistered protected physical-DB semantic EX experiment guards."""
import json
from pathlib import Path

from experiments import qwen9b_final_protected64 as final
from experiments import qwen9b_grounded_plan_train96 as historical


def test_exact_preexisting_qwen_plan_and_generator_mechanism_is_frozen():
    assert final.MODEL == historical.MODEL
    assert final.DIGEST == historical.DIGEST
    assert final.PLAN_SYSTEM == historical.PLAN_SYSTEM
    assert final.PLAN_REQUEST == historical.PLAN_REQUEST
    assert final.SQL_PLAN_PREAMBLE == historical.SQL_PLAN_PREAMBLE
    assert final.DBS == 2 and final.PER_DB == 32 and final.N == 64
    assert final.MAX_DB_BYTES == 128 * 1024 * 1024


def test_final_cohort_uses_exact_old_question_only_seal():
    manifest = json.loads(Path(
        "manifests/bird-train-last-two-db-protected-holdout-20261118.json"
    ).read_text())
    assert manifest["score"] is None
    assert manifest["verified_exposed_prior_physical_databases"] == 49
    assert manifest["verified_prior_source_question_ids"] == 425
    records = manifest["holdout_database_records"]
    assert len(records) == 2
    assert {r["db_id"] for r in records} == {"simpson_episodes", "works_cycles"}
    ids = [i for r in records for i in r["protected_original_source_indices"]]
    assert len(ids) == len(set(ids)) == 64
    assert all(r["physical_SQLite_size_bytes"] <= final.MAX_DB_BYTES for r in records)
    assert final.GATE["min_net_correct"] == 2
    assert final.GATE["max_regressions"] == 2
