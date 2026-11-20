"""CPU-only regression guard: an entirely reference-unscorable DB does not block reporting."""
from __future__ import annotations

import json
from pathlib import Path

from experiments.qwen9b_semantic_roles96 import (
    DIGEST, GATE, N, N_DB, PER_DB, PROTOCOL, summarize,
)


def test_missing_publisher_reference_database_preserves_88_paired_scores(tmp_path):
    # Synthetic paired rows, not benchmark results; no model or gold source accessed.
    dbs=[f"db_{i}" for i in range(N_DB)]
    manifest=tmp_path/"source.json"
    manifest.write_text(json.dumps({"dbs":dbs, "source_sha256":"frozen-test-source"}))
    parts=[]
    for sh, db in enumerate(dbs):
        cases=[]
        for j in range(PER_DB):
            scorable = sh != 11
            cases.append({
                "source_index":sh*PER_DB+j,
                "db_id":db,
                "reference_scorable":scorable,
                "reference_failure":None if scorable else "OperationalError",
                "baseline_selected_set_EX":False if scorable else None,
                "baseline_second_set_EX":False if scorable else None,
                "baseline_oracle_at2_set_EX":False if scorable else None,
                "treatment_second_set_EX":False if scorable else None,
                "treatment_oracle_at2_set_EX":False if scorable else None,
                "treatment_selected_set_EX":False if scorable else None,
                "baseline_selected_bag_EX":False if scorable else None,
                "treatment_selected_bag_EX":False if scorable else None,
                "probe_executable":False,
                "treatment_second_executable":True,
                "baseline_candidate2_executable":True,
                "treatment_final_executable":True,
                "baseline_calls":2,
                "treatment_calls":2,
                "baseline_wall_sec":1.,
                "treatment_wall_sec":1.2,
            })
        p=tmp_path/f"scored-{sh}.json"
        p.write_text(json.dumps({
            "status":"ORIGINAL_QWEN9B_PAIRED_PHYSICAL_STRICT_EX_AFTER_GOLD_BLIND_SQL",
            "protocol":PROTOCOL,
            "model_digest":DIGEST,
            "source_sha256":"frozen-test-source",
            "db_id":db, "shard":sh,"n":PER_DB,
            "DEV500_unused":True, "HOLDOUT300_unused":True,
            "cases":cases,
        }))
        parts.append(p)
    dst=tmp_path/"final.json"
    report=summarize(parts,manifest,dst)
    assert report["n"]==N==96
    assert report["scorable"]==88 and report["censored"]==8
    assert report["publisher_reference_SQL_unscorable"]==8
    assert report["unscorable_database_ids"]==["db_11"]
    assert report["paired_db_cluster_CI95"] is None
    assert report["bootstrap_CI_status"]=="UNESTIMABLE_SOME_PHYSICAL_DATABASES_UNSCORABLE"
    assert report["accepted"] is False
    assert report["baseline_model_calls"]==report["treatment_model_calls"]==192
    assert json.loads(dst.read_text())==report
