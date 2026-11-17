"""CPU-only contracts; these synthetic data do not assert BIRD accuracy."""
import json
from hashlib import sha256
import pytest
from experiments import qwen9b_count_join_grain16 as m


def test_source_lock_requires_all_eight_old_physical_artifacts():
    with pytest.raises(ValueError,match="eight"):
        m.prior_exclusions({},[])


def test_aggregator_never_accepts_partial_physical_shards(tmp_path):
    with pytest.raises(ValueError):
        m.aggregate([])
    with pytest.raises((FileNotFoundError,ValueError)):
        m.aggregate([tmp_path/"missing"]*4)


def _part(shard):
    rows=[]
    for i in range(4):
        case=shard*4+i
        old=case<5
        improved=old or case==8
        rows.append({"source_index":case,"db_id":f"db_{shard}",
                     "baseline":{"set_ex":old,"bag_ex":old,"executable":True},
                     "grain_guard":{"set_ex":improved,"bag_ex":improved,"executable":True},
                     "grain_changed":case==8,"grain_reason":"question_targets_distinct_parent_entities" if case==8 else "unsupported_sql_form",
                     "new_model_calls":0,"question":f"How many entities {case}?"})
    return {"status":"PHYSICAL_4_CASES_COMPLETE","protocol":m.PROTOCOL,
            "shard":shard,"db_id":f"db_{shard}","n":4,
            "source_sha256":sha256(b"frozen").hexdigest(),
            "model_digest":m.MODEL_DIGEST,"max_repairs":2,"new_training_steps":0,
            "cases":rows}


def test_physical_scores_aggregate_only_if_all_16_complete(tmp_path):
    paths=[]
    for i in range(4):
        p=tmp_path/f"s{i}.json"
        p.write_text(json.dumps(_part(i)))
        paths.append(p)
    out=m.aggregate(paths)
    assert out["n"]==16 and out["totals"]["baseline"]["strict_set_ex"]==5
    assert out["totals"]["grain_guard"]["strict_set_ex"]==6
    assert out["changed_grain_cases"]==1
    assert out["positive_screen_only"] is False
    assert out["promotion_allowed"] is False
    assert out["new_model_calls"]==0
    # Deliberately swap one physical source to test SHA-fail-closed.
    broken=json.loads(paths[2].read_text())
    broken["source_sha256"]="0"*64
    paths[2].write_text(json.dumps(broken))
    with pytest.raises(ValueError,match="SHA mismatch"):
        m.aggregate(paths)


def test_fixed_model_and_no_training_for_sql():
    assert m.MODEL=="qwen3.5:9b-q4_K_M"
    assert len(m.MODEL_DIGEST)==64
    assert m.N==16 and m.N_DB==4 and m.PER_DB==4
    assert len(m.PRIOR_SHA)==8
    assert m.GATE["min_actual_grain_corrections"]==2
    assert m.GATE["max_regressions"]==0
