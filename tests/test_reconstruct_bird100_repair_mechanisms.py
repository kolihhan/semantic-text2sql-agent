"""Fail-closed structural tests; live original-artifact replay runs separately in CI."""
import copy
import pytest
from research.reconstruct_bird100_repair_mechanisms import audit

def test_requires_exactly_four_original_shards():
    with pytest.raises(ValueError, match="four"):
        audit([])
    with pytest.raises(ValueError, match="four"):
        audit([{}])

def test_rejects_wrong_historical_model_and_missing_raw_cases():
    a={"status":"complete","sample_size":25,"cases":[{}]*25,
       "model_digest":"wrong","source_sha256":"bad"}
    with pytest.raises(ValueError, match="shard 0"):
        audit([copy.deepcopy(a) for _ in range(4)])

def test_never_relabels_aggregate_only_summary_as_raw_evidence():
    with pytest.raises(ValueError):
        audit([{"metrics":{"direct":{"official_bird_ex":0.33}}}]*4)
