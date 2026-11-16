from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.qwen9b_fk32_prospective import aggregate, load_cohort,PROTOCOL,MODEL_DIGEST


def test_aggregate_requires_eight_source_frozen_shards(tmp_path):
    path=tmp_path/"shard.json"
    path.write_text(json.dumps({"status":"complete","shard":0,"n":4,"cases":[]}))
    with pytest.raises(ValueError,match="8 shards"):
        aggregate([path])


def test_aggregate_rejects_duplicate_ids_or_case_count(tmp_path):
    paths=[]
    for i in range(8):
        item={"status":"complete","protocol":PROTOCOL,"cohort_sha256":"ee1b5035302d36a1241952bd2e5acd3dfc2518d1fba2581f557e04e179960535",
              "model_digest":MODEL_DIGEST,"same_model":True,
              "n":4,"shard":i,"cases":[]}
        p=tmp_path/f"shard-{i}.json"
        p.write_text(json.dumps(item))
        paths.append(p)
    with pytest.raises(ValueError,match="Run configuration/outcome"):
        aggregate(paths)
