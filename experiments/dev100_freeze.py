"""Freeze EXACT historical v5 DEV100 identities without leaking reference SQL to inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SOURCE_SHA = "ffd8018378ddb1a8794753e0a31cfc81862ff7318a5184c22f3dc4ce03a03feb"
HISTORICAL_SHA = "0310b7ca2dfb6247234e2073f93e7a75106ee41ba55231f75d06d2ef68efe2ec"
MODEL_DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"


def freeze(source: Path, reference: Path, target: Path) -> dict:
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA, "wrong publisher dataset"
    assert hashlib.sha256(raw.replace(b"\n", b"\r\n")).hexdigest() == HISTORICAL_SHA
    ref = json.loads(reference.read_text(encoding="utf-8"))
    assert ref["source_sha256"] == HISTORICAL_SHA
    assert ref["status"] == "complete" and len(ref["cases"]) == 100
    src = {str(r["question_id"]): r for r in json.loads(raw)}
    cases = []
    for case in ref["cases"]:
        key = str(case["case_id"])
        row = src[key]
        assert row["db_id"] == case["database_id"]
        assert row["question"] == case["question"]
        cases.append({"case_id":key, "database_id":row["db_id"],
                      "question":row["question"], "evidence":row.get("evidence") or ""})
    assert len(cases) == len({r["case_id"] for r in cases}) == 100
    identity_sha = hashlib.sha256(json.dumps(cases, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    payload={"status":"frozen_gold_free","model_digest":MODEL_DIGEST,
             "source_sha256":HISTORICAL_SHA,"cohort_sha256":identity_sha,"cases":cases}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"FROZEN DEV100=100 DBs={len({r['database_id'] for r in cases})} cohort_sha256={identity_sha}")
    return payload


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--reference",default="runs/bird-paired-dev-v5/paired.json")
    p.add_argument("--output",required=True)
    a=p.parse_args()
    freeze(Path(a.source),Path(a.reference),Path(a.output))
