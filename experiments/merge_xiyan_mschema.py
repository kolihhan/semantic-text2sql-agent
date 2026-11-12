"""Exact four-shard XiYan M-Schema comparison. Never use labels for prompting."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from statistics import median


def aggregate(shards: list[dict], *, min_correct: int = 61) -> dict:
    if len(shards) != 4:
        raise ValueError("expected exactly four shards")
    if any(p.get("schema") != "xiyan-mschema-frozen100-v1" for p in shards):
        raise ValueError("shard schema mismatch")
    if sorted(p["shard_index"] for p in shards) != [0, 1, 2, 3]:
        raise ValueError("shard indices must be distinct and complete")
    models = {p["model"] for p in shards}
    if len(models) != 1:
        raise ValueError("model names differ")
    rows = [r for p in shards for r in p["cases"]]
    case_ids = [str(r["case_id"]) for r in rows]
    if len(rows) != 100 or len(set(case_ids)) != 100:
        raise ValueError("missing or duplicated frozen100 cases")
    if any(r["baseline_truncated"] or r["mschema_truncated"] for r in rows):
        raise ValueError("truncated SQL result: do not treat as a valid EX comparison")
    rows.sort(key=lambda r: int(r["case_id"]))
    baseline = sum(bool(r["baseline_ex"]) for r in rows)
    candidate = sum(bool(r["mschema_ex"]) for r in rows)
    if baseline != 58:
        raise ValueError(f"frozen baseline drifted from 58/100: {baseline}")
    wrong_to_correct = sum(not r["baseline_ex"] and r["mschema_ex"] for r in rows)
    correct_to_wrong = sum(r["baseline_ex"] and not r["mschema_ex"] for r in rows)
    assert baseline + wrong_to_correct - correct_to_wrong == candidate
    return {
        "status": "complete",
        "schema": "xiyan-mschema-frozen100-aggregate-v1",
        "experiment": "same XiYan 7B with M-Schema; frozen lexical SQL reference",
        "model": next(iter(models)),
        "sample_size": 100,
        "scorer": "project set-equality execution match; not official BIRD submission",
        "baseline_lexical_ex": baseline,
        "mschema_ex": candidate,
        "wrong_to_correct": wrong_to_correct,
        "correct_to_wrong": correct_to_wrong,
        "net_gain": candidate - baseline,
        "mschema_execution_success": sum(r["mschema_execution_success"] for r in rows),
        "lexical_execution_success": sum(r["baseline_execution_success"] for r in rows),
        "median_ms_generation_s": median(r["generation_latency_s"] for r in rows),
        "min_correct_gate": min_correct,
        "gate_met": candidate >= min_correct,
        "limitations": "Same frozen100 previously used for SCV and DGSL; this is not an untouched independent test.",
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-correct", type=int, default=61)
    args = parser.parse_args()
    result = aggregate([json.loads(p.read_text(encoding="utf-8")) for p in args.inputs], min_correct=args.min_correct)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
