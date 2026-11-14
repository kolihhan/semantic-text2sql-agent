"""Gold-blind, post-hoc evaluation of a minimal SQL execution fallback.

Uses already-generated paired Qwen/XiYan SQL predictions. Primary XiYan is
selected if it passes preflight and produces an untruncated executed result;
otherwise choose the Qwen Guarded result. No labels inform the selection.
No inference or default runtime mutation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def select_without_gold(arms: dict) -> tuple[str, str]:
    xi = arms["xiyan"]
    if xi["preflight"] and xi["execution_success"]:
        return "xiyan", "primary_ok"
    return "qwen", "primary_failed"


def analyze(shards: list[dict], *, partial: bool = False) -> dict:
    indices = [s["shard"] for s in shards]
    if not shards or len(indices) != len(set(indices)) or not set(indices).issubset(set(range(10))):
        raise ValueError("missing, duplicate, or invalid shard index")
    if not partial and sorted(indices) != list(range(10)):
        raise ValueError("full BIRD DEV100 requires ten shards")
    models = shards[0]["models"]
    cohort_hash = shards[0]["cohort_sha256"]
    if any(s["models"] != models or s["cohort_sha256"] != cohort_hash
           or s["count"] != 10 or len(s["cases"]) != 10 for s in shards):
        raise ValueError("shard model/cohort/size drift")
    rows = [r for s in shards for r in s["cases"]]
    if len(rows) != 10 * len(shards) or len({r["case_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate or incomplete case identities")
    counts = Counter()
    by_db = Counter()
    selections = []
    for row in rows:
        arms = row["arms"]
        chosen, reason = select_without_gold(arms)
        counts["qwen_ex"] += bool(arms["qwen"]["ex"])
        counts["xiyan_ex"] += bool(arms["xiyan"]["ex"])
        counts["selected_ex"] += bool(arms[chosen]["ex"])
        counts["qwen_bag"] += bool(arms["qwen"]["bag_equal"])
        counts["xiyan_bag"] += bool(arms["xiyan"]["bag_equal"])
        counts["selected_bag"] += bool(arms[chosen]["bag_equal"])
        if chosen == "qwen":
            counts["fallbacks"] += 1
            counts["fallback_ex_rescues"] += bool(
                arms["qwen"]["ex"] and not arms["xiyan"]["ex"])
            counts["fallback_ex_regressions"] += bool(
                arms["xiyan"]["ex"] and not arms["qwen"]["ex"])
            by_db[row["database_id"]] += 1
        selections.append({"case_id": row["case_id"], "selected": chosen, "reason": reason})
    return {
        "status": "partial" if len(shards) < 10 else "complete",
        "shards": sorted(indices), "cases": len(rows),
        "model_digests": models, "cohort_sha256": cohort_hash,
        "metrics": dict(counts), "fallback_databases": dict(by_db),
        "selection": selections,
        "scope": "Offline gold-blind candidate; public retrospective DEV, not independent hidden-set validation",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = analyze(
        [json.loads(p.read_text()) for p in args.inputs],
        partial=args.allow_partial,
    )
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "selection"}))


if __name__ == "__main__":
    main()
