from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.xiyan_dgsl_frozen100 import aggregate_case_rows


def merge_payloads(paths: list[Path], *, min_correct: int) -> dict[str, object]:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    rows: list[dict[str, object]] = []
    models: set[str] = set()
    for payload in payloads:
        if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
            raise ValueError("every input must be a DGSL shard payload with a cases list")
        rows.extend(payload["cases"])
        model = payload.get("model")
        if model:
            models.add(str(model))

    case_ids = [str(row["case_id"]) for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate case_id across DGSL shards")
    if len(models) > 1:
        raise ValueError(f"mixed models across shards: {sorted(models)}")

    rows.sort(key=lambda row: int(str(row["case_id"])))
    return {
        "experiment": "XiYan lexical schema context vs DGSL v1 grounded schema context",
        "sample_size": len(rows),
        "model": next(iter(models), None),
        "gold_visible_to_generation_or_grounding": False,
        "gold_used_for_offline_metrics_only": True,
        "scorer": "project frozen set-equality execution match; not official BIRD EX",
        "aggregate": aggregate_case_rows(rows, min_correct),
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-correct", type=int, default=61)
    args = parser.parse_args()

    payload = merge_payloads(
        [Path(path) for path in args.inputs],
        min_correct=args.min_correct,
    )
    Path(args.output).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    aggregate = payload["aggregate"]
    print(
        f"DGSL {aggregate['dgsl_correct']}/{aggregate['sample_size']} "
        f"vs baseline {aggregate['baseline_correct']}/{aggregate['sample_size']} "
        f"(net {aggregate['net_correct_delta']:+d}); "
        f"gate_met={aggregate['gate_met']}"
    )


if __name__ == "__main__":
    main()
