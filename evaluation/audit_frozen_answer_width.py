"""Read-only projection upper bound on previously generated BIRD SQL."""

import argparse
from collections import Counter
from itertools import permutations
import json
from pathlib import Path

from evaluation.run_bird import _resolve_database
from semantic_sql.execution import execute_readonly


def audit(data: dict, root: Path) -> dict:
    if len(data["cases"]) != 100 or len({row["case_id"] for row in data["cases"]}) != 100:
        raise ValueError("the original 100 guarded SQL cases are required")
    summary = Counter()
    records = []
    for row in data["cases"]:
        case_id = row["case_id"]
        if row["guarded_status"] != "ok":
            summary["verification_refused"] += 1
            continue
        database = _resolve_database(root, row["db_id"])
        target = execute_readonly(database, row["gold_sql"], max_rows=50_000)
        pred = execute_readonly(database, row["original_guarded_sql"], max_rows=50_000)
        target_set = set(target.rows)
        same = not target.truncated and not pred.truncated and target_set == set(pred.rows)
        if same != row["guarded_ex"]:
            raise ValueError(f"original strict score changed for {case_id}")
        if same:
            summary["same_strict_score"] += 1
            if not pred.rows and not target.rows:
                summary["both_empty"] += 1
                summary["both_empty_different_width"] += len(pred.columns) != len(target.columns)
            continue
        if pred.truncated or target.truncated:
            summary["capped"] += 1
            continue
        summary["noncapped_executable_wrong"] += 1
        target_width, pred_width = len(target.columns), len(pred.columns)
        summary["wrong_result_width"] += pred_width != target_width
        if not pred.rows or not target.rows or pred_width < target_width:
            summary["unable_to_repair_by_projection"] += 1
            continue
        if pred_width > 10 or target_width > 4:
            summary["width_outside_projection_probe"] += 1
            continue
        for indices in permutations(range(pred_width), target_width):
            projected = [tuple(item[i] for i in indices) for item in pred.rows]
            if set(projected) != target_set:
                continue
            bag = Counter(projected) == Counter(target.rows)
            summary["projection_oracle_rescues"] += 1
            summary["projection_oracle_bag_rescues"] += bag
            records.append({
                "case": case_id, "db": row["db_id"], "prediction_width": pred_width,
                "reference_width": target_width, "oracle_columns": indices, "bag": bag,
            })
            break
        else:
            summary["unable_to_repair_by_projection"] += 1
    return {
        "metrics": dict(summary), "cases": records,
        "model_digest": data["model_digest"], "frozen_source": data["source_sha256"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", required=True, type=Path)
    parser.add_argument("--databases", required=True, type=Path)
    args = parser.parse_args()
    result = audit(json.loads(args.frozen.read_text(encoding="utf-8")), args.databases)
    print(json.dumps(result, indent=2))
