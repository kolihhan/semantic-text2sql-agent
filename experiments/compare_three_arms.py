"""CPU-only apples-to-apples Direct / Guarded Workflow / ETSA Graph score.

Reuse the 100 sealed, shared-Qwen9B initial SQL outputs. Zero new model calls.
B is the historical Guarded policy (internally implemented with LangGraph);
C is experimental evidence-triggered policy. This tests policies, not graph
library vs Python-language implementation overhead.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from evaluation.run_bird import _resolve_database
from semantic_sql.execution import SQLExecutionError, execute_readonly

EXPECTED_DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
EXPECTED_SOURCE = "ffd8018378ddb1a8794753e0a31cfc81862ff7318a5184c22f3dc4ce03a03feb"
ARMS = ("direct", "workflow", "langgraph")


def execute(database: Path, sql: str | None):
    if not sql:
        return None
    try:
        result = execute_readonly(database, sql, max_rows=50000)
        return None if result.truncated else result
    except SQLExecutionError:
        return None


def agrees(predicted, gold, *, bag: bool):
    if predicted is None or gold is None:
        return False
    if bag:
        return Counter(predicted.rows) == Counter(gold.rows)
    return set(predicted.rows) == set(gold.rows)


def report(frozen_path: Path, shard_dir: Path, source_path: Path, db_root: Path, previous_path: Path | None):
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    source_bytes = source_path.read_bytes()
    assert sha256(source_bytes).hexdigest() == EXPECTED_SOURCE, "wrong BIRD source"
    assert frozen["model_digest"] == EXPECTED_DIGEST
    assert frozen["status"] == "frozen_gold_free"
    assert len(frozen["cases"]) == 100

    expected = {str(c["case_id"]): c for c in frozen["cases"]}
    assert len(expected) == 100, "duplicate frozen case IDs"
    source = {str(row["question_id"]): row for row in json.loads(source_bytes)}
    by_case = {}
    for shard in range(4):
        path = shard_dir / f"etsa-shard-{shard}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["status"] == "complete"
        assert data["shard"] == shard
        assert data["digest"] == EXPECTED_DIGEST
        assert data["cohort_sha256"] == frozen["cohort_sha256"]
        assert len(data["cases"]) == 25
        for item in data["cases"]:
            key = str(item["case_id"])
            assert key in expected and key not in by_case, f"duplicate/missing {key}"
            assert item["status"] == "complete", f"incomplete case {key}"
            assert item["database_id"] == expected[key]["database_id"]
            assert item["question"] == expected[key]["question"]
            for field in ("initial_sql", "g0_sql", "g2_sql"):
                assert isinstance(item.get(field), str), (key, field)
            by_case[key] = item
    assert set(by_case) == set(expected), "not the exact DEV100 cohort"

    cases = []
    for case in frozen["cases"]:
        key = str(case["case_id"])
        item = by_case[key]
        reference = source[key]
        assert reference["db_id"] == item["database_id"]
        assert reference["question"] == item["question"]
        db = _resolve_database(db_root, item["database_id"])
        gold = execute(db, reference["SQL"])
        # Guarded refuses unverified SQL; do not turn refused SQL into a scored execution.
        guarded_ok = item["g0_status"] == "ok"
        sqls = {
            "direct": item["initial_sql"],
            "workflow": item["g0_sql"] if guarded_ok else None,
            "langgraph": item["g2_sql"] if guarded_ok else None,
        }
        output = {
            "case_id": key, "database_id": item["database_id"],
            "scorable": gold is not None,
            "routed": bool(item["routed"]),
            "graph_selected": bool(item["g2_selected"]),
        }
        for arm, sql in sqls.items():
            result = execute(db, sql)
            output[arm] = {
                "executed": result is not None,
                "set": agrees(result, gold, bag=False),
                "bag": agrees(result, gold, bag=True),
            }
        cases.append(output)
    counts = {arm: {
        "strict_set_ex": sum(c[arm]["set"] for c in cases),
        "strict_bag_ex": sum(c[arm]["bag"] for c in cases),
        "execution_success": sum(c[arm]["executed"] for c in cases),
    } for arm in ARMS}

    def delta(old: str, new: str):
        return {
            "set_rescues": sum(not c[old]["set"] and c[new]["set"] for c in cases),
            "set_regressions": sum(c[old]["set"] and not c[new]["set"] for c in cases),
            "bag_rescues": sum(not c[old]["bag"] and c[new]["bag"] for c in cases),
            "bag_regressions": sum(c[old]["bag"] and not c[new]["bag"] for c in cases),
        }

    metrics = {
        "arms": counts,
        "workflow_vs_direct": delta("direct", "workflow"),
        "langgraph_vs_workflow": delta("workflow", "langgraph"),
        "routed": sum(c["routed"] for c in cases),
        "langgraph_selected": sum(c["graph_selected"] for c in cases),
        "scorable_gold": sum(c["scorable"] for c in cases),
    }
    if previous_path:
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        assert previous["cohort"] == "historical exact DEV100"
        assert previous["model_digest"] == EXPECTED_DIGEST
        assert counts["workflow"]["strict_set_ex"] == previous["metrics"]["g0"]["strict_set_ex"]
        assert counts["workflow"]["strict_bag_ex"] == previous["metrics"]["g0"]["strict_bag_ex"]
        assert counts["langgraph"]["strict_set_ex"] == previous["metrics"]["g2"]["strict_set_ex"]
        assert counts["langgraph"]["strict_bag_ex"] == previous["metrics"]["g2"]["strict_bag_ex"]
        tokens = previous["metrics"]["calls_and_tokens"]
        assert tokens["g0_shared"]["calls"] >= 100
        metrics["model_calls"] = {
            "direct": 100,
            "workflow": tokens["g0_shared"]["calls"],
            "langgraph": tokens["g0_shared"]["calls"] + tokens["g2_extra"]["calls"],
        }
        metrics["tokens"] = {
            "workflow_prompt": tokens["g0_shared"]["prompt_eval_count"],
            "workflow_completion": tokens["g0_shared"]["eval_count"],
            "direct": None,
            "langgraph_extra": tokens["g2_extra"]["eval_count"],
        }
    metrics["actual_usd_cost"] = None
    output = {
        "status": "complete" if metrics["scorable_gold"] == 100 else "invalid",
        "experiment": "retrospective CPU-only 3-arm strict score from sealed same DEV100 generation",
        "cohort_sha256": frozen["cohort_sha256"],
        "model_digest": EXPECTED_DIGEST,
        "model_calls_new": 0,
        "caveat": "Workflow is the existing Guarded policy and internally uses LangGraph. This isolates decision policies, not LangGraph engine overhead. Old ETSA graph is REJECTED, not a demonstrated semantic gain.",
        "metrics": metrics, "cases": cases,
    }
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frozen", type=Path, required=True)
    p.add_argument("--shards", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--db-root", type=Path, required=True)
    p.add_argument("--previous", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    output = report(args.frozen, args.shards, args.source, args.db_root, args.previous)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["metrics"], indent=2))
    if output["status"] != "complete":
        raise SystemExit("not all 100 reference cases are scorable")


if __name__ == "__main__":
    main()
