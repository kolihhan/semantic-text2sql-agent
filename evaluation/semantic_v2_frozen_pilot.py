"""Frozen Qwen9B inference-free semantic-v2 paired pilot.

Reuses 40 old quoted-value and 20 later A/A semantic DEV records. No model
generation, tuning, or benchmark contamination at selection time. All
proposals are finalized before BIRD gold labels are accessed. The results
are exploratory because these public BIRD DEV cohorts were used for earlier
research and may overlap at the database/pretraining level.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from evaluation.bird_loader import load_bird_json
from evaluation.run_bird import _resolve_database
from semantic_sql.execution import SQLExecutionError, execute_readonly
from semantic_sql.semantic_v2 import evidence_gated_literal_revision
from semantic_sql.verifier import verify_sql_preflight


def score(database: Path, sql: str, gold_sql: str) -> bool:
    if not sql:
        return False
    try:
        pred = execute_readonly(database, sql, max_rows=50_000)
        gold = execute_readonly(database, gold_sql, max_rows=50_000)
    except (SQLExecutionError, ValueError):
        return False
    return not (pred.truncated or gold.truncated) and set(pred.rows) == set(gold.rows)


def freeze_inputs(literal40: dict, aa20: dict) -> list[dict]:
    if (literal40.get("schema") != "qwen9b-exact-value-grounding-paired40/v1"
        or literal40.get("sample_size") != 40
        or aa20.get("schema") != "qwen9b-semantic-aa-aggregate/v1"
        or aa20.get("count") != 20
        or aa20.get("model") != "qwen3.5:9b-q4_K_M"):
        raise ValueError("wrong frozen original Qwen9B generation sources")
    out = []
    for r in literal40["cases"]:
        out.append({
            "case_id": str(r["case_id"]),
            "database_id": r["database_id"],
            "cohort": "literal40",
            "baseline_sql": r["baseline_sql"],
            "baseline_status": r.get("baseline_status"),
            "pre_scored": r["ex"]["baseline"],
        })
    for r in aa20["cases"]:
        out.append({
            "case_id": str(r["case_id"]),
            "database_id": r["database_id"],
            "cohort": "aa20_" + r["group"],
            "baseline_sql": r["predictions"]["baseline_a"]["sql"],
            "baseline_status": r["predictions"]["baseline_a"]["status"],
            "pre_scored": r["ex"]["baseline_a"]["set"],
        })
    if (len(out) != 60 or len({r["case_id"] for r in out}) != 60
        or sum(r["cohort"] == "literal40" for r in out) != 40
        or sum(r["cohort"] == "aa20_value_hit" for r in out) != 10
        or sum(r["cohort"] == "aa20_no_quoted_literal" for r in out) != 10):
        raise ValueError("wrong sample counts or overlapping BIRD case IDs")
    return out


def evaluate(*, literal40: dict, aa20: dict, source: Path, database_root: Path) -> dict:
    original = freeze_inputs(literal40, aa20)
    questions, label_rows = load_bird_json(source)
    question_by_id = {str(q.case_id): q for q in questions}
    for r in original:
        q = question_by_id[r["case_id"]]
        if q.database_id != r["database_id"]:
            raise ValueError("database ID mismatch")
    cache: dict[str, Path] = {}
    frozen = []
    # No reference labels or historical EX results enter the selector.
    for r in original:
        q = question_by_id[r["case_id"]]
        if q.database_id not in cache:
            cache[q.database_id] = _resolve_database(database_root, q.database_id)
        db = cache[q.database_id]
        if (r["baseline_status"] != "ok"
            or not verify_sql_preflight(db, r["baseline_sql"]).ok):
            proposal = {"changed": False, "sql": r["baseline_sql"],
                        "reason": "baseline_not_verified"}
        else:
            dec = evidence_gated_literal_revision(
                database=db, question=q.question, sql=r["baseline_sql"]
            )
            proposal = {"changed": dec.changed, "sql": dec.sql, "reason": dec.reason}
        frozen.append({**r, "proposal": proposal})
    # Only after every proposal has been frozen do we inspect reference SQL
    # and historical correctness. This permits aggregate EX only.
    gold = {str(q.case_id): q.sql for q in label_rows}
    results = []
    for r in frozen:
        db = cache[r["database_id"]]
        a = score(db, r["baseline_sql"], gold[r["case_id"]])
        b = score(db, r["proposal"]["sql"], gold[r["case_id"]])
        if a is not bool(r["pre_scored"]):
            raise AssertionError("recomputed baseline disagrees with frozen source score")
        results.append({
            "case_id": r["case_id"],
            "cohort": r["cohort"],
            "changed": r["proposal"]["changed"],
            "reason": r["proposal"]["reason"],
            "baseline_correct": a,
            "semantic_v2_correct": b,
        })
    groups = defaultdict(list)
    for r in results:
        groups[r["cohort"]].append(r)
    def summarize(rows: list[dict]) -> dict:
        base = sum(r["baseline_correct"] for r in rows)
        new = sum(r["semantic_v2_correct"] for r in rows)
        rescues = sum(not r["baseline_correct"] and r["semantic_v2_correct"] for r in rows)
        regressions = sum(r["baseline_correct"] and not r["semantic_v2_correct"] for r in rows)
        assert new - base == rescues - regressions
        return {
            "count": len(rows), "baseline_correct": base, "semantic_v2_correct": new,
            "changed": sum(r["changed"] for r in rows),
            "rescues": rescues, "regressions": regressions,
            "net_gain": new-base,
            "reasons": dict(Counter(r["reason"] for r in rows)),
        }
    report = {
        "schema": "qwen9b-semantic-v2-evidence-pilot60/v1",
        "model": "qwen3.5:9b-q4_K_M",
        "overall": summarize(results),
        "groups": {name: summarize(rows) for name, rows in sorted(groups.items())},
        "promote": False,
        "gold_used_during_proposal": False,
        "full_bird_or_external_holdout": False,
        "caveat": "Reuses public BIRD DEV identities and earlier Qwen9B generated SQL, "
        "not new generation, not out-of-domain. Local set-of-tuples execution "
        "match; independent new-data verification needed before promotion.",
        "cases": sorted(results, key=lambda r: int(r["case_id"])),
    }
    assert report["overall"]["count"] == 60
    assert report["overall"]["baseline_correct"] == 33
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--literal40", type=Path, required=True)
    p.add_argument("--aa20", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--database-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("refuse overwrite original evidence")
    report = evaluate(
        literal40=json.loads(args.literal40.read_text(encoding="utf-8")),
        aa20=json.loads(args.aa20.read_text(encoding="utf-8")),
        source=args.source,
        database_root=args.database_root,
    )
    args.output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print("QWEN9B_SEMANTIC_V2_FROZEN60", json.dumps({
        k: v for k, v in report.items() if k != "cases"
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
