"""Immutable, post-hoc paired A0/A1/A2/A3 diagnostic on a completed run.

No model calls, no gold-aware selection, no threshold tuning. The run must have
been scored after all candidates were generated. Treat findings as exploratory
because public BIRD DEV database identities may overlap previous experiments.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from math import comb
import json
from pathlib import Path


def exact_mcnemar_p(wins: int, losses: int) -> float:
    """Two-sided exact matched-pair sign test for discordant cases."""
    if min(wins, losses) < 0:
        raise ValueError("invalid paired counts")
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def paired_transition(cases: list[dict], base: str, treatment: str) -> dict:
    wins = sum(not bool(c["ex"][base]) and bool(c["ex"][treatment]) for c in cases)
    losses = sum(bool(c["ex"][base]) and not bool(c["ex"][treatment]) for c in cases)
    return {
        "base": base, "treatment": treatment,
        "wrong_to_correct": wins, "correct_to_wrong": losses,
        "net_correct": wins - losses,
        "discordant_cases": wins + losses,
        "exact_mcnemar_two_sided_p_exploratory": exact_mcnemar_p(wins, losses),
    }


def analyze(result: dict) -> dict:
    if result.get("schema") != "semantic-abc-aggregate-v1" or result.get("sample_size") != 100:
        raise ValueError("invalid completed experiment artifact")
    cases = result.get("cases")
    if not isinstance(cases, list) or len(cases) != 100:
        raise ValueError("expected 100 completed results")
    ids = [str(c["case_id"]) for c in cases]
    if len(set(ids)) != 100:
        raise ValueError("duplicated case IDs")
    names = ("a0", "a1", "a2", "a3")
    for c in cases:
        ex = c.get("ex")
        if (not isinstance(ex, dict) or set(ex) != set(names)
            or not all(type(ex[n]) is bool for n in names)):
            raise ValueError("incomplete boolean paired execution flags")
        if ex["a0"] != ex["a1"]:
            raise ValueError("passive A1 mutated baseline result")
        if c.get("a3_reason") not in {
            "unchanged", "semantic_output_repair_accepted", "semantic_repair_rejected"
        }:
            raise ValueError("unexpected semantic action reason")
    computed = {name: sum(c["ex"][name] for c in cases) for name in names}
    if computed != result.get("scores"):
        raise ValueError("aggregate score corruption")
    transitions = {
        "a0_to_a1": paired_transition(cases, "a0", "a1"),
        "a0_to_a2": paired_transition(cases, "a0", "a2"),
        "a2_to_a3": paired_transition(cases, "a2", "a3"),
        "a0_to_a3": paired_transition(cases, "a0", "a3"),
    }
    per_db: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        per_db[str(c["database_id"])].append(c)
    db_reports = []
    for database_id, group in sorted(per_db.items()):
        pair = paired_transition(group, "a0", "a3")
        db_reports.append({
            "database_id": database_id, "questions": len(group),
            "a0_correct": sum(c["ex"]["a0"] for c in group),
            "a3_correct": sum(c["ex"]["a3"] for c in group),
            "wins": pair["wrong_to_correct"], "losses": pair["correct_to_wrong"],
            "net": pair["net_correct"],
        })
    actions = Counter(c["a2_reason"] for c in cases)
    actions.update(c["a3_reason"] for c in cases)
    accepted_semantic = [c for c in cases if c["a3_reason"] == "semantic_output_repair_accepted"]
    accepted_compiler = [c for c in cases if c["a2_reason"] == "compiler_repair_accepted"]
    report = {
        "schema": "semantic-abc-exploratory-analysis-v1",
        "sample_size": 100,
        "cohort_sha256": result["cohort_sha256"],
        "scores": computed,
        "paired_transitions": transitions,
        "actions": dict(sorted(actions.items())),
        "accepted_semantic_outcomes": {
            "count": len(accepted_semantic),
            "wrong_to_correct": sum(not c["ex"]["a2"] and c["ex"]["a3"] for c in accepted_semantic),
            "correct_to_wrong": sum(c["ex"]["a2"] and not c["ex"]["a3"] for c in accepted_semantic),
            "both_wrong": sum(not c["ex"]["a2"] and not c["ex"]["a3"] for c in accepted_semantic),
            "both_correct": sum(c["ex"]["a2"] and c["ex"]["a3"] for c in accepted_semantic),
        },
        "accepted_compiler_repair_count": len(accepted_compiler),
        "extra_model_calls": sum(int(c["extra_model_calls"]) for c in cases),
        "database_count": len(per_db),
        "per_database": db_reports,
        "important_limitations": [
            "BIRD DEV questions were preselected without prior labels but may share databases with previous evaluated BIRD DEV cohorts.",
            "Set-of-result-tuples execution equality is an imperfect correctness proxy.",
            "Post-hoc exact McNemar p-values do not correct for repeated experiments or multiple comparisons.",
            "Do not use this 100-case result to tune thresholds or prompts and call this same cohort a holdout.",
        ],
    }
    return report


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError("never overwrite posthoc source of truth")
    original=json.loads(args.input.read_text(encoding="utf-8"))
    report=analyze(original)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print("SEMANTIC_ABC_ANALYSIS",json.dumps({
        k:v for k,v in report.items() if k!="per_database" and k!="important_limitations"
    },sort_keys=True,ensure_ascii=False))


if __name__ == "__main__":
    main()
