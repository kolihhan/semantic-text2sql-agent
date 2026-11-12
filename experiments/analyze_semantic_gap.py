"""Post-hoc, gold-label-based error accounting for the frozen XiYan semantic audit.

Consumes ONLY an already-scored 4-shard aggregate from xiyan_semantic_shards.py.
Never participates in candidate generation or selection. This is diagnostic
evidence, NOT a fresh holdout or an instruction to tune to these same 100 IDs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def analyze_semantic_gap(aggregate: dict) -> dict:
    if aggregate.get("schema") != "semantic-shard-independent-aggregate-v1":
        raise ValueError("expected a completed independent semantic aggregate")
    cases = aggregate.get("cases")
    expected_count = aggregate.get("sample_size")
    if not isinstance(cases, list) or not cases or type(expected_count) is not int or len(cases) != expected_count:
        raise ValueError("incomplete or mismatched paired population")
    ids = [str(row.get("case_id", "")) for row in cases]
    if not all(ids) or len(set(ids)) != len(ids):
        raise ValueError("missing or duplicated paired identities")

    buckets = Counter()
    by_reason = defaultdict(Counter)
    for row in cases:
        required_booleans = (
            "baseline_ex", "revision_ex", "selected_ex",
            "baseline_valid", "revision_valid", "changed",
        )
        if any(type(row.get(k)) is not bool for k in required_booleans):
            raise ValueError("missing/invalid outcome or validity flag")
        arm = row.get("chosen_arm")
        if arm not in {"mschema", "semantic_revision"}:
            raise ValueError("invalid selected arm")
        base, revision, selected = (
            row["baseline_ex"], row["revision_ex"], row["selected_ex"]
        )
        if selected != (revision if arm == "semantic_revision" else base):
            raise ValueError("selected outcome disagrees with frozen candidate outcome")
        reason = str(row.get("reason") or "")
        if not reason:
            raise ValueError("missing selection reason")
        r = by_reason[reason]
        r["cases"] += 1
        r["baseline_correct"] += int(base)
        r["selected_correct"] += int(selected)
        r["wins"] += int(not base and selected)
        r["losses"] += int(base and not selected)

        buckets["baseline_correct"] += int(base)
        buckets["revision_correct"] += int(revision)
        buckets["selected_correct"] += int(selected)
        buckets["baseline_executable_but_wrong"] += int(row["baseline_valid"] and not base)
        buckets["both_candidates_wrong"] += int(not base and not revision)
        buckets["both_candidates_correct"] += int(base and revision)
        buckets["revision_only_correct"] += int(not base and revision)
        buckets["baseline_only_correct"] += int(base and not revision)
        buckets["revision_rescued"] += int(not base and selected)
        buckets["revision_opportunity_missed"] += int(not base and revision and not selected)
        buckets["selection_regression"] += int(base and not selected)
        buckets["revision_regression_prevented"] += int(base and not revision and selected)
        buckets["revision_selected"] += int(arm == "semantic_revision")
        buckets["sql_text_changed"] += int(row["changed"])
        buckets["executable_only_repair"] += int(
            not row["baseline_valid"] and row["revision_valid"]
            and arm == "semantic_revision" and not selected
        )
        base_missing, revised_missing = (
            row.get("baseline_violations"), row.get("revised_violations")
        )
        if not isinstance(base_missing, list) or not isinstance(revised_missing, list):
            raise ValueError("missing question-constraint fields")
        strictly_fewer = len(revised_missing) < len(base_missing)
        buckets["surface_contract_reduced_violations"] += int(strictly_fewer)
        buckets["surface_contract_reduced_but_still_wrong"] += int(strictly_fewer and not revision)

    for name, expected in (
        ("baseline_correct", aggregate.get("baseline_correct")),
        ("revision_correct", aggregate.get("revision_correct")),
        ("selected_correct", aggregate.get("selected_correct")),
    ):
        if type(expected) is not int or buckets[name] != expected:
            raise ValueError(f"{name} mismatches original frozen aggregate")
    net = buckets["selected_correct"] - buckets["baseline_correct"]
    if net != aggregate.get("net_gain"):
        raise ValueError("frozen aggregate net-gain mismatch")
    if buckets["revision_rescued"] != aggregate.get("wrong_to_correct"):
        raise ValueError("frozen aggregate wins mismatch")
    if buckets["selection_regression"] != aggregate.get("correct_to_wrong"):
        raise ValueError("frozen aggregate losses mismatch")

    return {
        "schema": "semantic-error-diagnostic/v1",
        "cohort_sha256": aggregate.get("cohort_sha256"),
        "count": expected_count,
        "net_gain": net,
        "counts": dict(sorted(buckets.items())),
        "selection_reasons": {
            name: dict(sorted(counts.items()))
            for name, counts in sorted(by_reason.items())
        },
        "interpretation": {
            "both_candidates_wrong": "generation/grounding unresolved; not a proven root cause",
            "revision_opportunity_missed": "candidate existed but gold-blind selector kept the wrong baseline",
            "selection_regression": "selector accepted a wrong revision over a correct baseline",
            "executable_only_repair": "operational repair without semantic answer rescue",
            "surface_contract_reduced_but_still_wrong": "keyword-level SQL checks are not semantic proof",
        },
        "restriction": (
            "Post-hoc analysis of already-exposed labels; never tune repeatedly "
            "on this same cohort and call subsequent gains independent."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refuse to overwrite prior diagnostic evidence")
    result = analyze_semantic_gap(
        json.loads(args.aggregate.read_text(encoding="utf-8"))
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "interpretation"}, sort_keys=True))


if __name__ == "__main__":
    main()
