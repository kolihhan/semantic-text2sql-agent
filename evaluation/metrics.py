from __future__ import annotations


def paired_transition_counts(records: list[dict]) -> dict[str, int]:
    counts = {
        "direct_wrong_to_guarded_correct": 0,
        "direct_correct_to_guarded_wrong": 0,
        "direct_execution_fail_to_guarded_success": 0,
        "direct_execution_success_to_guarded_fail": 0,
    }
    for record in records:
        direct = record["direct"]
        guarded = record["guarded"]
        counts["direct_wrong_to_guarded_correct"] += int(not direct["official_ex"] and guarded["official_ex"])
        counts["direct_correct_to_guarded_wrong"] += int(direct["official_ex"] and not guarded["official_ex"])
        counts["direct_execution_fail_to_guarded_success"] += int(not direct["execution_success"] and guarded["execution_success"])
        counts["direct_execution_success_to_guarded_fail"] += int(direct["execution_success"] and not guarded["execution_success"])
    return counts
