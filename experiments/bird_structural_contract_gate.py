"""Training-free, gold-blind NL/SQL semantic contract gate for frozen candidates.

Exploratory audit ONLY: use a predeclared narrow rule on existing frozen
BIRD100 A0/A1 candidates. This is not a learned verifier or DPC recreation.
The gate never reads gold, candidate EX labels, database values or question IDs.
Its risk is false positive contract violations from logically equivalent SQL.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlglot
from sqlglot import exp

SCHEMA = "bird-structural-contract-veto-v1"
_RULES = ("explicit_count", "explicit_average", "explicit_count_distinct", "explicit_top_n")
_PATTERNS = {
    "explicit_count": re.compile(
        r"\b(?:how many|number of|count of|count the|how often)\b", re.I),
    "explicit_average": re.compile(r"\b(?:average|mean)\b", re.I),
    "explicit_count_distinct": re.compile(
        r"\b(?:how many|number of)\b.{0,100}\b(?:distinct|different|unique)\b"
        r"|\b(?:distinct|different|unique)\b.{0,100}\b(?:how many|number of)\b", re.I),
    "explicit_top_n": re.compile(r"\btop\s+[2-9]\d*\b", re.I),
}


def requirements(question: str) -> tuple[str, ...]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("original question required")
    keys = [k for k in _RULES if _PATTERNS[k].search(question)]
    # Distinct-count implies count; evaluate both rather than double counting.
    return tuple(keys)


def structural_support(sql: str, rules: tuple[str, ...]) -> dict[str, bool]:
    if not sql or not sql.strip():
        return {k: False for k in rules}
    try:
        parsed = sqlglot.parse_one(sql, read="sqlite")
        if not isinstance(parsed, exp.Expression):
            raise ValueError("no parsed SQL")
        if not isinstance(parsed, exp.Select) and parsed.find(exp.Select) is None:
            raise ValueError("not SELECT-like SQL")
    except (sqlglot.errors.ParseError, ValueError, TypeError):
        return {k: False for k in rules}
    has_count = parsed.find(exp.Count) is not None
    has_avg = parsed.find(exp.Avg) is not None
    has_distinct = (parsed.find(exp.Distinct) is not None
                    or parsed.find(exp.Group) is not None)
    # Supports 'top 5' but not the overly ambiguous 'most common' case.
    has_order_limit = (
        parsed.find(exp.Order) is not None
        and parsed.find(exp.Limit) is not None)
    checks = {
        "explicit_count": has_count,
        "explicit_average": has_avg,
        "explicit_count_distinct": has_count and has_distinct,
        "explicit_top_n": has_order_limit,
    }
    return {k: bool(checks[k]) for k in rules}


def choose(question: str, baseline_sql: str, challenger_sql: str) -> dict:
    """Pure decision: never read any gold SQL, answer, or precomputed EX."""
    rules = requirements(question)
    a = structural_support(baseline_sql, rules)
    b = structural_support(challenger_sql, rules)
    if not rules or not baseline_sql or not challenger_sql or baseline_sql.strip() == challenger_sql.strip():
        use = False
    else:
        baseline_missing = any(not a[k] for k in rules)
        alternate_satisfies_everything = all(b[k] for k in rules)
        use = bool(baseline_missing and alternate_satisfies_everything)
    return {
        "chosen": "a1" if use else "a0",
        "reason": "a0_missing_explicit_contract_and_a1_supports_all" if use
                  else "conservative_baseline",
        "requirements": list(rules),
        "a0_support": a, "a1_support": b,
    }


def evaluate(paired: dict, questions: dict[str, str]) -> dict:
    if (paired.get("schema") != "bird-fk-bridge-aggregate-v1"
        or paired.get("count") != 100
        or len(paired.get("cases", [])) != 100
        or paired.get("baseline_correct") != 64
        or paired.get("question_grounded_correct") != 59
        or paired.get("wrong_to_correct") != 4
        or paired.get("correct_to_wrong") != 9
        or paired.get("promote") is not False):
        raise ValueError("expected known independent frozen 100-candidate dataset")
    rows = paired["cases"]
    if len({str(r["case_id"]) for r in rows}) != 100:
        raise ValueError("duplicated frozen case identities")
    if any(str(r["case_id"]) not in questions for r in rows):
        raise ValueError("missing original input question")
    # Freeze all decisions before reading a single oracle label.
    decisions = {}
    for r in rows:
        cid = str(r["case_id"])
        sql = r.get("sql")
        if not isinstance(sql, dict) or set(sql) != {"a0", "a1"}:
            raise ValueError("wrong candidate payload")
        decisions[cid] = choose(questions[cid], sql["a0"], sql["a1"])
    # Reference EX labels are accessed only here, after decisions are final.
    selected = Counter()
    original = Counter()
    alternate = Counter()
    rescued = Counter()
    regressions = Counter()
    groups = defaultdict(lambda: {"n": 0, "changed": 0, "a0": 0, "chosen": 0,
                                  "rescued": 0, "regressed": 0})
    for r in rows:
        cid = str(r["case_id"])
        dec = decisions[cid]
        group = "explicit_requirement" if dec["requirements"] else "no_requirement"
        if "a0" not in r["ex"] or "a1" not in r["ex"]:
            raise ValueError("missing independently scored candidates")
        ex0 = bool(r["ex"]["a0"])
        ex1 = bool(r["ex"]["a1"])
        chosen = ex1 if dec["chosen"] == "a1" else ex0
        original["correct"] += int(ex0)
        alternate["correct"] += int(ex1)
        selected["correct"] += int(chosen)
        selected["switched"] += int(dec["chosen"] == "a1")
        rescued["count"] += int(not ex0 and chosen)
        regressions["count"] += int(ex0 and not chosen)
        stats = groups[group]
        stats["n"] += 1
        stats["changed"] += int(dec["chosen"] == "a1")
        stats["a0"] += int(ex0)
        stats["chosen"] += int(chosen)
        stats["rescued"] += int(not ex0 and chosen)
        stats["regressed"] += int(ex0 and not chosen)
    if original["correct"] != 64 or alternate["correct"] != 59:
        raise ValueError("frozen baseline or candidate labels diverged")
    if selected["correct"] - original["correct"] != rescued["count"] - regressions["count"]:
        raise AssertionError("paired scoring inconsistency")
    return {
        "schema": SCHEMA,
        "count": 100,
        "source_cohort_sha256": paired["cohort_sha256"],
        "selector": "conservative explicit NL aggregate/top-k contract with SQLGlot",
        "no_learning_or_gold_in_selection": True,
        "candidate_oracle": 68,
        "baseline_correct": original["correct"],
        "challenger_correct": alternate["correct"],
        "selected_correct": selected["correct"],
        "switched": selected["switched"],
        "rescued": rescued["count"],
        "regressed": regressions["count"],
        "net_gain": selected["correct"] - original["correct"],
        "grouped": dict(groups),
        "promote": False,
        "caveat": "Exploratory test on previously inspected public BIRD DEV100; "
        "NL/AST syntax may have semantic equivalents. No independent external "
        "accuracy claim, no new candidate generation, no model training.",
        "decisions": [
            {"case_id": cid, "decision": dec}
            for cid, dec in sorted(decisions.items(), key=lambda item: int(item[0]))
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    if a.output.exists():
        raise FileExistsError("never overwrite analysis artifact")
    from evaluation.bird_loader import load_bird_json
    questions, _hidden_labels = load_bird_json(a.source)
    question_map = {str(c.case_id): c.question for c in questions}
    paired = json.loads(a.paired.read_text(encoding="utf-8"))
    result = evaluate(paired, question_map)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print("BIRD_EXPLICIT_CONTRACT_GATE", json.dumps({
        k: v for k, v in result.items() if k != "decisions"
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
