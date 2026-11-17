"""Frozen BIRD DEV500 K-sampling, label-blind selectors and paired CIs.

The selector sees ONLY SQL candidate cache and actual SQLite results. Publisher
gold is loaded after ALL predictions are frozen. We report both pass@1
(random draw mean) and first-draw accuracy, pass@K as oracle *upper bound*,
and a deterministic execution-result plurality selector (no oracle labels).
Both question-paired and DB-cluster bootstrap are shown. 11 BIRD DEV physical
databases are NOT independently new; question-level CI may be anti-conservative.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256, sha1
import json
from pathlib import Path
import random
from statistics import median
from typing import Any

from semantic_sql.execution import SQLExecutionError, execute_readonly

PROTOCOL = "FROZEN-BIRD-DEV500-QWEN35-9B-KCACHE-V1"
MANIFEST = "manifests/bird-dev500-holdout300-20261116-v2.json"
MANIFEST_GIT_BLOB_SHA1 = "692fc84b881746d32f70469d03c7026fcc67291a"
SOURCE_SHA256 = "ffd8018378ddb1a8794753e0a31cfc81862ff7318a5184c22f3dc4ce03a03feb"
MODEL_NAME = "qwen3.5:9b-q4_K_M"
MODEL_DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
MIN_EFFECT = 0.05
MIN_ORACLE_GAP = 0.08
BOOTSTRAP_ROUNDS = 4000
BOOTSTRAP_SEED = 20261117


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def file_git_blob_sha1(path: Path) -> str:
    raw = path.read_bytes()
    return sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()


def read_frozen_manifest(path: Path) -> dict[str, Any]:
    if file_git_blob_sha1(path) != MANIFEST_GIT_BLOB_SHA1:
        raise ValueError("existing frozen V2 DEV500/HOLDOUT300 manifest was modified")
    d = json.loads(path.read_text("utf-8"))
    dev, final = d.get("dev_case_ids"), d.get("holdout_case_ids")
    if (d.get("source_sha256") != SOURCE_SHA256 or
        d.get("protocol") != "bird-dev-stratified-question-disjoint/v1" or
        d.get("dev_size") != 500 or d.get("holdout_size") != 300 or
        not isinstance(dev, list) or not isinstance(final, list) or
        len(dev) != 500 or len(final) != 300 or
        len(set(dev)) != 500 or len(set(final)) != 300 or
        set(dev).intersection(final)):
        raise ValueError("existing frozen BIRD source/manifests invariant failed")
    return d


def source_cases(source: Path, manifest: dict, *, split: str = "dev",
                 allow_final: bool = False) -> list[dict]:
    if split not in ("dev", "holdout"):
        raise ValueError("must select frozen dev or final holdout only")
    if split == "holdout" and not allow_final:
        raise PermissionError("final HOLDOUT300 is reserved; explicit one-time release required")
    if file_sha256(source) != SOURCE_SHA256:
        raise ValueError("original pinned BIRD source bytes do not match")
    raw = json.loads(source.read_text("utf-8"))
    keys = [str(x["question_id"]) for x in raw]
    if len(raw) != 1534 or len(set(keys)) != 1534:
        raise ValueError("original BIRD question IDs/source rows have drifted")
    index = dict(zip(keys, raw))
    chosen = manifest["dev_case_ids" if split == "dev" else "holdout_case_ids"]
    if any(key not in index for key in chosen):
        raise ValueError("frozen case ID missing in original source")
    return [index[key] for key in chosen]


def candidate_seed(case_id: str, sample_idx: int, *, salt: str = "generation-v1") -> int:
    if sample_idx < 0:
        raise ValueError("negative candidate index")
    return int(sha256(f"{salt}:{case_id}:{sample_idx}".encode()).hexdigest()[:8], 16)


def canonical_row(row: tuple) -> str:
    return json.dumps(list(row), ensure_ascii=False, sort_keys=True, default=str,
                      separators=(",", ":"))


def candidate_outcome(database: Path, sql: str, *, max_rows: int = 50000) -> dict[str, Any]:
    """SQLite physical result fingerprint; no source SQL/gold/answer passed in."""
    try:
        got = execute_readonly(database, sql, max_rows=max_rows)
    except (SQLExecutionError, ValueError, OSError) as exc:
        return {"valid": False, "reason": type(exc).__name__, "signature": None,
                "set_hash": None, "bag_hash": None}
    if got.truncated:
        return {"valid": False, "reason": "row_cap", "signature": None,
                "set_hash": None, "bag_hash": None}
    keys = [canonical_row(row) for row in got.rows]
    distinct = sorted(set(keys))
    bags = sorted(Counter(keys).items())
    set_hash = sha256(json.dumps(distinct).encode()).hexdigest()
    bag_hash = sha256(json.dumps(bags).encode()).hexdigest()
    # Equal data and projection arity; no gold, no SQL-reference comparison.
    signature = sha256(json.dumps([len(got.columns), set_hash]).encode()).hexdigest()
    return {"valid": True, "reason": "ok", "signature": signature,
            "set_hash": set_hash, "bag_hash": bag_hash}


def majority_selector(outcomes: list[dict]) -> dict[str, Any]:
    """Plurality among physically executable candidate result sets, no gold."""
    if not outcomes:
        raise ValueError("candidate collection empty")
    valid = [(i, item["signature"]) for i, item in enumerate(outcomes)
             if item["valid"] and item["signature"]]
    if not valid:
        return {"index": 0, "vote_count": 0, "agreement": 0.0,
                "unique_valid_result_sets": 0, "valid": 0}
    votes = Counter(signature for _, signature in valid)
    best = max(votes.values())
    # On ties retain earliest candidate so the prediction remains deterministic.
    selected = next(i for i, signature in valid if votes[signature] == best)
    return {"index": selected, "vote_count": best,
            "agreement": best / len(outcomes),
            "unique_valid_result_sets": len(votes), "valid": len(valid)}


def sample_correctness(outcome: dict, gold: dict, *, bag: bool = False) -> bool:
    return (outcome["valid"] and gold["valid"] and
            outcome["bag_hash" if bag else "set_hash"] ==
            gold["bag_hash" if bag else "set_hash"])


def per_case_offline(cache: dict, db: Path, gold_sql: str, *,
                     max_rows: int = 50000) -> dict:
    """Selector decision happens BEFORE gold SQL is used for evaluation."""
    candidates = cache["samples"]
    if len(candidates) != cache["k"]:
        raise ValueError("noncomplete sampling cache; no scores")
    outcomes = [candidate_outcome(db, row["sql"], max_rows=max_rows) for row in candidates]
    selected = majority_selector(outcomes)
    # No gold variable exists in candidate_outcome or majority_selector scope.
    gold = candidate_outcome(db, gold_sql, max_rows=max_rows)
    set_ok = [sample_correctness(x, gold) for x in outcomes]
    bag_ok = [sample_correctness(x, gold, bag=True) for x in outcomes]
    chosen = selected["index"]
    return {
        "case_id": cache["case_id"], "db_id": cache["db_id"], "k": cache["k"],
        "first_set_ex": bool(set_ok[0]), "first_bag_ex": bool(bag_ok[0]),
        "pass1_empirical_set": sum(set_ok)/len(set_ok),
        "passK_oracle_set": bool(any(set_ok)),
        "passK_oracle_bag": bool(any(bag_ok)),
        "selected_set_ex": bool(set_ok[chosen]),
        "selected_bag_ex": bool(bag_ok[chosen]),
        "selected_index": chosen,
        "vote_agreement": selected["agreement"],
        "unique_valid_result_sets": selected["unique_valid_result_sets"],
        "candidate_execution_success": sum(x["valid"] for x in outcomes),
        "gold_scorable": bool(gold["valid"]),
        "zero_gold_in_selector": True
    }


def paired_bootstrap(records: list[dict], baseline: str, treatment: str, *,
                     cluster: bool = False, rounds: int = BOOTSTRAP_ROUNDS) -> dict:
    if not records:
        raise ValueError("empty paired records")
    rng = random.Random(BOOTSTRAP_SEED)
    diffs = [float(x[treatment]) - float(x[baseline]) for x in records]
    n = len(diffs)
    point = sum(diffs)/n
    by_db: dict[str, list[float]] = defaultdict(list)
    for record, delta in zip(records, diffs):
        by_db[str(record["db_id"])].append(delta)
    dbs = sorted(by_db)
    distributions = []
    for _ in range(rounds):
        if cluster:
            chosen = [rng.choice(dbs) for i in range(len(dbs))]
            values = [x for db in chosen for x in by_db[db]]
        else:
            values = [rng.choice(diffs) for i in range(n)]
        distributions.append(sum(values)/len(values))
    distributions.sort()
    return {"estimate": point,
            "ci95_lower": distributions[max(0, int(0.025*rounds))],
            "ci95_upper": distributions[min(rounds-1, int(0.975*rounds))],
            "unit": "physical_db_cluster" if cluster else "paired_question",
            "n_questions": n, "n_databases": len(dbs),
            "method": "deterministic paired percentile bootstrap; descriptive not official evaluator"}


def aggregate_offline(records: list[dict], *, required_ids: set[str] | None = None) -> dict:
    if not records:
        raise ValueError("no complete cached cases")
    ids = [str(r["case_id"]) for r in records]
    if len(set(ids)) != len(ids) or (required_ids is not None and set(ids) != required_ids):
        raise ValueError("missing/duplicate frozen manifest cases; no partial statistics")
    if len({r["k"] for r in records}) != 1:
        raise ValueError("candidate K differs across cases")
    if not all(r["gold_scorable"] for r in records):
        raise ValueError("unscorable publisher result; do not silently change denominator")
    n = len(records)
    rate = lambda key: sum(float(x[key]) for x in records)/n
    question_ci = paired_bootstrap(records,"first_set_ex","selected_set_ex")
    db_ci = paired_bootstrap(records,"first_set_ex","selected_set_ex",cluster=True)
    gap = rate("passK_oracle_set")-rate("pass1_empirical_set")
    point = question_ci["estimate"]
    return {
        "protocol": PROTOCOL, "n": n, "k": records[0]["k"],
        "first_sample_set_ex": rate("first_set_ex"),
        "pass1_empirical_set": rate("pass1_empirical_set"),
        "passK_oracle_set": rate("passK_oracle_set"),
        "passK_minus_pass1_gap": gap,
        "selector_plurarity_set_ex": rate("selected_set_ex"),
        "selector_gain_vs_first_sample": point,
        "question_paired_95ci": question_ci,
        "db_cluster_paired_95ci": db_ci,
        "passK_below_eight_point_headroom": gap < MIN_ORACLE_GAP,
        "selector_meets_five_point_and_ci_gate": (
            point >= MIN_EFFECT and question_ci["ci95_lower"] > 0 and
            db_ci["ci95_lower"] > 0),
        "not_an_oracle_selector": True,
        "disclaimer": "pass@K is oracle upper bound, not deployed pass rate; CI assumes fixed independent units and shared physical databases reduce effective sample size; source-local BIRD EX ≠ human-adjudicated semantics.",
    }
