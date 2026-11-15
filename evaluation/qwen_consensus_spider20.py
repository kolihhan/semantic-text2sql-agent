"""Same-Qwen SQL candidate diversity/consensus on new public Spider TRAIN databases.

Purpose: determine whether errors are primarily candidate-generation reachability
or gold-blind selection. No training, replacement model, extra schema, gold
feedback, benchmark-specific patches or production promotion.

The 3 stochastic runs use one fixed Qwen3.5-9B checkpoint at temperature 0.35
and seeds 103/211/307. Majority means equal *executed multiset of rows* on
at least two of three verified candidates. Otherwise retain original
temperature-0 Guarded output. A gold-dependent oracle@4 is hindsight only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic

from evaluation.run_bird import _ollama_model_digest
from evaluation.schema_context import select_schema_context
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.execution import SQLExecutionError, execute_readonly
from semantic_sql.inference import run_guarded
from semantic_sql.providers import OllamaProvider

MODEL = "qwen3.5:9b-q4_K_M"
DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
SEED_OLD = "xiyan-a5-100-spider-train-db-disjoint-from-previous-dev-20261113-v1"
SEED_IR = "qwen9b-typed-ir-spider-train-new-db20-20261115-v1"
SEED_NEW = "qwen9b-consensus-new-spider-db-20261115-v1"
PROTOCOL = "qwen9b-candidate-consensus-spider-shift20-v1"
N_SHARDS = 5
SAMPLE_SEEDS = (103, 211, 307)
SAMPLE_TEMPERATURE = 0.35


def _hash(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def select_cases(train: list[dict], dev: list[dict]) -> list[dict]:
    """Select fresh physical DBs/questions without touching reference SQL."""
    dev_ids = {str(x["db_id"]) for x in dev}
    groups: dict[str, list[dict]] = defaultdict(list)
    for idx, row in enumerate(train):
        db = str(row.get("db_id") or "")
        question = str(row.get("question") or "").strip()
        if not db or not question:
            raise ValueError("invalid Spider TRAIN question")
        if db in dev_ids:
            raise ValueError("Spider TRAIN/DEV database overlap")
        groups[db].append({
            "case_id": f"{db}:{idx}", "row_index": idx,
            "database_id": db, "question": question,
        })
    prior_order = sorted(groups, key=lambda db: _hash(f"{SEED_OLD}:database:{db}"))
    if len(prior_order) < 120:
        raise ValueError("fewer than 10 fresh physical databases after known cohorts")
    previous_100 = set(prior_order[:100])
    ir_candidates = sorted(
        (db for db, items in groups.items() if len(items) >= 2 and db not in previous_100),
        key=lambda db: _hash(f"{SEED_IR}:database:{db}"),
    )
    if len(ir_candidates) < 20:
        raise ValueError("too few physically independent databases")
    previous_ir10 = set(ir_candidates[:10])
    fresh = sorted(
        (db for db, items in groups.items()
         if len(items) >= 2 and db not in previous_100 and db not in previous_ir10),
        key=lambda db: _hash(f"{SEED_NEW}:database:{db}"),
    )
    if len(fresh) < 10:
        raise ValueError("fewer than 10 fresh physical databases")
    selected = []
    for db in fresh[:10]:
        candidates = sorted(
            groups[db], key=lambda row: _hash(f"{SEED_NEW}:{row['case_id']}")
        )
        selected.extend(candidates[:2])
    if len(selected) != 20 or len({x["case_id"] for x in selected}) != 20:
        raise AssertionError("invalid question/physical-database selection")
    return selected


def _dbpath(root: Path, db_id: str) -> Path:
    path = root / db_id / f"{db_id}.sqlite"
    if not path.is_file():
        raise FileNotFoundError(f"physical SQLite not found: {db_id}")
    return path


def _bag(run: dict) -> Counter:
    return Counter(tuple(row) for row in run["rows"])


def choose_consensus(samples: list[dict]) -> tuple[int | None, int]:
    """Require >=2 of the 3 independently executed candidates to agree.

    Fail closed on missing samples, failed execution, truncated rows, or
    different duplicates. This uses NO reference labels.
    """
    if len(samples) != 3:
        return None, 0
    bags = [_bag(sample) if sample.get("execution_success") else None for sample in samples]
    for i in range(3):
        if bags[i] is None:
            continue
        votes = sum(bag is not None and bag == bags[i] for bag in bags)
        if votes >= 2:
            return i, votes
    return None, 0


def _provider(*, temperature: float, seed: int) -> OllamaProvider:
    return OllamaProvider(
        model=MODEL, base_url="http://127.0.0.1:11434",
        timeout_s=900, temperature=temperature, seed=seed, num_predict=1024,
    )


def _run(db: Path, question: str, schema: str, provider: OllamaProvider) -> dict:
    start = monotonic()
    result = run_guarded(
        database=db, provider=provider, question=question,
        schema_context=schema, max_repairs=2, max_rows=50000,
    )
    valid = result.status == "ok" and not result.truncated
    return {
        "sql": result.candidate.sql if result.candidate else "",
        "status": result.status,
        "execution_success": valid,
        "rows": result.rows if valid else (),
        "repair_attempts": result.candidate.attempt if result.candidate else 0,
        "latency_s": round(monotonic() - start, 2),
    }


def _score(run: dict, expected: tuple) -> dict:
    got = run["rows"] if run["execution_success"] else ()
    return {
        "strict_ex": bool(run["execution_success"] and set(map(tuple,got)) == set(map(tuple,expected))),
        "bag": bool(run["execution_success"] and Counter(map(tuple,got)) == Counter(map(tuple,expected))),
        "execution_success": run["execution_success"],
        "latency_s": run["latency_s"],
        "sql": run["sql"],
        "repair_attempts": run["repair_attempts"],
    }


def run_shard(train_file: Path, dev_file: Path, root: Path, shard: int) -> dict:
    if shard not in range(N_SHARDS):
        raise ValueError("invalid shard index")
    train = json.loads(train_file.read_text(encoding="utf-8"))
    dev = json.loads(dev_file.read_text(encoding="utf-8"))
    selected = select_cases(train, dev)
    identity = _hash(json.dumps(
        [(x["case_id"], x["database_id"]) for x in selected], separators=(",", ":"),
    ))
    chosen = selected[4 * shard: 4 * (shard + 1)]
    if len(chosen) != 4:
        raise ValueError("incomplete shard selection")
    digest = _ollama_model_digest("http://127.0.0.1:11434", MODEL)
    if digest != DIGEST:
        raise RuntimeError("Qwen3.5-9B checkpoint digest mismatch")
    base_provider = _provider(temperature=0, seed=42)
    sample_providers = [_provider(temperature=SAMPLE_TEMPERATURE, seed=s) for s in SAMPLE_SEEDS]
    raw = []
    for row in chosen:
        path = _dbpath(root, row["database_id"])
        catalog = DatabaseCatalog.from_sqlite(path)
        schema, _ = select_schema_context(catalog, root, row["database_id"], row["question"])
        baseline = _run(path, row["question"], schema, base_provider)
        samples = [_run(path, row["question"], schema, p) for p in sample_providers]
        consensus_idx, votes = choose_consensus(samples)
        choice = samples[consensus_idx] if consensus_idx is not None else baseline
        raw.append({
            **row,
            "db_id": row["database_id"],
            "schema_sha256": _hash(schema),
            "baseline_raw": baseline,
            "samples_raw": samples,
            "choice_raw": choice,
            "consensus_index": consensus_idx,
            "votes": votes,
            "candidate_diversity": len({x["sql"].strip().casefold() for x in samples}),
        })
        print("QWEN_CONSENSUS_GENERATED", shard, row["case_id"],
              "votes", votes, "diversity", raw[-1]["candidate_diversity"], flush=True)

    # Only NOW read the gold SQL, after all candidate predictions are frozen.
    cases = []
    for row in raw:
        path = _dbpath(root, row["db_id"])
        try:
            reference = execute_readonly(
                path, str(train[row["row_index"]]["query"]), max_rows=50000,
            )
            scorable = not reference.truncated
            expected = reference.rows if scorable else ()
        except (SQLExecutionError, ValueError):
            scorable, expected = False, ()
        base = _score(row["baseline_raw"], expected) if scorable else None
        choices = [_score(x, expected) for x in row["samples_raw"]] if scorable else []
        consensus = _score(row["choice_raw"], expected) if scorable else None
        cases.append({
            "case_id": row["case_id"], "db_id": row["db_id"],
            "schema_sha256": row["schema_sha256"],
            "reference_scorable": scorable,
            "consensus_index": row["consensus_index"],
            "votes": row["votes"], "candidate_diversity": row["candidate_diversity"],
            "baseline": base, "samples": choices, "consensus": consensus,
            "oracle_correct": bool(base["strict_ex"] or any(x["strict_ex"] for x in choices)) if scorable else None,
            "total_latency_s": round(
                row["baseline_raw"]["latency_s"] +
                sum(x["latency_s"] for x in row["samples_raw"]), 2),
        })
    return {
        "protocol": PROTOCOL, "shard": shard,
        "cohort_sha256": identity, "model_digest": digest,
        "sample_seeds": SAMPLE_SEEDS, "sample_temperature": SAMPLE_TEMPERATURE,
        "cases": cases,
    }


def aggregate(shards: list[dict], *, expected_digest: str = DIGEST) -> dict:
    if len(shards) != N_SHARDS or sorted(s.get("shard") for s in shards) != list(range(N_SHARDS)):
        raise ValueError("five unique complete shards required")
    if (len({s.get("cohort_sha256") for s in shards}) != 1
        or any(s.get("protocol") != PROTOCOL or s.get("model_digest") != expected_digest
               for s in shards)):
        raise ValueError("cohort/protocol/model drift")
    cases = [c for shard in sorted(shards, key=lambda x: x["shard"]) for c in shard["cases"]]
    if (len(cases) != 20 or len({c["case_id"] for c in cases}) != 20
        or len({c["db_id"] for c in cases}) != 10
        or any(len(shard["cases"]) != 4 for shard in shards)):
        raise ValueError("case count or physical DB count mismatch")
    scorable = [c for c in cases if c["reference_scorable"]]
    if any(c["baseline"] is None or c["consensus"] is None for c in scorable):
        raise ValueError("missing scorable arm")
    rescues = sum(not c["baseline"]["strict_ex"] and c["consensus"]["strict_ex"] for c in scorable)
    regressions = sum(c["baseline"]["strict_ex"] and not c["consensus"]["strict_ex"] for c in scorable)
    base_ex = sum(c["baseline"]["strict_ex"] for c in scorable)
    new_ex = sum(c["consensus"]["strict_ex"] for c in scorable)
    if new_ex - base_ex != rescues - regressions:
        raise AssertionError("paired EX accounting drift")
    gained_db = {
        c["db_id"] for c in scorable
        if not c["baseline"]["strict_ex"] and c["consensus"]["strict_ex"]
    }
    # Evidence gate is informational, NEVER direct production promotion.
    useful = rescues - regressions >= 3 and regressions <= 1 and len(gained_db) >= 3
    return {
        "protocol": PROTOCOL, "cohort_sha256": shards[0]["cohort_sha256"],
        "model_digest": expected_digest,
        "requested": 20, "reference_scorable": len(scorable), "databases": 10,
        "baseline_ex": base_ex, "consensus_ex": new_ex,
        "baseline_bag": sum(c["baseline"]["bag"] for c in scorable),
        "consensus_bag": sum(c["consensus"]["bag"] for c in scorable),
        "oracle_ex": sum(c["oracle_correct"] is True for c in scorable),
        "rescues": rescues, "regressions": regressions,
        "agreement_cases": sum(c["consensus_index"] is not None for c in cases),
        "sql_diverse_cases": sum(c["candidate_diversity"] > 1 for c in cases),
        "latency_s": round(sum(c["total_latency_s"] for c in cases), 1),
        "positive_screen_only": useful, "promotion_allowed": False,
        "candidate_model": MODEL, "no_training": True,
        "gold_blind_selector": True, "oracle_not_deployable": True,
        "caveat": "Public Spider TRAIN, physically distinct from prior 100 and typed-IR 10 DBs; possible model pretraining contamination; not BIRD DEV100 or hidden test.",
        "cases": cases,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path)
    p.add_argument("--dev", type=Path)
    p.add_argument("--db-root", type=Path)
    p.add_argument("--shard", type=int)
    p.add_argument("--inputs", type=Path, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("refuse overwrite of previous score")
    if args.inputs:
        report = aggregate([json.loads(f.read_text(encoding="utf-8")) for f in args.inputs])
        print("QWEN_CONSENSUS_FINAL", json.dumps(
            {k:v for k,v in report.items() if k!="cases"}, sort_keys=True,
        ), flush=True)
    else:
        if any(x is None for x in (args.train,args.dev,args.db_root,args.shard)):
            p.error("shard requires train/dev/db-root/shard")
        report = run_shard(args.train,args.dev,args.db_root,args.shard)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
