"""Same-Qwen Guarded vs typed-IR compiler on unused public Spider TRAIN databases.

Independent of earlier 100 Spider TRAIN schemas, BIRD datasets and DEV schemas.
This is a screening experiment on PUBLIC examples, not hidden-test improvement.
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
from evaluation.typed_query_ir import draft_typed_query_sql
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.contracts import SQLCandidate
from semantic_sql.execution import SQLExecutionError, execute_readonly
from semantic_sql.inference import run_guarded
from semantic_sql.providers import OllamaProvider

MODEL = "qwen3.5:9b-q4_K_M"
MODEL_DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
SEED_OLD = "xiyan-a5-100-spider-train-db-disjoint-from-previous-dev-20261113-v1"
SEED_NEW = "qwen9b-typed-ir-spider-train-new-db20-20261115-v1"
PROTOCOL = "qwen9b-typed-ir-spider-shift20-v1"
ARMS = ("original_guarded", "typed_ir_guarded")
N_SHARDS = 5


def _hash(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def select_cases(train: list[dict], dev: list[dict]) -> list[dict]:
    """Freeze only questions/DB IDs, never reference SQL, model errors or outcomes."""
    if not isinstance(train, list) or not isinstance(dev, list):
        raise ValueError("invalid Spider source")
    dev_ids = {str(x["db_id"]) for x in dev}
    groups: dict[str, list[dict]] = defaultdict(list)
    for idx, row in enumerate(train):
        db, question = str(row.get("db_id") or ""), str(row.get("question") or "").strip()
        if not db or not question:
            raise ValueError("invalid source question")
        if db in dev_ids:
            raise ValueError("train/dev database overlap")
        groups[db].append({"case_id": f"{db}:{idx}", "row_index": idx,
                           "database_id": db, "question": question})
    eligible = [db for db, items in groups.items() if len(items) >= 2]
    # Reproduce old PR #21's first-100-database deterministic selection
    # (all TRAIN DBs there had >=1 case; independent of correctness and SQL).
    prior_order = sorted(groups, key=lambda d: _hash(f"{SEED_OLD}:database:{d}"))
    if len(prior_order) < 110:
        raise ValueError("not enough fresh physical databases after prior 100")
    old = set(prior_order[:100])
    fresh = sorted((db for db in eligible if db not in old),
                   key=lambda d: _hash(f"{SEED_NEW}:database:{d}"))
    if len(fresh) < 10:
        raise ValueError("fewer than ten fresh Spider databases")
    chosen = []
    for db in fresh[:10]:
        questions = sorted(groups[db],
                           key=lambda row: _hash(f"{SEED_NEW}:{row['case_id']}"))
        chosen.extend(questions[:2])
    if len(chosen) != 20 or len({x["case_id"] for x in chosen}) != 20:
        raise AssertionError("invalid disjoint paired cohort")
    return chosen


def _path(root: Path, db_id: str) -> Path:
    db = root / db_id / f"{db_id}.sqlite"
    if not db.is_file():
        raise FileNotFoundError(f"physical Spider SQLite not available: {db_id}")
    return db


def _result(result, elapsed: float, *, extra_calls: int = 0) -> dict:
    good = result.status == "ok" and not result.truncated
    return {
        "sql": result.candidate.sql if result.candidate else "",
        "status": result.status,
        "execution_success": good,
        "rows": result.rows if good else (),
        "repairs": result.candidate.attempt if result.candidate else 0,
        "extra_ir_calls": extra_calls,
        "latency_s": round(elapsed, 2),
    }


def run_shard(train_file: Path, dev_file: Path, db_root: Path, shard: int) -> dict:
    if not 0 <= shard < N_SHARDS:
        raise ValueError("bad shard")
    train = json.loads(train_file.read_text(encoding="utf-8"))
    dev = json.loads(dev_file.read_text(encoding="utf-8"))
    selected = select_cases(train, dev)
    identity = _hash(json.dumps([(x["case_id"], x["database_id"]) for x in selected],
                                separators=(",", ":")))
    chosen = selected[shard * 4: (shard + 1) * 4]
    if len(chosen) != 4:
        raise ValueError("incomplete shard")
    digest = _ollama_model_digest("http://127.0.0.1:11434", MODEL)
    if digest != MODEL_DIGEST:
        raise RuntimeError("wrong Qwen checkpoint digest")
    provider = OllamaProvider(
        model=MODEL, base_url="http://127.0.0.1:11434",
        timeout_s=900, temperature=0, seed=42, num_predict=1024,
    )
    # All candidates for the shard are frozen before any reference SQL is read.
    generated = []
    for case in chosen:
        db = _path(db_root, case["database_id"])
        cat = DatabaseCatalog.from_sqlite(db)
        schema, _ = select_schema_context(cat, db_root, case["database_id"], case["question"])
        began = monotonic()
        baseline = run_guarded(
            database=db, provider=provider, question=case["question"],
            schema_context=schema, max_repairs=2, max_rows=50000,
        )
        old_time = monotonic() - began
        began = monotonic()
        compiled = draft_typed_query_sql(case["question"], schema, cat, provider)
        if compiled is None:
            enhanced = baseline
            action = "abstain_original_preserved"
            extra_calls = 1
        else:
            enhanced = run_guarded(
                database=db, provider=provider, question=case["question"],
                schema_context=schema, initial_candidate=SQLCandidate(sql=compiled, attempt=0),
                max_repairs=2, max_rows=50000,
            )
            action = "compiled_guarded"
            extra_calls = 1
        new_time = monotonic() - began
        generated.append({
            "case_id": case["case_id"], "db_id": case["database_id"],
            "row_index": case["row_index"], "typed_ir_compiled": compiled is not None,
            "typed_ir_action": action, "schema_sha256": _hash(schema),
            "arms": {
                ARMS[0]: _result(baseline, old_time),
                ARMS[1]: _result(enhanced, new_time, extra_calls=extra_calls),
            },
        })
        print("QWEN_TYPED_IR_GENERATED", shard, case["case_id"], action, flush=True)
    for row in generated:
        db = _path(db_root, row["db_id"])
        gold_sql = str(train[row["row_index"]]["query"])
        try:
            gold = execute_readonly(db, gold_sql, max_rows=50000)
            if gold.truncated:
                raise ValueError("reference truncated")
            scorable = True
            expected_set, expected_bag = set(gold.rows), Counter(gold.rows)
        except (SQLExecutionError, ValueError):
            scorable = False
            expected_set, expected_bag = set(), Counter()
        row["reference_scorable"] = scorable
        for arm in row["arms"].values():
            rows = arm.pop("rows")
            arm["strict_ex"] = bool(scorable and arm["execution_success"] and set(rows) == expected_set) if scorable else None
            arm["bag"] = bool(scorable and arm["execution_success"] and Counter(rows) == expected_bag) if scorable else None
    return {"protocol": PROTOCOL, "shard": shard, "cohort_sha256": identity,
            "model_digest": digest, "source": "public Spider TRAIN (not unseen)",
            "cases": generated}


def aggregate(shards: list[dict]) -> dict:
    if len(shards) != 5 or sorted(s.get("shard") for s in shards) != list(range(5)):
        raise ValueError("all 5 distinct shards are required")
    if (len({s.get("cohort_sha256") for s in shards}) != 1
        or len({s.get("model_digest") for s in shards}) != 1
        or any(s.get("protocol") != PROTOCOL for s in shards)):
        raise ValueError("cohort/model/protocol drift")
    rows = [r for s in sorted(shards, key=lambda s: s["shard"]) for r in s["cases"]]
    if len(rows) != 20 or len({r["case_id"] for r in rows}) != 20 or len({r["db_id"] for r in rows}) != 10:
        raise ValueError("missing/duplicated questions or physical databases")
    valid = [r for r in rows if r["reference_scorable"]]
    old, new = ARMS
    rescues = sum(r["arms"][old]["strict_ex"] is False and r["arms"][new]["strict_ex"] is True for r in valid)
    regressions = sum(r["arms"][old]["strict_ex"] is True and r["arms"][new]["strict_ex"] is False for r in valid)
    counts = {arm: {
        "strict_ex": sum(r["arms"][arm]["strict_ex"] is True for r in valid),
        "bag": sum(r["arms"][arm]["bag"] is True for r in valid),
        "execution_success": sum(r["arms"][arm]["execution_success"] for r in rows),
        "latency_s": round(sum(r["arms"][arm]["latency_s"] for r in rows), 1),
    } for arm in ARMS}
    if counts[new]["strict_ex"] - counts[old]["strict_ex"] != rescues - regressions:
        raise AssertionError("paired score mismatch")
    gains = {r["db_id"] for r in valid
             if r["arms"][old]["strict_ex"] is False and r["arms"][new]["strict_ex"] is True}
    screen_positive = (rescues - regressions >= 3 and regressions <= 1 and
                       len(gains) >= 3 and counts[new]["bag"] >= counts[old]["bag"])
    return {
        "protocol": PROTOCOL, "model_digest": shards[0]["model_digest"],
        "cohort_sha256": shards[0]["cohort_sha256"],
        "requested": 20, "reference_scorable": len(valid), "databases": 10,
        "typed_ir_compiled": sum(r["typed_ir_compiled"] for r in rows),
        "typed_ir_abstained": sum(not r["typed_ir_compiled"] for r in rows),
        "arms": counts, "rescues": rescues, "regressions": regressions,
        "rescued_databases": len(gains), "positive_screen_only": screen_positive,
        "promotion_allowed": False,
        "caveat": "Public Spider TRAIN, different DBs from earlier Spider100/DEV and BIRD; never hidden model test, never evidence of BIRD DEV100 improvement.",
        "cases": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path)
    ap.add_argument("--dev", type=Path)
    ap.add_argument("--db-root", type=Path)
    ap.add_argument("--shard", type=int)
    ap.add_argument("--inputs", type=Path, nargs="+")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError("refuse overwriting paired output")
    if a.inputs:
        report = aggregate([json.loads(p.read_text()) for p in a.inputs])
        print("QWEN_TYPED_IR_SPIDER_FINAL", json.dumps({k: v for k, v in report.items() if k != "cases"}, sort_keys=True), flush=True)
    else:
        if any(x is None for x in (a.train, a.dev, a.db_root, a.shard)):
            ap.error("shard needs train/dev/database-root/shard")
        report = run_shard(a.train, a.dev, a.db_root, a.shard)
    a.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
