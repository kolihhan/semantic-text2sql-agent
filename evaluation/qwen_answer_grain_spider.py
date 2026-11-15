"""One-shot causal screen: answer-entity/grain plan before Guarded Qwen SQL.

Spider DEV physical database holdout. Not BIRD, not official Spider test-suite.
No SQL training, selector, gold in prompts, or production change.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256, file_digest
import json
from pathlib import Path
import re
from time import monotonic

from evaluation.run_bird import CountingProvider, _ollama_model_digest
from evaluation.schema_context import select_schema_context
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.execution import SQLExecutionError, execute_readonly
from semantic_sql.inference import run_guarded
from semantic_sql.providers import OllamaProvider

PROTOCOL = "qwen9b-spider-12-answer-grain-v1"
MODEL = "qwen3.5:9b-q4_K_M"
DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
ARCHIVE_SHA = "00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b"
EXCLUDED_DB = frozenset({
    "pets_1", "dog_kennels", "orchestra", "voter_1", "course_teach", "poker_player"
})
# Corpus selection by question text only; broad relation semantics, not observed failures.
RELATION_WORDS = frozenset({
    "each", "every", "all", "both", "same", "different", "other",
    "more", "less", "least", "most", "without", "except", "not",
    "between", "whose", "among", "than", "and"
})
ARMS = ("baseline", "grain")
N_SHARDS = 3
MAX_ROWS = 50000
SYSTEM = (
    "From the user question and visible SQLite schema, describe the intended "
    "answer without writing SQL. Output exactly three lines:\n"
    "ANSWER_ENTITY: the entity or measurement that the answer reports\n"
    "ROW_GRAIN: what one output row represents, or one aggregate if a scalar\n"
    "RELATION: the condition connecting entities, including whether it requires "
    "the same entity, a shared set membership, or only a shared attribute\n"
    "Do not invent filters or values. Do not guess database contents. "
    "If unspecified write UNSPECIFIED. Keep the three lines short."
)


def fingerprint(value: str) -> str:
    return sha256(f"{PROTOCOL}:{value}".encode()).hexdigest()


def database(root: Path, dbid: str) -> Path:
    if not re.fullmatch(r"[a-zA-Z0-9_]+", dbid):
        raise ValueError("invalid database id")
    path = root / dbid / (dbid + ".sqlite")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def freeze(train: list[dict], dev: list[dict], root: Path) -> dict:
    """Frozen before looking at labels/predictions; distinct DBs and questions."""
    train_dbs = {str(row["db_id"]) for row in train}
    grouped = defaultdict(dict)
    for index, row in enumerate(dev):
        dbid = str(row["db_id"])
        if dbid in EXCLUDED_DB or dbid in train_dbs:
            continue
        question = " ".join(str(row.get("question") or "").split())
        if len(question) < 20:
            continue
        words = set(re.findall(r"[a-z]+", question.casefold()))
        if len(words & RELATION_WORDS) < 2:
            continue
        grouped[dbid].setdefault(question.casefold(), {
            "db_id": dbid, "index": index, "id": f"{dbid}:{index}", "question": question
        })
    eligible = sorted(
        (dbid for dbid, by_q in grouped.items() if len(by_q) >= 2),
        key=lambda dbid: fingerprint("db:" + dbid)
    )
    chosen = []
    for dbid in eligible:
        physical = database(root, dbid)
        if len(DatabaseCatalog.from_sqlite(physical).tables) < 2:
            continue
        rows = sorted(grouped[dbid].values(), key=lambda x: fingerprint(x["id"]))[:2]
        chosen.extend(rows)
        if len(chosen) == 12:
            break
    if len(chosen) != 12:
        raise ValueError(f"insufficient untouched relation questions: {len(chosen)}")
    assert len({x["db_id"] for x in chosen}) == 6
    assert len({x["id"] for x in chosen}) == 12
    return {
        "protocol": PROTOCOL, "model_digest": DIGEST, "archive_sha256": ARCHIVE_SHA,
        "cohort_sha256": sha256(json.dumps(chosen, sort_keys=True).encode()).hexdigest(),
        "total": 12, "physical_dbs": 6, "gold_blind": True,
        "conditional_cohort": "Spider DEV questions with >=2 general relation words",
        "excluded_prior_physical_dbs": sorted(EXCLUDED_DB), "cases": chosen
    }


def generate(db: Path, question: str, schema: str) -> tuple[dict, object]:
    provider = CountingProvider(OllamaProvider(
        model=MODEL, timeout_s=600, temperature=0, seed=42, num_predict=700
    ))
    start = monotonic()
    response = run_guarded(
        database=db, provider=provider, question=question,
        schema_context=schema, max_repairs=2, max_rows=MAX_ROWS
    )
    elapsed = round(monotonic() - start, 2)
    return {
        "sql": response.candidate.sql if response.candidate else "",
        "status": response.status, "capped": response.truncated,
        "calls": provider.total_calls, "seconds": elapsed,
    }, response


def run_shard(manifest: Path, train_path: Path, dev_path: Path, root: Path, shard: int) -> dict:
    saved = json.loads(manifest.read_text())
    train = json.loads(train_path.read_text())
    dev = json.loads(dev_path.read_text())
    if saved != freeze(train, dev, root) or saved["archive_sha256"] != ARCHIVE_SHA:
        raise ValueError("source, database selection, or preregistration drift")
    if _ollama_model_digest("http://127.0.0.1:11434", MODEL) != DIGEST:
        raise ValueError("Qwen model digest drift")
    if shard not in range(N_SHARDS):
        raise ValueError("invalid shard")
    cohort = saved["cases"][4 * shard:4 * shard + 4]
    predictions = []
    for i, row in enumerate(cohort):
        db = database(root, row["db_id"])
        catalog = DatabaseCatalog.from_sqlite(db)
        schema, _ = select_schema_context(catalog, root, row["db_id"], row["question"])
        start = monotonic()
        planner = OllamaProvider(
            model=MODEL, timeout_s=600, temperature=0, seed=42, num_predict=180
        )
        plan = planner.complete_text(
            system=SYSTEM, user=f"Question: {row['question']}\nSQLite schema:\n{schema}"
        ).strip()
        plan_latency = round(monotonic() - start, 2)
        if len(plan) > 1000 or not all(
            key + ":" in plan for key in ("ANSWER_ENTITY", "ROW_GRAIN", "RELATION")
        ):
            raise ValueError("unusable semantic contract; do not alter it after observing gold")
        contexts = {
            "baseline": schema,
            "grain": schema + "\nQuestion-level answer target (unverified model hypothesis):\n" + plan,
        }
        order = ARMS if (shard + i) % 2 == 0 else ARMS[::-1]
        outputs = {}
        result_rows = {}
        for arm in order:
            record, response = generate(db, row["question"], contexts[arm])
            outputs[arm] = record
            result_rows[arm] = response.rows
        predictions.append((row, plan, plan_latency, outputs, result_rows))
        print("GRAIN_PREDICTIONS_FROZEN", row["id"], flush=True)

    # Reference SQL is opened only after BOTH predictions on ALL shard cases exist.
    cases = []
    for row, plan, plan_latency, outputs, result_rows in predictions:
        db = database(root, row["db_id"])
        try:
            reference = execute_readonly(db, str(dev[row["index"]]["query"]), max_rows=MAX_ROWS)
            scorable = not reference.truncated
        except (SQLExecutionError, ValueError):
            reference = None
            scorable = False
        for arm in ARMS:
            record = outputs[arm]
            correct = record["status"] == "ok" and not record["capped"]
            record["set_ex"] = (
                bool(correct and set(map(tuple, result_rows[arm])) == set(reference.rows))
                if scorable else None
            )
            record["bag_ex"] = (
                bool(correct and Counter(map(tuple, result_rows[arm])) == Counter(reference.rows))
                if scorable else None
            )
        with db.open("rb") as file:
            db_sha = file_digest(file, "sha256").hexdigest()
        cases.append({
            "id": row["id"], "db_id": row["db_id"], "db_sha256": db_sha,
            "plan": plan, "plan_seconds": plan_latency, "scorable": scorable,
            "baseline": outputs["baseline"], "grain": outputs["grain"]
        })
    return {
        "protocol": PROTOCOL, "cohort_sha256": saved["cohort_sha256"],
        "model_digest": DIGEST, "shard": shard, "cases": cases
    }


def aggregate(shards: list[dict], expected_digest: str = DIGEST) -> dict:
    if len(shards) != N_SHARDS or sorted(x["shard"] for x in shards) != list(range(N_SHARDS)):
        raise ValueError("incomplete matrix")
    if any(
        x["protocol"] != PROTOCOL or x["model_digest"] != expected_digest
        or len(x["cases"]) != 4 for x in shards
    ) or len({x["cohort_sha256"] for x in shards}) != 1:
        raise ValueError("model/cohort mismatch")
    cases = [r for item in sorted(shards, key=lambda x: x["shard"]) for r in item["cases"]]
    if len({r["id"] for r in cases}) != 12 or len({r["db_id"] for r in cases}) != 6:
        raise ValueError("duplicate task or DB matrix")
    if any(len({r["db_sha256"] for r in cases if r["db_id"] == dbid}) != 1
           for dbid in {r["db_id"] for r in cases}):
        raise ValueError("nonidentical physical DB snapshots")
    scored = [r for r in cases if r["scorable"]]
    wins = [r for r in scored if r["grain"]["set_ex"] and not r["baseline"]["set_ex"]]
    losses = [r for r in scored if r["baseline"]["set_ex"] and not r["grain"]["set_ex"]]
    arms = {
        a: {
            "set_ex": sum(bool(r[a]["set_ex"]) for r in scored),
            "bag_ex": sum(bool(r[a]["bag_ex"]) for r in scored),
            "calls": sum(r[a]["calls"] for r in cases) + (len(cases) if a == "grain" else 0),
            "seconds": round(sum(r[a]["seconds"] for r in cases) +
                             (sum(r["plan_seconds"] for r in cases) if a == "grain" else 0), 2)
        } for a in ARMS
    }
    passed = (
        len(scored) == 12 and len(wins) - len(losses) >= 2 and len(losses) == 0
        and len({r["db_id"] for r in wins}) >= 2
        and arms["grain"]["bag_ex"] >= arms["baseline"]["bag_ex"]
    )
    return {
        "protocol": PROTOCOL, "total": 12, "scorable": len(scored),
        "physical_dbs": 6, "model_digest": expected_digest, "arms": arms,
        "rescues": [r["id"] for r in wins], "regressions": [r["id"] for r in losses],
        "positive_screen_only": passed, "promotion_allowed": False,
        "no_model_training": True, "not_BIRD_or_official_Spider": True,
        "cases": cases,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("train", "dev", "db-root", "manifest", "output"):
        p.add_argument("--" + key, type=Path)
    p.add_argument("--shard", type=int)
    p.add_argument("--inputs", type=Path, nargs="+")
    a = p.parse_args()
    if a.output is None or a.output.exists():
        raise FileExistsError("output path must be new")
    if a.inputs:
        result = aggregate([json.loads(f.read_text()) for f in a.inputs])
        print("SPIDER_QWEN_GRAIN_FINAL", json.dumps(
            {k: v for k, v in result.items() if k != "cases"}, sort_keys=True
        ), flush=True)
    elif a.manifest:
        result = run_shard(a.manifest, a.train, a.dev, a.db_root, a.shard)
    else:
        if any(x is None for x in (a.train, a.dev, a.db_root)):
            p.error("preflight requires train, dev and physical db root")
        result = freeze(json.loads(a.train.read_text()), json.loads(a.dev.read_text()), a.db_root)
        print("SPIDER_QWEN_GRAIN_FROZEN", result["cohort_sha256"], flush=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
