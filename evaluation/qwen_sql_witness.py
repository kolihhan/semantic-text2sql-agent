"""Paired Qwen Guarded trial with one executable pre-generation probe."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from hashlib import file_digest, sha256
import json
from pathlib import Path
import sqlite3
from time import monotonic

from evaluation.run_bird import CountingProvider, _ollama_model_digest, _write
from evaluation.schema_context import select_schema_context
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.direct import _strip_fence
from semantic_sql.execution import SQLExecutionError, _open_readonly_connection, execute_readonly
from semantic_sql.inference import run_guarded
from semantic_sql.providers import OllamaProvider
from semantic_sql.verifier import verify_sql_preflight


BASE_URL = "http://127.0.0.1:11434"
MODEL = "qwen3.5:9b-q4_K_M"
DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
PROTOCOL = "qwen9b-real-sql-witness-v1"
ARMS = ("guarded", "probe_text", "probe_rows")
SOURCES = {"chinook": "heldout.jsonl", "sakila": "sakila_questions.jsonl"}
MAX_ROWS = 50_000
PROBE_SYSTEM = """Choose one narrow SQLite SELECT or CTE to check a data assumption needed to answer the question.
Use the supplied schema. Focus on a relevant stored label/code, value representation, or join multiplicity.
Use an aggregate or return at most five example rows. Check one assumption. Return only the SQL for that check.
If a data check would add nothing, return NONE."""


def freeze(source: Path, dataset: str) -> list[dict]:
    rows = [json.loads(line) for line in (source / SOURCES[dataset]).read_text().splitlines() if line.strip()]
    selected = []
    for difficulty in ("medium", "hard"):
        group = [r for r in rows if r["difficulty"] == difficulty]
        group.sort(key=lambda r: sha256(f"{PROTOCOL}:{dataset}:{r['id']}".encode()).digest())
        if len(group) < 4:
            raise ValueError("selected source is missing independently authored tasks")
        selected.extend(group[:4])
    return selected


def sample(db: Path, sql: str) -> dict:
    if not sql or sql.upper() == "NONE":
        return {"status": "skipped"}
    if not verify_sql_preflight(db, sql).ok:
        return {"status": "rejected"}
    started = monotonic()
    conn, _ = _open_readonly_connection(db)
    try:
        with closing(conn):
            conn.set_progress_handler(lambda: int(monotonic() - started > 2), 2000)
            cur = conn.execute(sql)
            rows = cur.fetchmany(6)
            return {
                "status": "ok", "columns": [col[0] for col in cur.description],
                "rows": [[value if not isinstance(value, (str, bytes)) else str(value)[:120]
                          for value in row] for row in rows[:5]],
                "truncated": len(rows) > 5,
            }
    except sqlite3.Error:
        return {"status": "execution_error"}


def generate(db: Path, question: str, schema: str, evidence: str | None) -> tuple:
    provider = CountingProvider(OllamaProvider(
        model=MODEL, base_url=BASE_URL, timeout_s=300, temperature=0, seed=42, num_predict=700,
    ))
    began = monotonic()
    result = run_guarded(database=db, provider=provider, question=question,
                         schema_context=schema, evidence=evidence, max_repairs=2, max_rows=MAX_ROWS)
    return result, provider.total_calls, round(monotonic() - began, 2)


def benchmark(source: Path, db: Path, dataset: str, shard: int) -> dict:
    if _ollama_model_digest(BASE_URL, MODEL) != DIGEST:
        raise ValueError("unexpected physical Qwen manifest")
    rows = freeze(source, dataset)
    cat = DatabaseCatalog.from_sqlite(db)
    cases = []
    for offset, row in enumerate(rows[shard * 4:shard * 4 + 4]):
        question = row["question"]
        schema, _ = select_schema_context(cat, db.parent, dataset, question)
        provider = OllamaProvider(model=MODEL, base_url=BASE_URL, timeout_s=300,
                                  temperature=0, seed=42, num_predict=256)
        started = monotonic()
        probe = _strip_fence(provider.complete_text(system=PROBE_SYSTEM,
            user=f"Question: {question}\nSQLite schema:\n{schema}"))
        observed = sample(db, probe)
        probe_seconds = round(monotonic() - started, 2)
        hints = {"guarded": None, "probe_text": None, "probe_rows": None}
        if observed["status"] == "ok":
            text = f"Preliminary read-only data probe:\n{probe}\n"
            hints["probe_text"] = text + "Observed data: not supplied."
            hints["probe_rows"] = text + "Observed data: " + json.dumps(observed, ensure_ascii=False)
        shift = (offset + shard + (dataset == "sakila")) % len(ARMS)
        order = ARMS[shift:] + ARMS[:shift]
        predictions = {}
        for arm in order:
            prior = next((seen for seen in predictions if hints[seen] == hints[arm]), None)
            predictions[arm] = predictions[prior] if prior else generate(db, question, schema, hints[arm])
        cases.append((row, predictions, probe, observed, probe_seconds))
        print("WITNESS_FROZEN_PREDICTIONS", dataset, row["id"], observed["status"], flush=True)

    scored = []
    for row, predictions, probe, observed, probe_seconds in cases:
        gold = row["gold_sql"]
        references = [gold] if isinstance(gold, str) else gold
        reference_rows = []
        for statement in references:
            try:
                target = execute_readonly(db, statement, max_rows=MAX_ROWS)
            except SQLExecutionError:
                continue
            if not target.truncated:
                reference_rows.append(target.rows)
        scorable = bool(reference_rows)
        target_sets = [set(rows) for rows in reference_rows]
        target_bags = [Counter(rows) for rows in reference_rows]
        output = {"id": row["id"], "dataset": dataset, "difficulty": row["difficulty"],
                  "probe": probe, "observation": observed, "probe_seconds": probe_seconds,
                  "scorable": scorable, "reference_sql": gold, "scorable_references": len(reference_rows)}
        for arm, (result, calls, seconds) in predictions.items():
            usable = result.status == "ok" and not result.truncated
            output[arm] = {
                "sql": result.candidate.sql if result.candidate else "",
                "set_ex": bool(usable and set(result.rows) in target_sets) if scorable else None,
                "bag_ex": bool(usable and Counter(result.rows) in target_bags) if scorable else None,
                "seconds": seconds, "calls": calls, "status": result.status, "capped": result.truncated,
            }
        scored.append(output)
    with db.open("rb") as source_db:
        fingerprint = file_digest(source_db, "sha256").hexdigest()
    return {"protocol": PROTOCOL, "digest": DIGEST, "dataset": dataset, "shard": shard,
            "db_sha": fingerprint, "cases": scored}


def aggregate(shards: list[dict]) -> dict:
    keys = {(s["dataset"], s["shard"]) for s in shards}
    if len(shards) != 4 or keys != {(db, i) for db in SOURCES for i in range(2)} or any(
        s["protocol"] != PROTOCOL or s["digest"] != DIGEST or len(s["cases"]) != 4 for s in shards
    ):
        raise ValueError("incomplete physical database comparison")
    if any(len({s["db_sha"] for s in shards if s["dataset"] == db}) != 1 for db in SOURCES):
        raise ValueError("database snapshot changed across shards")
    rows = [row for shard in sorted(shards, key=lambda s: (s["dataset"], s["shard"])) for row in shard["cases"]]
    if len({(r["dataset"], r["id"]) for r in rows}) != 16:
        raise ValueError("duplicate benchmark identity")
    scored = [r for r in rows if r["scorable"]]
    by_arm = {a: {k: sum(r[a][k] for r in scored) for k in ("set_ex", "bag_ex")} for a in ARMS}
    for arm in ARMS:
        by_arm[arm].update(
            calls=sum(r[arm]["calls"] for r in rows) + (len(rows) if arm != "guarded" else 0),
            seconds=round(sum(r[arm]["seconds"] for r in rows) +
                          (sum(r["probe_seconds"] for r in rows) if arm != "guarded" else 0), 1),
        )
    transitions = {}
    for control in ARMS[:2]:
        rescues = [r for r in scored if r["probe_rows"]["set_ex"] and not r[control]["set_ex"]]
        regress = [r for r in scored if not r["probe_rows"]["set_ex"] and r[control]["set_ex"]]
        fmt = lambda items: [f"{r['dataset']}:{r['id']}" for r in items]
        transitions[control] = {"rescues": fmt(rescues), "regressions": fmt(regress),
                                "rescued_dbs": sorted({r["dataset"] for r in rescues})}
    primary = transitions["guarded"]
    causal = transitions["probe_text"]
    net = lambda x: len(x["rescues"]) - len(x["regressions"])
    passed = (len(scored) >= 14 and net(primary) >= 3 and len(primary["regressions"]) <= 1
              and len(primary["rescued_dbs"]) == 2 and net(causal) >= 2
              and by_arm["probe_rows"]["bag_ex"] >= by_arm["guarded"]["bag_ex"])
    return {
        "protocol": PROTOCOL, "digest": DIGEST, "total": len(rows), "scorable": len(scored),
        "physical_observations": dict(Counter(r["observation"]["status"] for r in rows)),
        "nonempty_observations": sum(bool(r["observation"].get("rows")) for r in rows),
        "database_sha256": {s["dataset"]: s["db_sha"] for s in shards}, "arms": by_arm,
        "transitions": transitions, "positive_screen": passed, "promotion_allowed": False, "cases": rows,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path)
    p.add_argument("--db", type=Path)
    p.add_argument("--dataset", choices=SOURCES)
    p.add_argument("--shard", type=int, choices=(0, 1))
    p.add_argument("--inputs", nargs="+", type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.inputs:
        result = aggregate([json.loads(path.read_text()) for path in a.inputs])
        print("WITNESS_FINAL", json.dumps({k: v for k, v in result.items() if k != "cases"}), flush=True)
    else:
        if a.source is None or a.db is None or a.dataset is None or a.shard is None:
            p.error("physical shards require --source, --db, --dataset and --shard")
        result = benchmark(a.source, a.db, a.dataset, a.shard)
    _write(a.output, result)


if __name__ == "__main__":
    main()
