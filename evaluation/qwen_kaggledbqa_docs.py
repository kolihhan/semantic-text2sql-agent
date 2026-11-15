"""Paired real SQLite test of publisher-authored column definitions on KaggleDBQA."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from hashlib import file_digest, sha256
import json
from pathlib import Path
import sqlite3
from time import monotonic

from evaluation.run_bird import CountingProvider, _ollama_model_digest, _resolve_database, _write
from evaluation.schema_context import select_schema_context
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.execution import SQLExecutionError, execute_readonly, validate_readonly_sql
from semantic_sql.inference import run_guarded
from semantic_sql.providers import OllamaProvider


MODEL = "qwen3.5:9b-q4_K_M"
DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
DATA_REV = "ab6325c9b5749f2f3509a1f64299bfa30396e6b0"
PROTOCOL = "kaggledbqa-qwen9b-column-docs-18-v1"
DBS = (
    "GeoNuclearData", "GreaterManchesterCrime", "Pesticide", "StudentMathScore",
    "USWildFires", "WhatCDHipHop",
)
PER_DB = 3
MAX_ROWS = 50000
ARMS = ("baseline", "documented")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def questions_for(db_id: str, raw: list[dict]) -> list[tuple[int, str]]:
    # Select on question identity only. Never choose after observing either score.
    unique: dict[str, tuple[int, str]] = {}
    for index, row in enumerate(raw):
        if row["db_id"] != db_id or not row["question"].strip():
            raise ValueError("source database/question mismatch")
        question = row["question"].strip()
        unique.setdefault(" ".join(question.casefold().split()), (index, question))
    selected = sorted(
        unique.values(),
        key=lambda item: sha256(f"{PROTOCOL}:{db_id}:{item[0]}:{item[1]}".encode()).hexdigest(),
    )[:PER_DB]
    if len(selected) != PER_DB:
        raise ValueError(f"insufficient distinct official test questions for {db_id}")
    return selected


def column_docs(meta: dict, catalog: DatabaseCatalog, visible: list[str]) -> dict[tuple[str, str], str]:
    descriptions = {}
    for (table_index, column), text in zip(
        meta["column_names_original"], meta["column_descriptions"], strict=True,
    ):
        if table_index < 0 or not text:
            continue
        table = meta["table_names_original"][table_index]
        if table not in catalog.tables or column not in catalog.tables[table].columns:
            raise ValueError(f"published definition does not match physical schema: {table}.{column}")
        if table in visible:
            descriptions[table, column] = " ".join(text.split())
    return descriptions


def referenced_columns(db: Path, sql: str) -> set[tuple[str, str]]:
    """Ask SQLite to resolve base column references, including aliases and joins."""
    if not sql:
        return set()
    columns = set()

    def record(action, table, column, database, trigger):
        if action == sqlite3.SQLITE_READ and database == "main" and column:
            columns.add((table, column))
        return sqlite3.SQLITE_OK

    try:
        with closing(sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
            connection.set_authorizer(record)
            connection.execute("EXPLAIN " + validate_readonly_sql(sql))
    except (sqlite3.Error, SQLExecutionError):
        return set()
    return columns


def run_pair(db: Path, question: str, inputs: dict[str, str], order: tuple[str, ...]):
    predictions = {}
    for arm in order:
        provider = CountingProvider(OllamaProvider(
            model=MODEL, timeout_s=600, temperature=0, seed=42, num_predict=1024,
        ))
        started = monotonic()
        result = run_guarded(
            database=db, provider=provider, question=question,
            schema_context=inputs[arm], max_repairs=2, max_rows=MAX_ROWS,
        )
        predictions[arm] = (result, provider.total_calls, round(monotonic() - started, 2))
    return predictions


def run_database(source: Path, db_root: Path, db_id: str) -> dict:
    if _ollama_model_digest("http://127.0.0.1:11434", MODEL) != DIGEST:
        raise ValueError("wrong physical Qwen model digest")
    db = _resolve_database(db_root, db_id)
    catalog = DatabaseCatalog.from_sqlite(db)
    meta = next(row for row in load_json(source / "KaggleDBQA_tables.json") if row["db_id"] == db_id)
    raw = load_json(source / "examples" / f"{db_id}_test.json")
    frozen = questions_for(db_id, raw)
    print("KDB_FROZEN", db_id, [index for index, _ in frozen], flush=True)
    generated = []
    for position, (index, question) in enumerate(frozen):
        baseline, diag = select_schema_context(catalog, db_root, db_id, question)
        definitions = column_docs(meta, catalog, diag["selected_tables"])
        if not definitions:
            raise ValueError(f"no source documentation for frozen {db_id}:{index}")
        augmented = baseline + "\nPublished column definitions:\n" + "\n".join(
            f"- {table}.{column}: {description}" for (table, column), description in definitions.items()
        )
        order = ARMS if (DBS.index(db_id) + position) % 2 == 0 else ARMS[::-1]
        predictions = run_pair(db, question, {"baseline": baseline, "documented": augmented}, order)
        generated.append((index, question, definitions, predictions))
        print("KDB_PAIR_GENERATED", db_id, index, flush=True)

    cases = []
    for index, question, definitions, predictions in generated:
        gold_sql = raw[index]["query"]
        try:
            gold = execute_readonly(db, gold_sql, max_rows=MAX_ROWS)
            reference_error = "truncated" if gold.truncated else None
        except SQLExecutionError as exc:
            gold = None
            reference_error = str(exc)
        target_refs = set(definitions) & referenced_columns(db, gold_sql)
        case = {
            "id": f"{db_id}:{index}", "db": db_id, "question": question, "reference_sql": gold_sql,
            "reference_error": reference_error, "documentation_columns": len(definitions),
            "gold_documented_columns": sorted(target_refs),
        }
        for arm, (result, calls, seconds) in predictions.items():
            sql = result.candidate.sql if result.candidate else ""
            valid = result.status == "ok" and not result.truncated
            refs = referenced_columns(db, sql) if valid else set()
            scorable = reference_error is None
            case[arm] = {
                "sql": sql, "executable": result.status == "ok", "truncated": result.truncated,
                "set_ex": bool(valid and set(result.rows) == set(gold.rows)) if scorable else None,
                "bag_ex": bool(valid and Counter(result.rows) == Counter(gold.rows)) if scorable else None,
                "gold_documented_columns_used": sorted(target_refs & refs),
                "calls": calls, "seconds": seconds,
            }
        cases.append(case)
    with db.open("rb") as stream:
        db_sha = file_digest(stream, "sha256").hexdigest()
    return {"protocol": PROTOCOL, "model_digest": DIGEST, "db": db_id, "db_sha": db_sha, "cases": cases}


def aggregate(shards: list[dict]) -> dict:
    if sorted(item["db"] for item in shards) != sorted(DBS) or any(
        item["protocol"] != PROTOCOL or item["model_digest"] != DIGEST
        or len(item["cases"]) != PER_DB or any(row["db"] != item["db"] for row in item["cases"])
        for item in shards
    ):
        raise ValueError("unexpected physical-db matrix, model or cases")
    if len({item["db_sha"] for item in shards}) != len(DBS):
        raise ValueError("duplicate physical SQLite databases")
    cases = sorted((row for item in shards for row in item["cases"]), key=lambda row: row["id"])
    if len({row["id"] for row in cases}) != len(DBS) * PER_DB:
        raise ValueError("repeated case id")
    scored = [row for row in cases if row["reference_error"] is None]
    summary = {
        arm: {
            "set_ex": sum(row[arm]["set_ex"] for row in scored),
            "bag_ex": sum(row[arm]["bag_ex"] for row in scored),
            "executed": sum(row[arm]["executable"] for row in cases),
            "truncated": sum(row[arm]["truncated"] for row in cases),
            "calls": sum(row[arm]["calls"] for row in cases),
            "seconds": round(sum(row[arm]["seconds"] for row in cases), 1),
        }
        for arm in ARMS
    }
    rescued = [row for row in scored if not row["baseline"]["set_ex"] and row["documented"]["set_ex"]]
    regressed = [row for row in scored if row["baseline"]["set_ex"] and not row["documented"]["set_ex"]]
    improved_dbs = sorted({row["db"] for row in rescued})
    positive = (
        len(scored) >= 16 and len(rescued) - len(regressed) >= 3 and len(regressed) <= 1
        and len(improved_dbs) >= 3 and summary["documented"]["bag_ex"] >= summary["baseline"]["bag_ex"]
    )
    return {
        "protocol": PROTOCOL, "source_revision": DATA_REV, "model": MODEL, "model_digest": DIGEST,
        "database_sha256": {item["db"]: item["db_sha"] for item in shards},
        "total": len(cases), "scorable": len(scored), "arms": summary, "improved_dbs": improved_dbs,
        "rescues": [row["id"] for row in rescued], "regressions": [row["id"] for row in regressed],
        "rescues_with_more_gold_documented_columns": sum(
            len(row["documented"]["gold_documented_columns_used"])
            > len(row["baseline"]["gold_documented_columns_used"])
            for row in rescued
        ),
        "positive_screen": positive, "promotion_allowed": False, "cases": cases,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path)
    p.add_argument("--db-root", type=Path)
    p.add_argument("--db", choices=DBS)
    p.add_argument("--inputs", type=Path, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.inputs:
        result = aggregate([load_json(path) for path in args.inputs])
        print("KDB_QWEN9B_DOCS_FINAL", json.dumps({k: v for k, v in result.items() if k != "cases"}), flush=True)
    else:
        if args.source is None or args.db_root is None or args.db is None:
            p.error("--source, --db-root and --db are required in a physical shard")
        result = run_database(args.source, args.db_root, args.db)
    _write(args.output, result)


if __name__ == "__main__":
    main()
