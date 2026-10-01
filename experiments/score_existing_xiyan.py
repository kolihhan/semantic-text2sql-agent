from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sqlite3


def normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip()).rstrip(";")


def resolve_database(database_root: Path, db_id: str) -> Path:
    direct = database_root / db_id / f"{db_id}.sqlite"
    if direct.is_file():
        return direct
    matches = [p for p in database_root.rglob("*.sqlite") if p.stem == db_id and "__MACOSX" not in p.parts]
    if len(matches) != 1:
        raise FileNotFoundError(f"SQLite database not found for {db_id}")
    return matches[0]


def execute(database: Path, sql: str) -> tuple[bool, set[tuple[object, ...]]]:
    try:
        with sqlite3.connect(database) as conn:
            rows = conn.execute(sql).fetchall()
        return True, set(rows)
    except sqlite3.Error:
        return False, set()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--database-root", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source_rows = json.loads(Path(args.source).read_text(encoding="utf-8"))
    by_gold: dict[str, list[dict[str, object]]] = {}
    for row in source_rows:
        by_gold.setdefault(normalize_sql(str(row["SQL"])), []).append(row)

    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    database_root = Path(args.database_root)
    scored = []

    for row in results:
        matches = by_gold.get(normalize_sql(str(row["gold"])), [])
        if len(matches) != 1:
            raise RuntimeError(f"Expected one BIRD row for gold SQL in case {row['id']}; got {len(matches)}")
        bird_row = matches[0]
        database = resolve_database(database_root, str(bird_row["db_id"]))
        gold_ok, gold_rows = execute(database, str(row["gold"]))
        if not gold_ok:
            raise RuntimeError(f"Gold SQL failed for case {row['id']}")
        baseline_exec, baseline_rows = execute(database, str(row["baseline"]))
        xiyan_exec, xiyan_rows = execute(database, str(row["generated"]))
        scored.append({
            "id": row["id"],
            "kind": row["kind"],
            "db_id": bird_row["db_id"],
            "baseline_execution_success": baseline_exec,
            "baseline_ex": baseline_exec and baseline_rows == gold_rows,
            "xiyan_execution_success": xiyan_exec,
            "xiyan_ex": xiyan_exec and xiyan_rows == gold_rows,
            "baseline": row["baseline"],
            "xiyan": row["generated"],
            "gold": row["gold"],
        })

    summary = {
        "total": len(scored),
        "baseline_ex": sum(bool(r["baseline_ex"]) for r in scored),
        "xiyan_ex": sum(bool(r["xiyan_ex"]) for r in scored),
        "wrong_to_correct": sum((not bool(r["baseline_ex"])) and bool(r["xiyan_ex"]) for r in scored),
        "correct_to_wrong": sum(bool(r["baseline_ex"]) and (not bool(r["xiyan_ex"])) for r in scored),
        "baseline_execution_success": sum(bool(r["baseline_execution_success"]) for r in scored),
        "xiyan_execution_success": sum(bool(r["xiyan_execution_success"]) for r in scored),
    }
    payload = {
        "note": "Handpicked diagnostic sample from the completed XiYan spike; not an unbiased benchmark.",
        "mapping": "Rows matched to the pinned BIRD release by normalized gold SQL, not unstable positional case id.",
        "summary": summary,
        "cases": scored,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
