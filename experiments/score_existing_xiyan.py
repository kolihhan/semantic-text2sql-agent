from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3


def execute(database: Path, sql: str) -> tuple[bool, set[tuple[object, ...]]]:
    try:
        with sqlite3.connect(database) as conn:
            rows = conn.execute(sql).fetchall()
        return True, set(rows)
    except sqlite3.Error:
        return False, set()


def resolve_database(database_root: Path, gold_sql: str) -> tuple[Path, set[tuple[object, ...]]]:
    matches: list[tuple[Path, set[tuple[object, ...]]]] = []
    for database in database_root.rglob("*.sqlite"):
        if "__MACOSX" in database.parts:
            continue
        ok, rows = execute(database, gold_sql)
        if ok:
            matches.append((database, rows))
    if len(matches) != 1:
        names = [str(path) for path, _ in matches]
        raise RuntimeError(f"Expected gold SQL to identify one database; got {len(matches)}: {names}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-root", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    database_root = Path(args.database_root)
    scored = []

    for row in results:
        database, gold_rows = resolve_database(database_root, str(row["gold"]))
        baseline_exec, baseline_rows = execute(database, str(row["baseline"]))
        xiyan_exec, xiyan_rows = execute(database, str(row["generated"]))
        scored.append({
            "id": row["id"],
            "kind": row["kind"],
            "database": database.stem,
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
        "mapping": "Each database is identified by the unique BIRD SQLite database on which the stored gold SQL executes successfully; positional case ids are not used.",
        "summary": summary,
        "cases": scored,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
