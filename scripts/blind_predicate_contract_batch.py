"""Label-blind, fail-closed batch diagnostic. No gold SQL, no scoring, no rewrites."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from scripts.predicate_contract_probe import inspect_predicate

FORBIDDEN = ("holdout300", "simpson_episodes", "works_cycles")
REQUIRED = {"case_id", "question", "selected_sql", "database", "database_sha256"}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def diagnose(rows: list[dict], allowed_root: Path) -> dict:
    out = []
    seen = set()
    root = allowed_root.resolve()
    for row in rows:
        if set(row) != REQUIRED:
            raise ValueError("Unexpected/missing fields: labels and gold SQL forbidden")
        case_id = str(row["case_id"])
        if case_id in seen:
            raise ValueError("Duplicate case_id")
        seen.add(case_id)
        db = Path(row["database"]).resolve()
        if any(x in str(db).lower() for x in FORBIDDEN):
            raise ValueError("Protected database")
        if not db.is_relative_to(root) or not db.is_file():
            raise ValueError("Database outside allowlisted root or missing")
        actual = sha256(db)
        if actual != row["database_sha256"]:
            raise ValueError("Database digest mismatch")
        observation = inspect_predicate(row["question"], row["selected_sql"], db)
        # Do not export SQL, literal values, question, or physical database path.
        out.append({"case_id": case_id, "db_sha256": actual,
                    "candidate_sha256": hashlib.sha256(row["selected_sql"].encode()).hexdigest(),
                    "status": observation["status"], "reason": observation["reason"]})
    return {"protocol": "blind-predicate-contract-v1", "count": len(out), "cases": out}

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--allowed-db-root", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    source = Path(args.input)
    if any(x in str(source).lower() for x in FORBIDDEN):
        raise ValueError("Protected source")
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    result = diagnose(rows, Path(args.allowed_db_root))
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

if __name__ == "__main__":
    main()
