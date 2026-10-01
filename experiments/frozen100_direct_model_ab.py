from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "evaluation"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bird_loader import load_bird_json
from schema_context import select_schema_context
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.direct import generate_direct_sql
from semantic_sql.execution import SQLExecutionError, execute_readonly
from semantic_sql.providers import OllamaProvider

BASELINE = ROOT / "runs" / "bird-paired-dev-v4" / "paired.json"
EXPECTED_SOURCE_SHA256 = "0310b7ca2dfb6247234e2073f93e7a75106ee41ba55231f75d06d2ef68efe2ec"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_database(database_root: Path, db_id: str) -> Path:
    direct = database_root / db_id / f"{db_id}.sqlite"
    if direct.is_file():
        return direct
    matches = list((database_root / db_id).glob("*.sqlite"))
    if len(matches) == 1:
        return matches[0]
    matches = [p for p in database_root.rglob("*.sqlite") if p.stem == db_id]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"SQLite database not found for {db_id}")


def ex_equal(database: Path, predicted_sql: str, gold_sql: str) -> tuple[bool, bool]:
    try:
        predicted = execute_readonly(database, predicted_sql, max_rows=50_000)
    except SQLExecutionError:
        return False, False
    try:
        gold = execute_readonly(database, gold_sql, max_rows=50_000)
    except SQLExecutionError:
        raise RuntimeError("gold SQL failed")
    if predicted.truncated or gold.truncated:
        raise RuntimeError("truncated execution invalidates EX")
    return True, set(predicted.rows) == set(gold.rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--database-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    source = Path(args.source)
    database_root = Path(args.database_root)
    output = Path(args.output)
    source_hash = sha256(source)

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    baseline_rows = {str(row["case_id"]): row for row in baseline["cases"]}
    selected_ids = list(baseline_rows)
    cases, labels = load_bird_json(source)
    case_by_id = {c.case_id: c for c in cases}
    label_by_id = {l.case_id: l for l in labels}

    missing = [case_id for case_id in selected_ids if case_id not in case_by_id or case_id not in label_by_id]
    if missing:
        raise RuntimeError(f"selected cases missing from source: {missing[:5]}")
    mismatched = []
    for case_id in selected_ids:
        case = case_by_id[case_id]
        old = baseline_rows[case_id]
        if (
            case.database_id != str(old["database_id"])
            or case.question != str(old["question"])
            or len(case.evidence) != int(old["evidence_chars"])
        ):
            mismatched.append(case_id)
    if mismatched:
        raise RuntimeError(f"frozen selected-case content mismatch: {mismatched[:10]}")

    payload = {
        "model": args.model,
        "sample_size": len(selected_ids),
        "source_sha256": source_hash,
        "baseline_source_sha256": EXPECTED_SOURCE_SHA256,
        "selected_case_identity": "db_id + question + evidence length matched v4 for all 100",
        "cases": [],
    }
    if args.resume and output.is_file():
        payload = json.loads(output.read_text(encoding="utf-8"))
    done = {str(row["case_id"]) for row in payload["cases"]}

    provider = OllamaProvider(model=args.model, timeout_s=900.0)
    catalogs: dict[str, tuple[Path, DatabaseCatalog]] = {}

    for index, case_id in enumerate(selected_ids, start=1):
        if case_id in done:
            continue
        case = case_by_id[case_id]
        label = label_by_id[case_id]
        if case.database_id not in catalogs:
            database = resolve_database(database_root, case.database_id)
            catalogs[case.database_id] = (database, DatabaseCatalog.from_sqlite(database))
        database, catalog = catalogs[case.database_id]
        schema_context, _ = select_schema_context(catalog, database_root, case.database_id, case.question)
        started = time.perf_counter()
        candidate = generate_direct_sql(
            case.question,
            schema_context,
            provider,
            external_evidence=case.evidence.strip() or None,
        )
        latency_s = time.perf_counter() - started
        execution_success, official_ex = ex_equal(database, candidate.sql, label.sql)
        payload["cases"].append({
            "case_id": case_id,
            "database_id": case.database_id,
            "sql": candidate.sql,
            "execution_success": execution_success,
            "official_ex": official_ex,
            "latency_s": latency_s,
        })
        payload["execution_success"] = sum(r["execution_success"] for r in payload["cases"]) / len(payload["cases"])
        payload["official_bird_ex"] = sum(r["official_ex"] for r in payload["cases"]) / len(payload["cases"])
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            f"[{index:03d}/{len(selected_ids)}] {case_id} exec={execution_success} ex={official_ex} "
            f"running_ex={payload['official_bird_ex']:.3f} latency={latency_s:.1f}s",
            flush=True,
        )

    print(json.dumps({
        "model": payload["model"],
        "n": len(payload["cases"]),
        "execution_success": payload["execution_success"],
        "official_bird_ex": payload["official_bird_ex"],
        "source_sha256": payload["source_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
