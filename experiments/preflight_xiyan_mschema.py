"""CPU-only preflight over every frozen BIRD question before model inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.bird_loader import load_bird_json
from evaluation.schema_context import _description_rows, select_schema_context
from semantic_sql.catalog import DatabaseCatalog
from experiments.xiyan_mschema_frozen100 import build_mschema, resolve_database


def preflight(*, source: Path, frozen: Path, database_root: Path) -> dict:
    manifest = json.loads(frozen.read_text(encoding="utf-8"))
    ids = [str(row["case_id"]) for row in manifest["cases"]]
    if len(ids) != 100 or len(set(ids)) != 100:
        raise ValueError("frozen manifest must have exactly 100 unique cases")
    cases, _ = load_bird_json(source)
    by_id = {str(row.case_id): row for row in cases}
    if set(ids) - set(by_id):
        raise ValueError("frozen cases missing from BIRD source")
    cache = {}
    checks = 0
    for case_id in ids:
        row = by_id[case_id]
        if row.database_id not in cache:
            path = resolve_database(database_root, row.database_id)
            catalog = DatabaseCatalog.from_sqlite(path)
            descriptions = {(t, c): d for t, c, d in _description_rows(database_root, row.database_id)}
            cache[row.database_id] = (catalog, descriptions)
        catalog, descriptions = cache[row.database_id]
        _, lexical = select_schema_context(catalog, database_root, row.database_id, row.question)
        context = build_mschema(
            catalog=catalog, database_id=row.database_id,
            selected_tables=lexical["selected_tables"], descriptions=descriptions,
        )
        if "Table:" not in context:
            raise ValueError("M-Schema missing all tables")
        checks += 1
    return {"status": "success", "frozen_cases": checks, "databases": len(cache)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(preflight(source=args.source, frozen=args.frozen, database_root=args.database_root)))


if __name__ == "__main__":
    main()
