"""BIRD filtered TRAIN -> schema-grounded, semantic SQL supervised fine-tuning data.

No BIRD DEV SQL, dev labels, prior frozen100 oracle results, or model answers are
allowed here. Exact SQL remains a syntactic SFT target; accuracy MUST be judged
separately by execution result equivalence on held-out databases.

Inputs:
  birdsql/bird23-train-filtered: data/train-00000-of-00001.jsonl
  Deema/BIRD-SQL: train_tables.json (schema-only BIRD train metadata)
Both are non-GAIA, BIRD TRAIN-only. Use dataset terms and attribute authors.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import re
from typing import Any


def quote(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("invalid SQL identifier")
    return '"' + name.replace('"', '""') + '"'


def render_bird_schema(meta: dict) -> str:
    """Render original table/column names, PKs, and FK relationships; no content leakage."""
    names = meta["table_names_original"]
    columns = meta["column_names_original"]
    types = meta.get("column_types") or ["text"] * len(columns)
    if len(columns) != len(types) or not isinstance(names, list) or not names:
        raise ValueError("invalid BIRD table metadata")
    pk = meta.get("primary_keys") or []
    pk_indices: set[int] = set()
    for value in pk:
        if isinstance(value, int):
            pk_indices.add(value)
        elif isinstance(value, (list, tuple)):
            pk_indices.update(x for x in value if isinstance(x, int))
        elif isinstance(value, dict) and isinstance(value.get("column_id"), int):
            pk_indices.add(value["column_id"])
    fks = meta.get("foreign_keys") or []
    by_table: dict[int, list[tuple[int, str, str]]] = defaultdict(list)
    for i, pair in enumerate(columns):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("bad column index record")
        table_idx, col = pair
        if not isinstance(table_idx, int) or table_idx == -1:
            continue
        if not 0 <= table_idx < len(names) or not isinstance(col, str):
            raise ValueError("bad column reference")
        kind = str(types[i]).casefold()
        sql_kind = "INTEGER" if kind in {"number", "integer", "int", "boolean"} else "REAL" if kind in {"real", "float"} else "TEXT"
        by_table[table_idx].append((i, col, sql_kind))
    blocks = []
    for t, table_name in enumerate(names):
        cols = by_table.get(t, [])
        if not cols:
            raise ValueError("schema contains a table without columns")
        fields = [f"  {quote(col)} {type_name}" for _, col, type_name in cols]
        owned_pk = [quote(col) for idx, col, _ in cols if idx in pk_indices]
        if owned_pk:
            fields.append("  PRIMARY KEY (" + ", ".join(owned_pk) + ")")
        for fk in fks:
            if not isinstance(fk, (list, tuple)) or len(fk) != 2:
                raise ValueError("invalid foreign-key metadata")
            src, dst = fk
            if not isinstance(src, int) or not isinstance(dst, int) or not 0 <= src < len(columns) or not 0 <= dst < len(columns):
                raise ValueError("invalid foreign-key index")
            src_table, src_col = columns[src]
            dst_table, dst_col = columns[dst]
            if src_table == t and isinstance(dst_table, int) and 0 <= dst_table < len(names):
                fields.append(
                    f"  FOREIGN KEY ({quote(src_col)}) REFERENCES {quote(names[dst_table])} ({quote(dst_col)})"
                )
        blocks.append(f"CREATE TABLE {quote(table_name)} (\n" + ",\n".join(fields) + "\n);")
    return "\n\n".join(blocks)


def sql_is_readonly(sql: str) -> bool:
    """Reject all DML, DDL, multi-statements, SQL parser errors and non-SELECT queries."""
    if not isinstance(sql, str) or not sql.strip():
        return False
    try:
        import sqlglot
        from sqlglot import exp
        asts = sqlglot.parse(sql, read="sqlite")
    except (ValueError, TypeError) as exc:
        return False
    except Exception as exc:
        # SQLglot ParseError and missing parser libs both fail closed.
        if isinstance(exc, ImportError):
            raise RuntimeError("sqlglot is required for semantic SFT source verification") from exc
        return False
    if len(asts) != 1 or asts[0] is None:
        return False
    ast = asts[0]
    if not isinstance(ast, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        return False
    forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Alter, exp.Command)
    return not any(isinstance(node, forbidden) for node in ast.walk())


def make_semantic_prompt(question: str, evidence: str, schema_ddl: str) -> str:
    if not question.strip() or not schema_ddl.strip():
        raise ValueError("empty question or grounded schema")
    return (
        "You are an expert SQLite query writer. Produce one single read-only SQL query "
        "that answers the user's question exactly. Use only the provided schema, "
        "respect joins and primary/foreign keys, match filters and literal values, "
        "and use grouping, aggregation, sorting and LIMIT only when justified. "
        "Return SQL only, no markdown and no explanation.\n\n"
        "[DATABASE_SCHEMA]\n" + schema_ddl +
        "\n[/DATABASE_SCHEMA]\n\n[QUESTION]\n" + question.strip() +
        "\n[/QUESTION]\n\n[EVIDENCE]\n" + (evidence or "").strip() +
        "\n[/EVIDENCE]\n\nSQL:\n"
    )


def prepare_semantic_sft_rows(
    rows: list[dict],
    schema_rows: list[dict],
    *,
    seed: str = "semantic-sft-db-split-v1",
    validation_db_fraction: float = 0.15,
    dev_db_ids: set[str] | None = None,
    min_train: int = 1000,
    min_val: int = 100,
    max_schema_chars: int = 24000,
) -> tuple[list[dict], list[dict], dict]:
    """Deterministic database-disjoint split, fail closed on silent data attrition."""
    if not 0 < validation_db_fraction < 0.5:
        raise ValueError("validation fraction must be between 0 and 0.5")
    schema_by_db = {}
    for schema in schema_rows:
        db_id = str(schema.get("db_id") or "")
        if not db_id or db_id in schema_by_db:
            raise ValueError("duplicate/invalid database schema id")
        schema_by_db[db_id] = render_bird_schema(schema)
    dev_db_ids = set(dev_db_ids or ())
    if dev_db_ids.intersection(schema_by_db):
        raise ValueError("train/dev database overlap forbidden")
    kept: list[dict] = []
    rejected = Counter()
    seen = set()
    for row in rows:
        db_id = str(row.get("db_id") or "")
        question = str(row.get("question") or "").strip()
        evidence = str(row.get("evidence") or "")
        gold_sql = str(row.get("SQL") or "").strip()
        if not db_id or db_id not in schema_by_db or not question or not gold_sql:
            rejected["missing_schema_or_question"] += 1
            continue
        schema_ddl = schema_by_db[db_id]
        if len(schema_ddl) > max_schema_chars:
            rejected["schema_exceeds_budget"] += 1
            continue
        if not sql_is_readonly(gold_sql):
            rejected["non_readonly_or_unparseable_sql"] += 1
            continue
        digest = sha256((db_id + "\0" + " ".join(question.lower().split())).encode()).hexdigest()
        if digest in seen:
            rejected["duplicate_question_in_db"] += 1
            continue
        seen.add(digest)
        prompt = make_semantic_prompt(question, evidence, schema_ddl)
        kept.append({
            "db_id": db_id, "question_hash": digest, "prompt": prompt,
            "schema_ddl": schema_ddl, "target_sql": gold_sql,
            "question": question, "evidence": evidence,
        })
    db_ids = sorted({item["db_id"] for item in kept}, key=lambda x: sha256(f"{seed}:{x}".encode()).hexdigest())
    if len(db_ids) < 2:
        raise ValueError("need multiple databases to create independent held-out split")
    val_count = min(len(db_ids)-1, max(1, round(len(db_ids) * validation_db_fraction)))
    heldout_dbs = set(db_ids[:val_count])
    train = [item for item in kept if item["db_id"] not in heldout_dbs]
    val = [item for item in kept if item["db_id"] in heldout_dbs]
    if len(train) < min_train or len(val) < min_val:
        raise ValueError(f"insufficient database-disjoint data: train={len(train)} val={len(val)} expected >= {min_train}/{min_val}")
    # Randomize within each partition without mixing database identity.
    random.Random(seed + ":train").shuffle(train)
    random.Random(seed + ":val").shuffle(val)
    report = {
        "schema": "semantic-sql-bird-filtered-train-only/v1",
        "source": "BIRD filtered TRAIN ONLY",
        "input": len(rows), "accepted": len(kept),
        "train": len(train), "validation": len(val),
        "train_dbs": sorted({r["db_id"] for r in train}),
        "validation_dbs": sorted(heldout_dbs),
        "excluded_dev_dbs": sorted(dev_db_ids),
        "rejected": dict(sorted(rejected.items())),
        "seed": seed, "max_schema_chars": max_schema_chars,
        "caveat": "Dataset-filtered SQL may still be semantically wrong; this step verifies syntax, schema identities and split, not gold execution."
    }
    return train, val, report


def tokenize_semantic_pair(tokenizer, row: dict, *, max_length: int = 3072) -> dict:
    """Supervise ONLY correct SQL completion tokens; never truncate any schema."""
    prompt = tokenizer.encode(row["prompt"], add_special_tokens=False)
    answer = tokenizer.encode(row["target_sql"] + (tokenizer.eos_token or ""), add_special_tokens=False)
    if len(prompt) + len(answer) > max_length:
        raise ValueError(f"too_long: {len(prompt)+len(answer)} > {max_length}")
    if not prompt or not answer:
        raise ValueError("empty tokenized prompt or target")
    ids = prompt + answer
    return {
        "input_ids": ids, "attention_mask": [1] * len(ids),
        "labels": [-100] * len(prompt) + answer,
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filtered-train", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--dev-tables", type=Path, help="Optional BIRD DEV schemas for leakage checking")
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-train", type=int, default=1000)
    parser.add_argument("--min-val", type=int, default=100)
    args = parser.parse_args()
    for p in [args.train_output, args.validation_output, args.report]:
        if p.exists():
            raise FileExistsError(f"refuse to overwrite existing frozen data: {p}")
    data = [json.loads(line) for line in args.filtered_train.read_text(encoding="utf-8").splitlines() if line.strip()]
    schemas = json.loads(args.tables.read_text(encoding="utf-8"))
    dev = json.loads(args.dev_tables.read_text(encoding="utf-8")) if args.dev_tables else []
    train, val, report = prepare_semantic_sft_rows(
        data, schemas, dev_db_ids={str(item["db_id"]) for item in dev},
        min_train=args.min_train, min_val=args.min_val,
    )
    for path in [args.train_output, args.validation_output, args.report]:
        path.parent.mkdir(parents=True, exist_ok=True)
    for path, subset in [(args.train_output, train), (args.validation_output, val)]:
        path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in subset), encoding="utf-8")
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in {"train_dbs","validation_dbs","excluded_dev_dbs"}}, sort_keys=True))


if __name__ == "__main__":
    main()
