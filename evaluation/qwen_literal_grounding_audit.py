"""Inspect frozen Qwen3.5-9B SQL literal predicates against real SQLite values.

CPU-only, zero new model calls. Report observed evidence and possible format
alternatives, but DO NOT silently rewrite or deploy SQL. This is a deliberately
conservative parser for simple WHERE table.column = 'literal' predicates;
complex queries and failed probes remain unclassified. No benchmark gold SQL
is used to propose an alternative.

Usage:
  python -m evaluation.qwen_literal_grounding_audit --inputs old-shards/*.json \
      --database-root BIRD_ROOT --output qwen-literal-audit.json
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
from urllib.parse import quote

from evaluation.run_bird import _resolve_database

MODEL = "qwen3.5:9b-q4_K_M"
MODEL_DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
SOURCE_SHA = "0310b7ca2dfb6247234e2073f93e7a75106ee41ba55231f75d06d2ef68efe2ec"
QUALIFIED_EQ = re.compile(
    r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\s*=\s*'((?:''|[^'])+)'",
    flags=re.I,
)
RELATION = re.compile(
    r"\b(?:FROM|JOIN)\s+([A-Za-z_]\w*)"
    r"(?:\s+(?:AS\s+)?(?!WHERE\b|ON\b|JOIN\b|INNER\b|LEFT\b|RIGHT\b|"
    r"FULL\b|GROUP\b|ORDER\b|LIMIT\b|HAVING\b|CROSS\b|OUTER\b)"
    r"([A-Za-z_]\w*))?",
    flags=re.I,
)
WHERE = re.compile(r"\bWHERE\b(.*)", flags=re.I | re.S)
TRAILING = re.compile(r"\b(?:GROUP\s+BY|ORDER\s+BY|LIMIT|HAVING)\b", flags=re.I)
SQL_KEYWORDS = {
    "WHERE", "ON", "JOIN", "INNER", "LEFT", "RIGHT", "FULL",
    "GROUP", "ORDER", "LIMIT", "HAVING", "CROSS", "OUTER",
}


@dataclass(frozen=True)
class LiteralPredicate:
    table: str
    column: str
    value: str


@dataclass(frozen=True)
class LiteralProbe:
    status: str
    positive_count: int = 0
    replacement: str | None = None
    operator: str | None = None
    interrupted: bool = False


def _ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def extract_simple_equalities(
    sql: str, physical_tables: dict[str, tuple[str, ...]],
) -> tuple[LiteralPredicate, ...]:
    """Only qualified, direct equals within simple WHERE and unambiguous aliases."""
    tables = {t.casefold(): t for t in physical_tables}
    aliases: dict[str, str] = {}
    conflicted: set[str] = set()
    for match in RELATION.finditer(sql):
        table = tables.get(match.group(1).casefold())
        if table is None:
            continue
        alias = match.group(2)
        name = alias if alias and alias.upper() not in SQL_KEYWORDS else table
        for key in {table.casefold(), name.casefold()}:
            if key in aliases and aliases[key] != table:
                conflicted.add(key)
            aliases[key] = table
    where = WHERE.search(sql)
    if not where:
        return ()
    fragment = TRAILING.split(where.group(1), maxsplit=1)[0]
    # OR introduces alternative semantics or a nested query. Stay conservative.
    if re.search(r"\bOR\b", fragment, flags=re.I):
        return ()
    result: list[LiteralPredicate] = []
    for hit in QUALIFIED_EQ.finditer(fragment):
        alias, name, value = hit.groups()
        if alias.casefold() in conflicted:
            continue
        table = aliases.get(alias.casefold())
        if not table:
            continue
        column = next((c for c in physical_tables[table] if c.casefold() == name.casefold()), None)
        if column is None:
            continue
        # Exclude SQL expressions like LOWER(t.name) = 'A'.
        before = fragment[max(0, hit.start() - 32):hit.start()]
        if re.search(r"\b(?:LOWER|UPPER|TRIM|CAST|COALESCE)\s*\(\s*$", before, re.I):
            continue
        clean = value.replace("''", "'")
        if len(clean) < 3 or clean.isdecimal():
            continue
        pred = LiteralPredicate(table, column, clean)
        if pred not in result:
            result.append(pred)
    return tuple(result)


def _year_first_date(value: str) -> str | None:
    if not re.fullmatch(r"\d{4}[/-]\d{1,2}[/-]\d{1,2}", value):
        return None
    for pattern in ("%Y/%m/%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, pattern).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


def _minute_second_prefix(value: str) -> str | None:
    # Only transform explicitly hour:minute:second, zero-hour timestamps
    # into the well-known SQLite BIRD M:SS.mmm representation.
    found = re.fullmatch(r"0:(\d{1,2}):([0-5]\d)", value)
    if not found:
        return None
    return f"{int(found.group(1))}:{found.group(2)}.%"


def check_literal(database: str | Path, pred: LiteralPredicate) -> LiteralProbe:
    """Only positive, bounded read-only probes. No conclusion from interruption."""
    uri = "file:" + quote(str(Path(database).resolve())) + "?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True, timeout=5) as db:
            db.execute("PRAGMA query_only=ON")
            # Validate identifier existence even for direct use of check_literal.
            columns = db.execute(f"PRAGMA table_info({_ident(pred.table)})").fetchall()
            if pred.column not in {row[1] for row in columns}:
                return LiteralProbe("unsupported_column")
            ticks = [0]
            def watchdog() -> int:
                ticks[0] += 1
                return int(ticks[0] > 150)
            db.set_progress_handler(watchdog, 1000)
            try:
                sql = (f"SELECT 1 FROM {_ident(pred.table)} "
                       f"WHERE {_ident(pred.column)} = ? LIMIT 1")
                if db.execute(sql, (pred.value,)).fetchone():
                    return LiteralProbe("exact_present", positive_count=1)
                normalized = _year_first_date(pred.value)
                if normalized and normalized != pred.value:
                    if db.execute(sql, (normalized,)).fetchone():
                        return LiteralProbe(
                            "date_format_match", positive_count=1,
                            replacement=normalized, operator="=",
                        )
                time_prefix = _minute_second_prefix(pred.value)
                if time_prefix:
                    cursor = db.execute(
                        f"SELECT DISTINCT {_ident(pred.column)} "
                        f"FROM {_ident(pred.table)} "
                        f"WHERE {_ident(pred.column)} LIKE ? LIMIT 65",
                        (time_prefix,),
                    )
                    values = cursor.fetchall()
                    if values and len(values) <= 64:
                        return LiteralProbe(
                            "time_precision_match", positive_count=len(values),
                            replacement=time_prefix, operator="LIKE",
                        )
                cursor = db.execute(
                    f"SELECT DISTINCT {_ident(pred.column)} "
                    f"FROM {_ident(pred.table)} "
                    f"WHERE {_ident(pred.column)} = ? COLLATE NOCASE LIMIT 2",
                    (pred.value,),
                )
                values = [row[0] for row in cursor.fetchall()]
                if len(values) == 1 and str(values[0]) != pred.value:
                    return LiteralProbe(
                        "casefold_match", positive_count=1,
                        replacement=str(values[0]), operator="=",
                    )
                return LiteralProbe("no_positive_evidence")
            except sqlite3.OperationalError as exc:
                if "interrupted" in str(exc).casefold():
                    return LiteralProbe("probe_interrupted", interrupted=True)
                return LiteralProbe("probe_error")
            finally:
                db.set_progress_handler(None, 0)
    except (sqlite3.Error, OSError, ValueError):
        return LiteralProbe("probe_error")


def load_frozen_qwen_shards(paths: list[Path]) -> list[dict]:
    if len(paths) != 4:
        raise ValueError("expected exactly four prior Qwen9B inference artifacts")
    cases: list[dict] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (payload.get("sample_size") != 25 or len(payload.get("cases", [])) != 25
            or payload.get("model") != MODEL or payload.get("model_digest") != MODEL_DIGEST
            or payload.get("source_sha256") != SOURCE_SHA or payload.get("think") is not False
            or payload.get("status") not in ("complete", "invalid")):
            raise ValueError(f"Qwen9B shard provenance mismatch {path.name}")
        cases.extend(payload["cases"])
    if len(cases) != 100 or len({str(c["case_id"]) for c in cases}) != 100:
        raise ValueError("missing or duplicated baseline Qwen cases")
    return sorted(cases, key=lambda x: int(x["case_id"]))


def audit(paths: list[Path], dbroot: Path) -> dict:
    cases = load_frozen_qwen_shards(paths)
    cache: dict[str, tuple[Path, dict[str, tuple[str, ...]]]] = {}
    records = []
    for case in cases:
        dbid = str(case["database_id"])
        if dbid not in cache:
            path = _resolve_database(dbroot, dbid)
            with sqlite3.connect("file:" + quote(str(path.resolve())) + "?mode=ro", uri=True) as db:
                schema = {}
                for (name,) in db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                ):
                    schema[name] = tuple(c[1] for c in db.execute(f"PRAGMA table_info({_ident(name)})"))
            cache[dbid] = (path, schema)
        dbpath, tables = cache[dbid]
        guarded = case["guarded"]
        sql = str(guarded.get("final_sql") or "")
        predicates = extract_simple_equalities(sql, tables)
        reports = []
        if guarded.get("execution_success"):
            for pred in predicates:
                r = check_literal(dbpath, pred)
                reports.append({**asdict(pred), **asdict(r)})
        records.append({
            "case_id": str(case["case_id"]), "db_id": dbid,
            "baseline_official_ex": bool(guarded.get("official_ex")),
            "baseline_execution_success": bool(guarded.get("execution_success")),
            "candidate_sql_sha256": sha256(sql.encode()).hexdigest(),
            "extractable_simple_predicates": len(predicates),
            "probes": reports,
        })
    wrong_exec = [x for x in records if x["baseline_execution_success"]
                  and not x["baseline_official_ex"]]
    positive_types = {"time_precision_match", "date_format_match", "casefold_match"}
    def count_issues(items: list[dict]) -> dict:
        return {
            "cases": len(items),
            "with_extractable_equalities": sum(x["extractable_simple_predicates"]>0 for x in items),
            "with_probe_results": sum(bool(x["probes"]) for x in items),
            "with_physical_exact_present": sum(any(p["status"]=="exact_present" for p in x["probes"]) for x in items),
            "with_positive_representation_candidate": sum(any(p["status"] in positive_types for p in x["probes"]) for x in items),
            "with_no_positive_value_evidence": sum(any(p["status"]=="no_positive_evidence" for p in x["probes"]) for x in items),
            "with_probe_unknown": sum(any(p["status"] in {"probe_error","probe_interrupted"} for p in x["probes"]) for x in items),
        }
    return {
        "protocol": "qwen9b-sql-conditioned-literal-value-gate-v1",
        "input_qwen": MODEL, "model_digest": MODEL_DIGEST,
        "no_new_qwen_inference": True, "no_sql_training": True,
        "baseline_100": count_issues(records),
        "wrong_but_executable": count_issues(wrong_exec),
        "candidate_status_counts": dict(Counter(
            p["status"] for r in records for p in r["probes"]
        )),
        "cases": records,
        "claim_boundary": (
            "Historic pinned public BIRD DEV100; these Qwen answers were already evaluated. "
            "Only simple qualified WHERE equals are parsed. One frozen source/SQLite "
            "version; not an independent model holdout. Positive results are only "
            "observed SQLite evidence, not validated semantic SQL repairs. "
            "A missing value is NOT proof the SQL is wrong."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, nargs="+", required=True)
    p.add_argument("--database-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("will not overwrite an existing report")
    result = audit(args.inputs, args.database_root)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print("QWEN_LITERAL_GROUNDING_AUDIT",
          json.dumps({key:value for key,value in result.items() if key != "cases"},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
