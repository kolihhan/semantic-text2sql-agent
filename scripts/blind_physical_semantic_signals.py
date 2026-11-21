"""Blind physical evidence screen for original Qwen3.5-9B BIRD DEV SQL.

Three separation stages: prepare strips labels, detect sees only SQL/question/schema,
score joins sealed labels AFTER detections freeze. No model calls or SQL modification.

All reported signals are *suspicions*, not verified semantic errors.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
from typing import Any

SOURCE_SHA = "0310b7ca2dfb6247234e2073f93e7a75106ee41ba55231f75d06d2ef68efe2ec"
MODEL_DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
SHARD_STATUS = {"0": "complete", "1": "complete", "2": "complete", "3": "invalid"}
N = 100
# String-encoded signals are preregistered here; do not revise after opening labels.
SIGNALS = (
    "VALUE_CASE_ONLY_PHYSICAL_MATCH",
    "VALUE_ABSENT_FROM_COLUMN",
    "EXPLICIT_UNIQUE_COUNT_WITHOUT_DISTINCT",
    "RATIO_WITHOUT_COMPUTATION",
    "EXPLICIT_TOPK_WITHOUT_LIMIT",
    "QUESTION_QUOTED_LITERAL_ABSENT_FROM_SQL",
    "DECLARED_FK_JOIN_KEY_MISMATCH",
)
SOURCE = "original_qwen35_9b_frozen100_exposed_DEV"
_SQL_EQ = re.compile(
    r"""(?<![\w])(?:(?P<alias>"?[A-Za-z_]\w*"?)\s*\.\s*)?(?P<column>"?[A-Za-z_]\w*"?)\s*=\s*'(?P<value>(?:''|[^'])*)'""",
    flags=re.I,
)
_TABLE = re.compile(
    r"""(?:\bFROM\b|\bJOIN\b)\s+(?P<table>"?[A-Za-z_]\w*"?)\s*(?:(?:AS\s+)?(?P<alias>"?[A-Za-z_]\w*"?))?""",
    re.I,
)
_SQL_LITERAL = re.compile(r"'((?:''|[^'])*)'")
_QUESTION_LITERAL = re.compile(r"""(?:"([^"]{3,90})"|'([^']{3,90})')""")
_RESERVED = {"where", "join", "on", "left", "right", "inner", "outer", "group", "order", "limit", "having", "cross", "full", "union", "as", "by"}
_FK_COMPARE = re.compile(r"""(?P<a>[A-Za-z_]\w*)\s*\.\s*(?P<ac>[A-Za-z_]\w*)\s*=\s*(?P<b>[A-Za-z_]\w*)\s*\.\s*(?P<bc>[A-Za-z_]\w*)""", re.I)


def _write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf8")


def _load(path: Path) -> Any:
    return json.loads(path.read_text("utf8"))


def _canonical_json_hash(payload: Any) -> str:
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def prepare(shard_dir: Path, output: Path, labels: Path) -> None:
    candidates = []
    sealed = []
    shard_statuses = {}
    for shard in range(4):
        path = shard_dir / f"qwen9b-shard-{shard}.json"
        d = _load(path)
        if d.get("status") != SHARD_STATUS[str(shard)] or len(d.get("cases", [])) != 25:
            raise ValueError("original shard status/count drift")
        if d.get("source_sha256") != SOURCE_SHA or d.get("model_digest") != MODEL_DIGEST:
            raise ValueError("original source or model identity mismatch")
        if str(d.get("model")) != "qwen3.5:9b-q4_K_M":
            raise ValueError("original model identity drift")
        shard_statuses[str(shard)] = d["status"]
        for row in d["cases"]:
            case_id = str(row["case_id"])
            candidate = row.get("guarded", {}).get("final_sql")
            if not candidate and row.get("guarded", {}).get("initial_sql"):
                candidate = row["guarded"]["initial_sql"]
            if not candidate:
                candidate = ""
            candidates.append({
                "case_id": case_id, "db_id": row["database_id"], "question": row["question"],
                "selected_sql": candidate, "shard": shard,
            })
            sealed.append({
                "case_id": case_id, "shard": shard, "db_id": row["database_id"],
                "shard_status": d["status"], "execution_success": bool(row["guarded"]["execution_success"]),
                "strict_ex": bool(row["guarded"]["official_ex"]),
                "direct_strict_ex": bool(row["direct"]["official_ex"]),
            })
    if len(candidates) != N or len({x["case_id"] for x in candidates}) != N:
        raise ValueError("historical candidate set incomplete or repeated")
    if len({x["db_id"] for x in candidates}) != 11:
        raise ValueError("historical database coverage mismatch")
    src = {"study": SOURCE, "status": "frozen_unlabeled", "n": N, "source_sha256": SOURCE_SHA,
           "model_digest": MODEL_DIGEST, "cases": sorted(candidates, key=lambda x: int(x["case_id"]))}
    labs = {"study": SOURCE, "status": "sealed_original_scores", "n": N, "source_sha256": SOURCE_SHA,
            "labels": sorted(sealed, key=lambda x: int(x["case_id"])), "original_shards": shard_statuses}
    _write(output, src)
    _write(labels, labs)
    print("PREPARE_FROZEN", json.dumps({
        "candidate_sha256": _canonical_json_hash(src), "n": N,
        "labeled_separately": True, "invalid_shards": ["3"],
    }), flush=True)


def _identifier(name: str) -> str:
    return name.strip('"').lower()


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _aliases(sql: str, actual_tables: dict[str, str]) -> dict[str, str]:
    found = {}
    for m in _TABLE.finditer(sql):
        name = _identifier(m.group("table"))
        if name not in actual_tables:
            continue
        alias = _identifier(m.group("alias") or name)
        if alias in _RESERVED:
            alias = name
        found[alias] = name
        found[name] = name
    return found


def _schema(conn: sqlite3.Connection) -> tuple[dict[str, str], dict[str, dict[str, str]], set[tuple[str, str, str, str]]]:
    tables = {str(x[0]).lower(): str(x[0]) for x in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
    columns = {}
    fks = set()
    for canonical, original in tables.items():
        columns[canonical] = {str(x[1]).lower(): str(x[1]) for x in conn.execute(f"PRAGMA table_info({_quote(original)})")}
        for row in conn.execute(f"PRAGMA foreign_key_list({_quote(original)})"):
            # pragma columns: id,seq,table,from,to,...
            target = str(row[2]).lower()
            fks.add((canonical, str(row[3]).lower(), target, str(row[4]).lower()))
    return tables, columns, fks


def _exists(conn: sqlite3.Connection, table: str, column: str, value: str, *, nocase: bool) -> bool | None:
    clause = " COLLATE NOCASE" if nocase else " COLLATE BINARY"
    query = f"SELECT 1 FROM {_quote(table)} WHERE {_quote(column)}{clause} = ? LIMIT 1"
    # Safety budget for each read-only probe: max 2M VM instructions; no SQL from model executed.
    conn.set_progress_handler(lambda: 1, 2000)
    try:
        # Abort handler above intentionally too tight? Run with normal bounded VM counter below.
        count = [0]
        def limited() -> int:
            count[0] += 1
            return 1 if count[0] > 2000 else 0
        conn.set_progress_handler(limited, 1000)
        return conn.execute(query, (value,)).fetchone() is not None
    except sqlite3.Error:
        return None
    finally:
        conn.set_progress_handler(None, 0)


def detect_one(question: str, sql: str, conn: sqlite3.Connection | None) -> dict[str, Any]:
    names = []
    observations = []
    skips = []
    normalized_question = question.casefold()
    normalized_sql = sql.casefold()
    def add(flag: str, why: str, *, strength: str = "HEURISTIC") -> None:
        if flag not in names:
            names.append(flag)
            observations.append({"signal": flag, "strength": strength, "explanation": why})

    if re.search(r"\b(?:distinct|unique|different)\s+(?:\w+\s+){0,3}\b", normalized_question) and re.search(r"\bCOUNT\s*\(\s*(?!DISTINCT\b)", sql, re.I):
        add("EXPLICIT_UNIQUE_COUNT_WITHOUT_DISTINCT", "question mentions uniqueness and SQL COUNT lacks DISTINCT")
    if re.search(r"\b(?:percentage|percent|proportion|ratio|rate)\b", normalized_question):
        if not re.search(r"[/]|\bAVG\s*\(|\bPERCENT\b|\bROUND\s*\(\s*\d", sql, re.I):
            add("RATIO_WITHOUT_COMPUTATION", "question requests ratio/rate but no recognized division/aggregate")
    if re.search(r"\btop\s+\d+\b", normalized_question) and not re.search(r"\bLIMIT\b|\bROW_NUMBER\b|\bRANK\b", sql, re.I):
        add("EXPLICIT_TOPK_WITHOUT_LIMIT", "explicit top-N question lacks recognized limiting/ranking")
    required = {
        (a or b).strip().casefold() for a, b in _QUESTION_LITERAL.findall(question)
        if len((a or b).strip()) >= 3
    }
    quoted = {v.replace("''", "'").strip().casefold() for v in _SQL_LITERAL.findall(sql)}
    if required and not required.intersection(quoted):
        add("QUESTION_QUOTED_LITERAL_ABSENT_FROM_SQL", "quoted question literal not found verbatim in selected SQL")
    if conn is None:
        skips.append("PHYSICAL_DB_UNAVAILABLE")
        return {"flags": names, "observations": observations, "abstentions": skips}

    tables, cols, fks = _schema(conn)
    aliases = _aliases(sql, tables)
    # This is a targeted SQL-operand probe, not a general SQL AST parser.
    for match in list(_SQL_EQ.finditer(sql))[:12]:
        value = match.group("value").replace("''", "'")
        if len(value) < 2 or len(value) > 100:
            continue
        qualifier = _identifier(match.group("alias") or "")
        column = _identifier(match.group("column"))
        if qualifier:
            table_id = aliases.get(qualifier)
        else:
            matching = [k for k in set(aliases.values()) if column in cols.get(k, {})]
            table_id = matching[0] if len(matching) == 1 else None
        if not table_id or column not in cols.get(table_id, {}):
            skips.append("EQ_OPERAND_UNRESOLVED")
            continue
        table_name, column_name = tables[table_id], cols[table_id][column]
        exact = _exists(conn, table_name, column_name, value, nocase=False)
        casefold = _exists(conn, table_name, column_name, value, nocase=True)
        if exact is False and casefold is True:
            add("VALUE_CASE_ONLY_PHYSICAL_MATCH",
                f"exact value absent in {table_id}.{column}; case-insensitive physical match exists",
                strength="PHYSICAL_OBSERVATION")
        elif exact is False and casefold is False:
            add("VALUE_ABSENT_FROM_COLUMN",
                f"value absent in tested {table_id}.{column} (possible legitimate empty result)",
                strength="PHYSICAL_OBSERVATION")
        elif exact is None or casefold is None:
            skips.append("SQLITE_PROBE_TIMEOUT_OR_ERROR")
    # Only inspect two-table simple JOINs. Declared FKs may not cover valid join paths.
    referenced = set(aliases.values())
    if len(referenced) == 2 and "join" in normalized_sql:
        a, b = sorted(referenced)
        between = {fk for fk in fks if {fk[0], fk[2]} == {a, b}}
        if between:
            seen_pairs = set()
            for m in _FK_COMPARE.finditer(sql):
                left, right = aliases.get(m.group("a").lower()), aliases.get(m.group("b").lower())
                if left and right:
                    seen_pairs.add((left, m.group("ac").lower(), right, m.group("bc").lower()))
            if seen_pairs and not any((x in seen_pairs or (x[2], x[3], x[0], x[1]) in seen_pairs) for x in between):
                add("DECLARED_FK_JOIN_KEY_MISMATCH",
                    "two joined tables have declared FK, but qualified equality does not use it")
        else:
            skips.append("NO_DECLARED_FK_FOR_JOIN")
    return {"flags": sorted(names), "observations": observations, "abstentions": sorted(set(skips))}


def _db_path(root: Path, db_id: str) -> Path | None:
    direct = root / db_id / f"{db_id}.sqlite"
    if direct.is_file():
        return direct
    found = list((root / db_id).glob("*.sqlite"))
    return found[0] if len(found) == 1 else None


def detect(source: Path, db_root: Path, output: Path) -> None:
    src = _load(source)
    if src.get("study") != SOURCE or src.get("status") != "frozen_unlabeled" or len(src.get("cases", [])) != N:
        raise ValueError("unlabeled source incomplete")
    # Fail closed if any score, EX, gold or label field leaks into candidate payload.
    for row in src["cases"]:
        if set(row) != {"case_id", "db_id", "question", "selected_sql", "shard"}:
            raise ValueError("unexpected field in unlabeled candidate")
    records = []
    connections: dict[str, sqlite3.Connection] = {}
    try:
        for row in src["cases"]:
            dbid = row["db_id"]
            if dbid not in connections:
                path = _db_path(db_root, dbid)
                if path is None:
                    raise FileNotFoundError(f"database missing: {dbid}")
                conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
                conn.execute("PRAGMA query_only=ON")
                connections[dbid] = conn
            detection = detect_one(row["question"], row["selected_sql"], connections[dbid])
            records.append({"case_id": row["case_id"], "db_id": dbid, "shard": row["shard"], **detection})
    finally:
        for conn in connections.values():
            conn.close()
    sealed = {
        "study": SOURCE, "status": "signals_sealed_no_labels",
        "candidate_sha256": _canonical_json_hash(src), "signals_version": "physical_schema_value_v1",
        "signal_names": list(SIGNALS), "n": len(records), "cases": records,
    }
    _write(output, sealed)
    print("DETECTION_SEALED", json.dumps({
        "sha256": _canonical_json_hash(sealed), "n": len(records), "signals": dict(Counter(
            s for row in records for s in row["flags"])),
        "with_any_signal": sum(bool(row["flags"]) for row in records),
    }), flush=True)


def _wilson(k: int, n: int) -> list[float] | None:
    if not n:
        return None
    from math import sqrt
    z = 1.96
    d = 1 + z*z/n
    center = (k/n + z*z/(2*n)) / d
    half = z*sqrt(k*(n-k)/(n*n*n) + z*z/(4*n*n))/d
    return [round(max(0, center-half), 4), round(min(1, center+half), 4)]


def _metrics(rows: list[dict], flag: str | None = None) -> dict:
    eligible = [r for r in rows if r["execution_success"]]
    wrong = [r for r in eligible if not r["strict_ex"]]
    correct = [r for r in eligible if r["strict_ex"]]
    hits = [r for r in eligible if flag in r["flags"]] if flag else [r for r in eligible if r["flags"]]
    hits_bad = [r for r in hits if not r["strict_ex"]]
    hits_good = [r for r in hits if r["strict_ex"]]
    return {
        "n_executable": len(eligible), "n_wrong_executable": len(wrong), "n_correct_executable": len(correct),
        "flagged_n": len(hits), "flagged_wrong": len(hits_bad), "flagged_correct": len(hits_good),
        "error_enrichment_ppv": round(len(hits_bad)/len(hits), 4) if hits else None,
        "error_enrichment_ppv_wilson95": _wilson(len(hits_bad), len(hits)),
        "wrong_sql_coverage": round(len(hits_bad)/len(wrong), 4) if wrong else None,
        "right_sql_false_alarm_rate": round(len(hits_good)/len(correct), 4) if correct else None,
        "flagged_wrong_dbs": sorted({r["db_id"] for r in hits_bad}),
        "flagged_correct_dbs": sorted({r["db_id"] for r in hits_good}),
    }


def score(signals: Path, labels: Path, output: Path) -> None:
    frozen, sealed = _load(signals), _load(labels)
    if frozen.get("status") != "signals_sealed_no_labels" or sealed.get("status") != "sealed_original_scores":
        raise ValueError("staging not frozen")
    if frozen.get("study") != SOURCE or sealed.get("study") != SOURCE or frozen.get("n") != N or sealed.get("n") != N:
        raise ValueError("study identity mismatch")
    all_labels = {x["case_id"]: x for x in sealed["labels"]}
    if len(all_labels) != N or len(frozen["cases"]) != N:
        raise ValueError("labelled sample incomplete")
    joined = []
    for r in frozen["cases"]:
        key = r["case_id"]
        label = all_labels[key]
        if r["shard"] != label["shard"] or r["db_id"] != label["db_id"]:
            raise ValueError("case/shard/database mismatch")
        joined.append({**r, **label})
    primary = [x for x in joined if x["shard_status"] == "complete"]
    exploratory = [x for x in joined if x["shard_status"] != "complete"]
    if len(primary) != 75 or len(exploratory) != 25:
        raise ValueError("status stratification drift")
    m = _metrics(primary)
    by_signal = {name: _metrics(primary, name) for name in SIGNALS}
    # preregistered screening only for future independent audit, not live intervention.
    candidate_signals = [
        name for name, met in by_signal.items()
        if met["flagged_wrong"] >= 5 and len(met["flagged_wrong_dbs"]) >= 3
        and met["flagged_correct"] <= 2
        and met["error_enrichment_ppv"] is not None and met["error_enrichment_ppv"] >= .8
    ]
    report = {
        "study": SOURCE, "status": "HISTORICAL_EXPLORATORY_ONLY",
        "detector_sha256": _canonical_json_hash(frozen),
        "has_gold_sql_in_detector": False, "new_model_calls": 0,
        "no_production_change": True, "no_sql_rewrite": True, "no_holdout_used": True,
        "original_shard_statuses": sealed["original_shards"],
        "primary_original_complete_shards": {
            "n": len(primary), "n_total": N,
            "metrics_any_flag": m,
            "signals": by_signal,
        },
        "invalid_shard_separately": {"n": len(exploratory), "metrics_any_flag": _metrics(exploratory),
                                     "signals": {name: _metrics(exploratory, name) for name in SIGNALS}},
        "predeclared_future_candidate_signals": candidate_signals,
        "research_gate": "REQUEST_INDEPENDENT_MECHANISM_ADJUDICATION" if candidate_signals else "HOLD_NO_DEMONSTRATED_SIGNAL",
        "limitations": [
            "EX association is not a causal error-class label; no precision of semantic contradiction established.",
            "All original BIRD DEV physical databases are previously exposed; no source-disjoint validity.",
            "SQL operand regex intentionally abstains on complex/ambiguous expressions; measured coverage is partial.",
            "Casefold matches and FK paths can legitimately differ; signals are risks, not proofs of incorrect answers.",
            "No SQL candidate altered, executed as a treatment, or rescored for purported EX uplift.",
        ],
    }
    _write(output, report)
    print("BLIND_SIGNALS_SCORE", json.dumps({
        "status": report["status"], "gate": report["research_gate"],
        "n_primary": len(primary), "any_flag": m,
        "per_signal": {k: {"wrong":v["flagged_wrong"],"correct":v["flagged_correct"],
                            "ppv":v["error_enrichment_ppv"],"dbs":len(v["flagged_wrong_dbs"])}
                        for k,v in by_signal.items()},
    }), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    first = sub.add_parser("prepare")
    first.add_argument("--shard-dir", type=Path, required=True)
    first.add_argument("--unlabeled", type=Path, required=True)
    first.add_argument("--labels", type=Path, required=True)
    second = sub.add_parser("detect")
    second.add_argument("--unlabeled", type=Path, required=True)
    second.add_argument("--database-root", type=Path, required=True)
    second.add_argument("--output", type=Path, required=True)
    third = sub.add_parser("score")
    third.add_argument("--signals", type=Path, required=True)
    third.add_argument("--labels", type=Path, required=True)
    third.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.command == "prepare":
        prepare(args.shard_dir, args.unlabeled, args.labels)
    elif args.command == "detect":
        detect(args.unlabeled, args.database_root, args.output)
    elif args.command == "score":
        score(args.signals, args.labels, args.output)


if __name__ == "__main__":
    main()
