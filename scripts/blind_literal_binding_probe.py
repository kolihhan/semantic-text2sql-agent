"""Gold-blind physical value-binding adjudication, not an SQL auto-rewriter.

Input is the original frozen NO-LABEL candidate and detection artifacts.
Each normalized/alternative value is observed from the real SQLite DB in read-only
mode. No publisher SQL or EX outcome is available to this stage.

The only safe intended output is a diagnostic classification or ABSTAIN.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
import time

from scripts.blind_physical_semantic_signals import (
    SOURCE, N, _SQL_EQ, _aliases, _schema, _identifier, _quote,
)

STRONG_ORIGINAL_FLAGS = {"VALUE_ABSENT_FROM_COLUMN", "VALUE_CASE_ONLY_PHYSICAL_MATCH"}
# All source/SQL-only choices frozen before correctness labels become accessible.
CONTROL_N = 18
MAX_EQS = 12
MAX_COLUMNS_PER_CASE = 40
MAX_VM_CALLBACKS = 1200
QUERY_TIMEOUT_S = 1.5
MAX_TEXT_VALUE_LEN = 96


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def digest(obj):
    return sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def physical_probe(conn, table: str, column: str, predicate: str, literal: str):
    """Return (censored 0–2 witness values, status). No model SQL executes."""
    query = f"SELECT DISTINCT {_quote(column)} FROM {_quote(table)} WHERE {predicate} LIMIT 3"
    count = [0]
    started = time.monotonic()

    def limit():
        count[0] += 1
        return int(count[0] >= MAX_VM_CALLBACKS or (time.monotonic() - started) >= QUERY_TIMEOUT_S)

    conn.set_progress_handler(limit, 1000)
    try:
        values = [str(row[0]) for row in conn.execute(query, (literal,)).fetchall()
                  if row[0] is not None]
        return values[:3], "OK"
    except sqlite3.Error:
        return [], "TIMEOUT_OR_SQL_ERROR"
    finally:
        conn.set_progress_handler(None, 0)


def classify_operand(conn, tables, columns, db_table: str, column: str, literal: str, question: str, selected_tables: list[str]):
    literal = literal.replace("''", "'")
    table_name = tables[db_table]
    physical = columns[db_table][column]
    literal_support = literal.casefold() in question.casefold()
    evidence = {
        "bound_table": db_table, "bound_column": column, "literal_sha256": sha256(literal.encode()).hexdigest(),
        "literal_in_question_verbatim_ignore_case": literal_support,
        "status": "ABSTAIN", "alternative_column_count": 0,
        "possible_compatible_values": 0, "candidate_value_sha256": [],
        "reason": "UNKNOWN",
    }

    exact, ex_status = physical_probe(conn, table_name, physical, f"{_quote(physical)} COLLATE BINARY = ?", literal)
    if ex_status != "OK":
        evidence["reason"] = "PHYSICAL_PROBE_UNAVAILABLE"
        return evidence
    if exact:
        evidence["status"] = "ALREADY_EXACT"; evidence["reason"] = "VALUE_EXISTS_IN_BOUND_COLUMN"
        return evidence

    # Exact string absent; check same-column alternative only.
    normalized, norm_status = physical_probe(
        conn, table_name, physical,
        f"TRIM(CAST({_quote(physical)} AS TEXT)) COLLATE NOCASE = ?",
        literal.strip(),
    )
    if norm_status != "OK":
        evidence["reason"] = "PHYSICAL_PROBE_UNAVAILABLE"
        return evidence
    unique = {v for v in normalized}
    if len(unique) == 1:
        evidence.update(status="NORMALIZED_SAME_COLUMN", reason="EXACT_NORMALIZED_PHYSICAL_WITNESS",
                        possible_compatible_values=1,
                        candidate_value_sha256=[sha256(next(iter(unique)).encode()).hexdigest()])
        return evidence
    if len(unique) > 1:
        evidence["reason"] = "NON_UNIQUE_NORMALIZED_VALUE"
        evidence["possible_compatible_values"] = len(unique)
        return evidence

    # Prefix/substring is a high-risk hypothesis, *never* a replacement rule.
    if len(literal.strip()) >= 5:
        # Probe only value-inside-stored-string, not reverse; LIKE escapes wildcards.
        escaped = literal.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        substring, substatus = physical_probe(
            conn, table_name, physical,
            f"CAST({_quote(physical)} AS TEXT) LIKE ? ESCAPE '\\'",
            "%" + escaped + "%",
        )
        if substatus == "OK" and substring:
            evidence.update(status="SUBSTRING_AMBIGUOUS", reason="PARTIAL_PHYSICAL_MATCH_NOT_SYNONYM_PROOF",
                            possible_compatible_values=len(set(substring)))
            return evidence

    # Candidate elsewhere in selected SQL tables; not necessarily semantically correct.
    # Explicitly bounded, and never claim a column swap as correct without external proof.
    alt = []
    for candidate_table in selected_tables[:5]:
        for candidate_column in sorted(columns.get(candidate_table, {}))[:MAX_COLUMNS_PER_CASE]:
            if candidate_table == db_table and candidate_column == column:
                continue
            found, status = physical_probe(
                conn, tables[candidate_table], columns[candidate_table][candidate_column],
                f"CAST({_quote(columns[candidate_table][candidate_column])} AS TEXT) COLLATE BINARY = ?",
                literal,
            )
            if status == "OK" and found:
                alt.append((candidate_table, candidate_column))
                if len(alt) >= 3:
                    break
        if len(alt) >= 3:
            break
    evidence["alternative_column_count"] = len(alt)
    evidence["alternate_binding_columns"] = [
        f"{t}.{c}" for t, c in alt[:3]
    ]
    if len(alt) == 1:
        evidence["status"] = "POSSIBLE_COLUMN_MISBIND"
        evidence["reason"] = "UNIQUE_EXACT_VALUE_ON_OTHER_SELECTED_COLUMN_NOT_SEMANTIC_PROOF"
    elif len(alt) > 1:
        evidence["reason"] = "MULTIPLE_ALTERNATIVE_COLUMN_MATCHES"
    else:
        evidence["reason"] = "NO_PHYSICAL_ALTERNATIVE_FOUND"
    return evidence


def diagnose(case: dict, frozen_signal: dict, conn: sqlite3.Connection):
    sql, question = case["selected_sql"], case["question"]
    tables, cols, _fks = _schema(conn)
    aliases = _aliases(sql, tables)
    findings, skipped = [], []
    for m in list(_SQL_EQ.finditer(sql))[:MAX_EQS]:
        lit = m.group("value")
        if not (1 < len(lit) <= MAX_TEXT_VALUE_LEN):
            continue
        qualifier = _identifier(m.group("alias") or "")
        col = _identifier(m.group("column"))
        if qualifier:
            bound = aliases.get(qualifier)
        else:
            matched = [t for t in set(aliases.values()) if col in cols.get(t, {})]
            bound = matched[0] if len(matched) == 1 else None
        if bound is None or col not in cols.get(bound, {}):
            skipped.append("UNRESOLVED_SQL_OPERAND")
            continue
        findings.append(classify_operand(conn, tables, cols, bound, col, lit, question, sorted(set(aliases.values()))))
    return {
        "case_id": case["case_id"], "db_id": case["db_id"], "shard": case["shard"],
        "original_blind_flags": frozen_signal["flags"], "findings": findings,
        "abstentions": sorted(set(skipped)),
    }


def run_probe(candidates: Path, signals: Path, db_root: Path, output: Path):
    original, detect = load(candidates), load(signals)
    if original["status"] != "frozen_unlabeled" or original["study"] != SOURCE or len(original["cases"]) != N:
        raise ValueError("frozen unlabeled input source mismatch")
    if detect["status"] != "signals_sealed_no_labels" or detect["study"] != SOURCE or len(detect["cases"]) != N:
        raise ValueError("frozen detector artifact mismatch")
    if any(set(row) != {"case_id", "db_id", "question", "selected_sql", "shard"} for row in original["cases"]):
        raise ValueError("unexpected potentially labeled input")
    if any("strict_ex" in json.dumps(row) or "gold_sql" in json.dumps(row)
           for row in detect["cases"]):
        raise ValueError("detector contains forbidden outcome data")
    signal_by_id = {x["case_id"]: x for x in detect["cases"]}
    if len(signal_by_id) != N:
        raise ValueError("duplicate signal cases")
    selected = [x for x in original["cases"]
                if STRONG_ORIGINAL_FLAGS.intersection(signal_by_id[x["case_id"]]["flags"])]
    # source-first deterministic unflagged controls; all labels absent.
    unflagged = [x for x in original["cases"] if not signal_by_id[x["case_id"]]["flags"]]
    controls = sorted(unflagged, key=lambda x: sha256(("literal-control-v1:" + x["case_id"]).encode()).hexdigest())[:CONTROL_N]
    chosen = selected + controls
    if len({x["case_id"] for x in chosen}) != len(chosen):
        raise ValueError("overlapping controls")
    conn_cache = {}
    results = []
    try:
        for x in chosen:
            dbid = x["db_id"]
            if dbid not in conn_cache:
                paths = list((db_root / dbid).glob("*.sqlite"))
                direct = db_root / dbid / f"{dbid}.sqlite"
                if direct.is_file():
                    paths = [direct]
                if len(paths) != 1:
                    raise FileNotFoundError("ambiguous or absent physical SQLite db " + dbid)
                conn = sqlite3.connect(paths[0].resolve().as_uri() + "?mode=ro", uri=True)
                conn.execute("PRAGMA query_only=ON")
                conn_cache[dbid] = conn
            row = diagnose(x, signal_by_id[x["case_id"]], conn_cache[dbid])
            row["selection"] = "ORIGINAL_PHYSICAL_SIGNAL" if x in selected else "HASH_SELECTED_UNFLAGGED_CONTROL"
            results.append(row)
    finally:
        for conn in conn_cache.values():
            conn.close()
    output_data = {
        "study": "blind-physical-value-binding-v1", "status": "FROZEN_BEFORE_LABELS",
        "input_candidate_sha256": digest(original), "input_signal_sha256": digest(detect),
        "targets": len(selected), "unflagged_controls": len(controls),
        "no_gold_sql": True, "no_model_calls": True, "no_sql_rewrite": True,
        "selected_case_ids": [x["case_id"] for x in chosen],
        "records": results,
        "hypothesis": "A subset of missing literal-column values has unambiguous physically observed same-column normalization alternatives; other missing literals should abstain",
    }
    write(output, output_data)
    print("PHYSICAL_BINDING_FROZEN",json.dumps({
        "n":len(results),"targets":len(selected),"controls":len(controls),
        "status_counts":dict(Counter(f["status"] for row in results for f in row["findings"]))
    }),flush=True)


def score_probe(frozen: Path, labels: Path, output: Path):
    result, score = load(frozen), load(labels)
    if result.get("status") != "FROZEN_BEFORE_LABELS" or result.get("no_gold_sql") is not True:
        raise ValueError("not a frozen blind probe")
    if score.get("status") != "sealed_original_scores" or score.get("study") != SOURCE:
        raise ValueError("original labels not sealed")
    label_by_id = {x["case_id"]: x for x in score["labels"]}
    if len(label_by_id) != N:
        raise ValueError("not all historic source labels present")
    x = []
    for row in result["records"]:
        lb = label_by_id[row["case_id"]]
        if row["db_id"] != lb["db_id"] or row["shard"] != lb["shard"]:
            raise ValueError("mismatched case provenance")
        statuses = sorted(set(v["status"] for v in row["findings"] if v["status"] not in ("ALREADY_EXACT", "ABSTAIN")))
        x.append({
            "case_id": row["case_id"], "db_id": row["db_id"], "shard_status": lb["shard_status"],
            "selection": row["selection"], "execution_success": lb["execution_success"],
            "strict_ex": lb["strict_ex"], "statuses": statuses,
        })
    primary = [z for z in x if z["shard_status"] == "complete"]
    if not primary:
        raise ValueError("no valid original shards")
    def summary(rows):
        return {"n":len(rows),"executable":sum(x["execution_success"] for x in rows),
                "original_ex_wrong_executable":sum(x["execution_success"] and not x["strict_ex"] for x in rows),
                "by_observation":{
                    status:{"n":len(samples),"original_wrong":sum(not s["strict_ex"] for s in samples),
                            "original_right":sum(s["strict_ex"] for s in samples),
                            "dbs":sorted({s["db_id"] for s in samples})}
                    for status in ("NORMALIZED_SAME_COLUMN", "SUBSTRING_AMBIGUOUS", "POSSIBLE_COLUMN_MISBIND")
                    if (samples:=[s for s in rows if s["execution_success"] and status in s["statuses"]])}
                }
    report={
        "study":result["study"],"status":"HISTORICAL_RISK_ASSOCIATION_ONLY",
        "primary_complete_original_shards":summary(primary),
        "invalid_original_shard_separate":summary([z for z in x if z["shard_status"] != "complete"]),
        "original_source_already_exposed":True,
        "actual_repaired_sqls":0,"actual_strict_ex_rescues":0,
        "no_new_model_calls":True,
        "promotion_decision":"HOLD_REQUIRES_INDEPENDENT_VALIDATION",
        "limitations":["No publisher SQL seen by detection stage","Original EX association is not a semantic diagnosis",
                       "Substring matches are unsafe as automatic substitutions",
                       "Correct-query false alarms and unlabeled controls retained",
                       "No codepath or model changes; zero causal EX improvement"],
        "cases":x,
    }
    write(output,report)
    print("PHYSICAL_BINDING_SCORE",json.dumps({"primary":report["primary_complete_original_shards"],
                                                 "invalid":report["invalid_original_shard_separate"]}),flush=True)


def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="mode", required=True)
    a=sub.add_parser("probe")
    a.add_argument("--candidates",type=Path,required=True)
    a.add_argument("--signals",type=Path,required=True)
    a.add_argument("--database-root",type=Path,required=True)
    a.add_argument("--output",type=Path,required=True)
    b=sub.add_parser("score")
    b.add_argument("--probes",type=Path,required=True)
    b.add_argument("--labels",type=Path,required=True)
    b.add_argument("--output",type=Path,required=True)
    v=p.parse_args()
    if v.mode=="probe":
        run_probe(v.candidates,v.signals,v.database_root,v.output)
    else:
        score_probe(v.probes,v.labels,v.output)


if __name__=="__main__":
    main()
