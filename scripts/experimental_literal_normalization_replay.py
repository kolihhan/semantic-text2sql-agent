"""Gold-blind deterministic literal normalization candidate -> later strict EX replay.

Research-only counterfactual: no default graph change, no LLM, no golden SQL
available during candidate construction. Historical BIRD DEV is EXPOSED DATA,
so improvements here are only mechanism feasibility, never independent proof.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

from scripts.blind_physical_semantic_signals import (
    SOURCE, N, _SQL_EQ, _aliases, _schema, _identifier,
)
from scripts.blind_literal_binding_probe import physical_probe
from semantic_sql.verifier import verify_sql_preflight
from semantic_sql.execution import execute_readonly, SQLExecutionError
from evaluation.run_bird import _resolve_database

SOURCE_SHA = "0310b7ca2dfb6247234e2073f93e7a75106ee41ba55231f75d06d2ef68efe2ec"
SHARDS = {0: "complete", 1: "complete", 2: "complete", 3: "invalid"}
STRONG_FLAGS = {"VALUE_ABSENT_FROM_COLUMN", "VALUE_CASE_ONLY_PHYSICAL_MATCH"}
MAX_SQL_EQ_OPERANDS = 12
MAX_ROWS = 50000


def read(path: Path):
    return json.loads(path.read_text("utf-8"))


def dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", "utf8")


def hash_json(obj):
    return sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _replacement(conn, sql: str, match, tables, columns, aliases):
    literal = match.group("value").replace("''", "'")
    if len(literal) <= 1 or len(literal) > 96:
        return None
    qualifier, col = _identifier(match.group("alias") or ""), _identifier(match.group("column"))
    if qualifier:
        table_id = aliases.get(qualifier)
    else:
        found = [t for t in set(aliases.values()) if col in columns.get(t, {})]
        table_id = found[0] if len(found) == 1 else None
    if table_id not in tables or col not in columns.get(table_id, {}):
        return None
    table_name, physical = tables[table_id], columns[table_id][col]
    exact, status = physical_probe(conn, table_name, physical, f'"{physical.replace(chr(34), chr(34)*2)}" COLLATE BINARY = ?', literal)
    if status != "OK" or exact:
        return None
    quoted = '"' + physical.replace('"', '""') + '"'
    values, status = physical_probe(conn, table_name, physical, f"TRIM(CAST({quoted} AS TEXT)) COLLATE NOCASE = ?", literal.strip())
    if status != "OK" or len(set(values)) != 1:
        return None
    corrected = values[0]
    if corrected == literal:
        return None
    # Only ASCII casing or boundary whitespace; do not use fuzzy edit distance,
    # substring, type coercion, synonyms or alternate column binding.
    if corrected.strip().casefold() != literal.strip().casefold():
        return None
    return {
        "start": match.start("value"), "end": match.end("value"),
        "original_literal_sha256": sha256(literal.encode()).hexdigest(),
        "replacement_literal_sha256": sha256(corrected.encode()).hexdigest(),
        "bound_table": table_id, "bound_column": col,
        "replacement_quoted_content": corrected.replace("'", "''"),
    }


def build_candidate(sql: str, conn, db_path: Path):
    tables, columns, _ = _schema(conn)
    aliases = _aliases(sql, tables)
    patches = []
    for match in list(_SQL_EQ.finditer(sql))[:MAX_SQL_EQ_OPERANDS]:
        p = _replacement(conn, sql, match, tables, columns, aliases)
        if p is not None:
            patches.append(p)
    if not patches:
        return {"status": "NO_UNIQUE_NORMALIZED_WITNESS", "sql": sql, "replacements": []}
    if any(x["start"] >= y["start"] for x, y in zip(patches, patches[1:])):
        return {"status": "OVERLAPPING_REPLACEMENTS_ABSTAIN", "sql": sql, "replacements": []}
    revised = sql
    for patch in reversed(patches):
        revised = revised[:patch["start"]] + patch["replacement_quoted_content"] + revised[patch["end"]:]
    if not verify_sql_preflight(db_path, revised).ok:
        return {"status": "FAILED_SQL_PREFLIGHT_ABSTAIN", "sql": sql, "replacements": []}
    clean = [{k: v for k, v in patch.items() if k != "replacement_quoted_content"} for patch in patches]
    return {"status": "FROZEN_NORMALIZATION_CANDIDATE", "sql": revised, "replacements": clean}


def freeze(candidates: Path, signals: Path, db_root: Path, output: Path):
    original, det = read(candidates), read(signals)
    if original.get("study") != SOURCE or original.get("status") != "frozen_unlabeled" or len(original.get("cases", [])) != N:
        raise ValueError("original candidate artifact missing")
    if det.get("study") != SOURCE or det.get("status") != "signals_sealed_no_labels" or len(det.get("cases", [])) != N:
        raise ValueError("original detector artifact missing")
    if any(set(x) != {"case_id","db_id","question","selected_sql","shard"} for x in original["cases"]):
        raise ValueError("unexpected gold/correctness field")
    flag_by_id = {x["case_id"]: x for x in det["cases"]}
    if len(flag_by_id) != N or len({x["case_id"] for x in original["cases"]}) != N:
        raise ValueError("incomplete/duplicate frozen input")
    selected = [x for x in original["cases"] if STRONG_FLAGS.intersection(flag_by_id[x["case_id"]]["flags"])]
    results, connections = [], {}
    try:
        for case in selected:
            dbid = case["db_id"]
            if dbid not in connections:
                db_path = _resolve_database(db_root, dbid)
                conn = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
                conn.execute("PRAGMA query_only=ON")
                connections[dbid] = (conn, db_path)
            conn, db_path = connections[dbid]
            original_sql = case["selected_sql"]
            attempt = build_candidate(original_sql, conn, db_path)
            results.append({
                "case_id": case["case_id"], "database_id": dbid, "shard": case["shard"],
                "original_sql_sha256": sha256(original_sql.encode()).hexdigest(),
                "status": attempt["status"], "candidate_sql": attempt["sql"] if attempt["status"] == "FROZEN_NORMALIZATION_CANDIDATE" else None,
                "replacements": attempt["replacements"],
            })
    finally:
        for conn, _ in connections.values():
            conn.close()
    d={
        "status": "SQL_CANDIDATES_SEALED_WITHOUT_GOLD",
        "source": SOURCE, "original_input_sha256": hash_json(original),
        "detector_input_sha256": hash_json(det), "original_n": N,
        "candidate_pool_n": len(selected),
        "proposals_n": sum(x["status"] == "FROZEN_NORMALIZATION_CANDIDATE" for x in results),
        "cases": results, "no_model_calls": True, "no_gold_at_proposal": True,
        "research_only": True,
        "population": "historically exposed BIRD DEV 100 original Qwen9B, invalid shard 3 separately",
        "algorithm": "same-column exact-absent, unique trim+nocase existing value; no synonym or column change",
    }
    dump(output,d)
    print("NORMALIZATION_CANDIDATES_SEALED",json.dumps({
        "pool":d["candidate_pool_n"],"proposed":d["proposals_n"],
        "status_counts":dict(Counter(x["status"] for x in results))}),flush=True)


def score(patches: Path, candidate_source: Path, labels: Path, bird_source: Path, db_root: Path, output: Path):
    proposals, original, sealed = read(patches), read(candidate_source), read(labels)
    if proposals.get("status") != "SQL_CANDIDATES_SEALED_WITHOUT_GOLD" or not proposals.get("no_gold_at_proposal"):
        raise ValueError("not frozen before original gold")
    if proposals.get("original_input_sha256") != hash_json(original):
        raise ValueError("candidate source payload changed")
    if sealed.get("status") != "sealed_original_scores" or sealed.get("study") != SOURCE:
        raise ValueError("original EX labels not sealed")
    if sha256(bird_source.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError("BIRD publisher source SHA mismatch")
    raw = read(bird_source)
    by_id = {str(x.get("question_id") or x.get("id") or i): x for i,x in enumerate(raw)}
    baseline = {x["case_id"]: x for x in sealed["labels"]}
    original_by_id = {x["case_id"]: x for x in original["cases"]}
    if len(baseline) != N or len(original_by_id) != N:
        raise ValueError("historical cohort incomplete")
    rows = []
    for row in proposals["cases"]:
        cid = row["case_id"]
        selected, old = original_by_id[cid], baseline[cid]
        if selected["db_id"] != row["database_id"] or selected["shard"] != row["shard"]:
            raise ValueError("candidate/shard mismatch")
        if sha256(selected["selected_sql"].encode()).hexdigest() != row["original_sql_sha256"]:
            raise ValueError("original SQL changed")
        result = {
            "case_id": cid, "db_id": row["database_id"],
            "shard_status": old["shard_status"], "status": row["status"],
            "original_strict_ex": old["strict_ex"], "original_execution_success": old["execution_success"],
            "corrected_strict_ex": None, "corrected_bag_ex": None, "execution_success": None,
            "score_status": "NO_PROPOSAL",
        }
        if row["status"] == "FROZEN_NORMALIZATION_CANDIDATE":
            public = by_id[cid]
            if str(public["db_id"]) != selected["db_id"] or str(public["question"]) != selected["question"]:
                raise ValueError("source case drift")
            db_path = _resolve_database(db_root, row["database_id"])
            try:
                candidate_out = execute_readonly(db_path, row["candidate_sql"], max_rows=MAX_ROWS)
                gold_out = execute_readonly(db_path, str(public["SQL"]), max_rows=MAX_ROWS)
                if candidate_out.truncated or gold_out.truncated:
                    result["score_status"] = "CENSORED_TRUNCATED"
                else:
                    result["execution_success"] = True
                    result["corrected_strict_ex"] = set(candidate_out.rows)==set(gold_out.rows)
                    result["corrected_bag_ex"] = Counter(candidate_out.rows)==Counter(gold_out.rows)
                    result["score_status"] = "SCORED"
            except SQLExecutionError:
                result["score_status"] = "EXECUTION_FAILED"
                result["execution_success"] = False
                result["corrected_strict_ex"] = False
        rows.append(result)
    def split_summary(items):
        proposed = [x for x in items if x["status"]=="FROZEN_NORMALIZATION_CANDIDATE"]
        scored = [x for x in proposed if x["score_status"]=="SCORED"]
        rescues = [x for x in scored if x["corrected_strict_ex"] and not x["original_strict_ex"]]
        regressions = [x for x in scored if not x["corrected_strict_ex"] and x["original_strict_ex"]]
        return {
            "original_total": len(items), "proposals": len(proposed), "scored": len(scored),
            "rescues": len(rescues), "regressions": len(regressions),
            "net_ex_delta":len(rescues)-len(regressions),
            "rescued_ids":[x["case_id"] for x in rescues],
            "regressed_ids":[x["case_id"] for x in regressions],
            "rescue_dbs":sorted({x["db_id"] for x in rescues}),
            "nonrescued":sum(not x["corrected_strict_ex"] for x in scored),
            "censored":sum(x["score_status"]=="CENSORED_TRUNCATED" for x in proposed),
            "execution_failures":sum(x["score_status"]=="EXECUTION_FAILED" for x in proposed),
        }
    valid = [x for x in rows if x["shard_status"]=="complete"]
    invalid=[x for x in rows if x["shard_status"]!="complete"]
    report={
        "status": "HISTORICAL_DIAGNOSTIC_NOT_INDEPENDENT_PROMOTION",
        "source": SOURCE, "n_original": N,
        "complete_shards_0_2": split_summary(valid), "invalid_shard_3_separate": split_summary(invalid),
        "original_baseline_75_scored": 27,
        "original_baseline_100_historical_reference": 39,
        "full_frozen_100_new_score":None,
        "no_new_qwen_calls":True,
        "independent_validation":False,"default_pipeline_changed":False,
        "promotion_decision":"HOLD_PENDING_INDEPENDENT_DATABASES",
        "warning":"Post-hoc 100 DEV already exposed; no 100-query paired EX promotion. The controls were original pre-generated SQL. Only modified proposals rescored with original BIRD gold, after source freeze.",
        "case_scores":rows,
    }
    dump(output,report)
    print("NORMALIZED_LITERAL_REPLAY",json.dumps({"complete":report["complete_shards_0_2"],
                                                 "invalid":report["invalid_shard_3_separate"]}),flush=True)


def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="cmd",required=True)
    a=sub.add_parser("freeze")
    a.add_argument("--candidates",type=Path,required=True);a.add_argument("--signals",type=Path,required=True)
    a.add_argument("--database-root",type=Path,required=True);a.add_argument("--output",type=Path,required=True)
    b=sub.add_parser("score")
    b.add_argument("--patches",type=Path,required=True);b.add_argument("--candidates",type=Path,required=True)
    b.add_argument("--labels",type=Path,required=True);b.add_argument("--source",type=Path,required=True)
    b.add_argument("--database-root",type=Path,required=True);b.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.cmd=="freeze":freeze(a.candidates,a.signals,a.database_root,a.output)
    else:score(a.patches,a.candidates,a.labels,a.source,a.database_root,a.output)


if __name__=="__main__":
    main()
