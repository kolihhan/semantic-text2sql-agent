"""Source-first physical schema audit of 81 public BIRD DEV500 Qwen3.5-9B failures.

No new model calls, no reference SQL read, no candidate selection/Guarded edit.
All source/cache/scorer hashes checked by the existing 215-case frozen audit.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time

from scripts.audit_dev500_zero_oracle_215 import load_original, load_dev

PILOT16 = {199, 235, 317, 474, 957, 852, 567, 477, 1497, 172, 227,
           344, 1229, 1093, 1251, 978}
COUNT_KEYS = {
    "109": "c.client_id", "1028": "m.id",
    "1256": "P.ID", "1289": "e.ID", "1441": "member.member_id"
}
CORRECT_CONTROLS = {"168", "1091", "1305"}
PROTOCOL = "ORIGINAL-QWEN9B-BIRD-DEV500-81-SOURCE-FIRST-PHYSICAL-GRAIN-V1"


def get_db(root: Path, db_id: str) -> Path:
    paths = list(root.rglob(db_id + ".sqlite"))
    if len(paths) != 1:
        raise ValueError("Physical DB for " + db_id + " ambiguous or missing: " + str(len(paths)))
    return paths[0]


def readonly_run(conn, sql: str) -> dict:
    if not re.match(r"(?is)^\s*(SELECT|WITH)\b", sql):
        return {"status": "not_select"}
    began = time.monotonic()
    conn.set_progress_handler(lambda: 1 if time.monotonic() - began > 8 else 0, 5000)
    try:
        rows = conn.execute(sql).fetchmany(200)
        data = json.dumps(rows, sort_keys=True, default=str)
        return {"status": "ok", "returned_rows": len(rows), "first_rows": rows[:8],
                "result_sha256": hashlib.sha256(data.encode()).hexdigest(),
                "truncated": len(rows) == 200}
    except sqlite3.Error as exc:
        return {"status": "sqlite_error", "exception": type(exc).__name__,
                "error": str(exc)[:120]}
    finally:
        conn.set_progress_handler(None, 0)


def physical_schema(conn, sql: str) -> dict:
    # Source-only lexical lookup: false positives are left for adjudication.
    tables = sorted(set(re.findall(r'(?i)\b(?:FROM|JOIN)\s+["]?([A-Za-z_]\w*)', sql)))
    out = {}
    for table in tables[:15]:
        safe = table.replace('"', '""')
        cols = [{"name": r[1], "type": r[2], "pk_order": r[5]} for r in
                conn.execute('PRAGMA table_info("' + safe + '")')]
        fks = [{"from": r[3], "to_table": r[2], "to_col": r[4]} for r in
               conn.execute('PRAGMA foreign_key_list("' + safe + '")')]
        out[table] = {"columns": cols,
                      "primary_key": [r["name"] for r in cols if r["pk_order"]],
                      "declared_fk": fks}
    return out


def question_target(q: str) -> str:
    q = q.lower()
    if re.search(r"\b(percent|percentage|ratio)\b", q):
        return "PERCENT: explicitly determine numerator and denominator units"
    if re.search(r"\bhow many\b|\bnumber of\b|\bcount\b", q):
        return "COUNT: determine answer entity and join multiplicity"
    if re.search(r"\b(highest|lowest|top|most|least|youngest)\b", q):
        return "RANK: determine ranked entity, scope and grouping"
    return "PROJECT/FILTER/RELATION: requires human role/key adjudication"


def run(a) -> dict:
    cache, scored = load_original(a.cache_dir, a.workers_dir)
    source = load_dev(a.source, a.manifest, set(cache))
    fully_wrong = {cid for cid, row in scored.items()
                   if row["candidate_executed"] == 8 and not row["pass8_oracle_set"]}
    assert len(fully_wrong) == 97
    remaining = fully_wrong - {str(i) for i in PILOT16}
    assert len(remaining) == 81
    connections = {}
    def connection(db):
        if db not in connections:
            filename = get_db(a.db_root, db)
            conn = sqlite3.connect(filename.as_uri() + "?mode=ro", uri=True)
            conn.execute("PRAGMA query_only=ON")
            connections[db] = conn
        return connections[db]
    try:
        cases = []
        for cid in sorted(remaining, key=int):
            orig = cache[cid]
            db = orig["db_id"]
            sql = next(x["sql"] for x in orig["samples"] if x["index"] == 0)
            question = source[cid]["question"]
            assert hashlib.sha256(question.encode()).hexdigest() == orig["question_sha256"]
            shapes = physical_schema(connection(db), sql)
            cases.append({
                "case_id": cid, "db_id": db, "question": question,
                "source_first_question_target": question_target(question),
                "first_original_qwen_sql": sql,
                "original_eight_sql_sha256": hashlib.sha256(json.dumps(
                    [x["sql"] for x in sorted(orig["samples"], key=lambda x: x["index"])],
                    sort_keys=True).encode()).hexdigest(),
                "source_physical_schema_and_declared_keys": shapes,
                "human_fk_and_entity_grain_review": "PENDING",
                "publisher_reference_sql_read": False,
            })
        probes = []
        for cid, key in COUNT_KEYS.items():
            assert cid in remaining
            db = cache[cid]["db_id"]
            sql = next(x["sql"] for x in cache[cid]["samples"] if x["index"] == 0)
            distinct_sql, n = re.subn(
                r"(?i)\bCOUNT\s*\(\s*\*\s*\)", "COUNT(DISTINCT " + key + ")",
                sql, count=1)
            assert n == 1
            original = readonly_run(connection(db), sql)
            distinct = readonly_run(connection(db), distinct_sql)
            probes.append({
                "case_id": cid, "db_id": db, "target_pk_hypothesis": key,
                "original_source_sql": sql,
                "count_distinct_counterfactual_sql": distinct_sql,
                "original_result": original, "distinct_result": distinct,
                "actual_physical_result_changed": (original["status"] == "ok"
                    and distinct["status"] == "ok"
                    and original["result_sha256"] != distinct["result_sha256"]),
                "no_gold_consulted": True,
                "not_new_model_generated_sql": True,
            })
        controls = []
        for cid in sorted(CORRECT_CONTROLS, key=int):
            assert scored[cid]["k8_set_ex"] is True
            controls.append({
                "case_id": cid, "db_id": cache[cid]["db_id"],
                "question": source[cid]["question"],
                "first_original_sql": next(x["sql"] for x in
                                           cache[cid]["samples"] if x["index"] == 0),
                "original_goldblind_k8_selector_was_correct": True,
            })
    finally:
        for conn in connections.values():
            conn.close()
    return {
        "protocol": PROTOCOL, "status": "SOURCE_FIRST_81_PHYSICAL_SCHEMA_COMPLETE",
        "original_model_digest": "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90",
        "n_full_source": 500, "n_zero_oracle": 215, "n_fully_executable_wrong": 97,
        "n_prior_pilot": 16, "remaining_count": len(cases),
        "database_count": len({x["db_id"] for x in cases}),
        "source_first_cases": cases, "predeclared_count_grain_probes": probes,
        "existing_correct_controls": controls,
        "no_new_model_calls": True, "no_gold_reference_consulted": True,
        "no_train_or_qlora": True, "no_guarded_code_changed": True,
        "no_new_bird_ex": True, "holdout300_untouched": True,
        "claim_boundary": ("A physical schema or COUNT(DISTINCT) result mismatch is not "
          "proof of correctness; full 81-case human target+FK review and new-source "
          "Guarded end-to-end BIRD SET EX A/B remain unperformed.")
    }


def main():
    p = argparse.ArgumentParser()
    for name in ("cache_dir", "workers_dir", "source", "manifest", "db_root", "out"):
        p.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError("evidence immutable, do not overwrite")
    result = run(a)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print("SOURCE_FIRST_81_PHYSICAL_SCHEMA_EVIDENCE", json.dumps({
        "cases": result["remaining_count"],
        "dbs": result["database_count"],
        "count_probes": [[x["case_id"], x["original_result"]["status"],
                          x["distinct_result"]["status"],
                          x["actual_physical_result_changed"]] for x in
                          result["predeclared_count_grain_probes"]],
        "new_model_calls": 0,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
