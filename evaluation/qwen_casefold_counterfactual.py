"""Exploratory CPU-only casefold counterfactual on frozen Qwen9B candidates.

Construct candidate SQL from a *positive* physical SQLite casefold value probe
without reading benchmark reference SQL. Freeze all revisions, then score both
original and candidate against original pinned BIRD DEV gold. Do NOT promote
post-hoc gains: the historic frozen100 has already been inspected repeatedly.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

from evaluation.qwen_literal_grounding_audit import (
    load_frozen_qwen_shards, SOURCE_SHA, _ident,
)
from evaluation.run_bird import _resolve_database
from semantic_sql.execution import SQLExecutionError, execute_readonly
from semantic_sql.verifier import verify_sql_preflight

RAW_SOURCE_SHA = "ffd8018378ddb1a8794753e0a31cfc81862ff7318a5184c22f3dc4ce03a03feb"
SQL_EQUALS = re.compile(
    r"\b(?P<alias>[A-Za-z_]\w*)\.(?P<column>[A-Za-z_]\w*)"
    r"\s*=\s*(?P<rhs>'(?:''|[^'])*')", re.I
)


def rewrite_verified_casefold(sql: str, probes: list[dict]) -> str | None:
    """Abstain unless all equalities with the literal use the same column."""
    hits = [p for p in probes if p.get("status") == "casefold_match"]
    if len(hits) != 1:
        return None
    info = hits[0]
    old = str(info["value"])
    new = str(info.get("replacement") or "")
    column = str(info["column"])
    if not new or old == new or old.casefold() != new.casefold():
        return None
    before = "'" + old.replace("'", "''") + "'"
    after = "'" + new.replace("'", "''") + "'"
    occurrences = sql.count(before)
    if occurrences == 0:
        return None
    matches = [m for m in SQL_EQUALS.finditer(sql) if m.group("rhs") == before]
    if len(matches) != occurrences or any(
        m.group("column").casefold() != column.casefold() for m in matches
    ):
        return None
    candidate = sql.replace(before, after)
    return candidate if candidate != sql else None


def _read_source(path: Path) -> dict[str, dict]:
    if sha256(path.read_bytes()).hexdigest() != RAW_SOURCE_SHA:
        raise ValueError("BIRD source content hash differs from frozen evaluation")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if len(rows) != 1534:
        raise ValueError("BIRD source row count changed")
    return {str(row["question_id"]): row for row in rows}


def audit_counterfactual(shards: list[Path], audit: Path, source: Path,
                         database_root: Path) -> dict:
    history = load_frozen_qwen_shards(shards)
    report = json.loads(audit.read_text(encoding="utf-8"))
    if (report.get("protocol") != "qwen9b-sql-conditioned-literal-value-gate-v1"
        or report.get("baseline_100",{}).get("cases") != 100
        or report.get("no_new_qwen_inference") is not True):
        raise ValueError("unrecognized earlier physical-evidence report")
    by_id = {str(row["case_id"]): row for row in report["cases"]}
    if len(by_id) != 100:
        raise ValueError("missing audited IDs")
    # Freeze all counterfactual SQLs before reading a single reference SQL.
    frozen = []
    for case in history:
        cid = str(case["case_id"])
        old_sql = str(case["guarded"].get("final_sql") or "")
        info = by_id[cid]
        if info["candidate_sql_sha256"] != sha256(old_sql.encode()).hexdigest():
            raise ValueError(f"candidate SQL drift at case {cid}")
        new_sql = rewrite_verified_casefold(old_sql, info["probes"])
        if new_sql is not None:
            frozen.append({
                "case_id": cid, "db_id": str(case["database_id"]),
                "question": str(case["question"]),
                "baseline_sql": old_sql, "revised_sql": new_sql,
                "changes": [p for p in info["probes"] if p["status"]=="casefold_match"],
                "baseline_archived_ex": bool(case["guarded"].get("official_ex")),
            })
    frozen_sha = sha256(json.dumps(
        [(x["case_id"], x["revised_sql"]) for x in frozen],
        ensure_ascii=False,separators=(",",":"),
    ).encode()).hexdigest()

    # Gold access only begins here.
    refs = _read_source(source)
    evaluated = []
    for item in frozen:
        raw = refs[item["case_id"]]
        if str(raw["db_id"]) != item["db_id"] or raw["question"] != item["question"]:
            raise ValueError("source identity mismatched original Qwen prediction")
        db = _resolve_database(database_root, item["db_id"])
        preflight = verify_sql_preflight(db, item["revised_sql"])
        if not preflight.ok:
            evaluated.append({**item, "scorable": False, "error": "candidate_preflight"})
            continue
        try:
            orig = execute_readonly(db, item["baseline_sql"], max_rows=50_000)
            revised = execute_readonly(db, item["revised_sql"], max_rows=50_000)
            gold = execute_readonly(db, raw["SQL"], max_rows=50_000)
            if orig.truncated or revised.truncated or gold.truncated:
                raise ValueError("truncated result invalidates strict EX")
        except (SQLExecutionError, ValueError) as exc:
            evaluated.append({**item, "scorable": False, "error":type(exc).__name__})
            continue
        evaluated.append({
            **item, "scorable":True,
            "baseline_ex": set(orig.rows) == set(gold.rows),
            "revised_ex": set(revised.rows) == set(gold.rows),
            "baseline_bag": Counter(orig.rows) == Counter(gold.rows),
            "revised_bag": Counter(revised.rows) == Counter(gold.rows),
            "baseline_rows":len(orig.rows),
            "revised_rows":len(revised.rows),
            "gold_rows":len(gold.rows),
        })
    valid=[x for x in evaluated if x["scorable"]]
    return {
        "protocol":"qwen9b-casefold-source-positive-counterfactual-v1",
        "qwen_archived_questions":100,
        "candidate_count_pre_gold":len(frozen),
        "candidate_set_sha256_pre_gold":frozen_sha,
        "valid_gold_and_execution":len(valid),
        "original_ex":sum(bool(x["baseline_ex"]) for x in valid),
        "revised_ex":sum(bool(x["revised_ex"]) for x in valid),
        "original_bag":sum(bool(x["baseline_bag"]) for x in valid),
        "revised_bag":sum(bool(x["revised_bag"]) for x in valid),
        "rescues":sum(not x["baseline_ex"] and x["revised_ex"] for x in valid),
        "regressions":sum(x["baseline_ex"] and not x["revised_ex"] for x in valid),
        "cases":evaluated,
        "no_qwen_inference":True,"no_sql_training":True,
        "research_only":True,
        "boundary":("Already evaluated, repeatedly inspected BIRD DEV100. "
                    "Counterfactual is retrospectively selected using real database "
                    "positive evidence, but scores are exploratory; no independent "
                    "generalization or production improvement is claimed."),
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs",type=Path,nargs="+",required=True)
    p.add_argument("--audit",type=Path,required=True)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--database-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError("refusing to overwrite a prior counterfactual report")
    result=audit_counterfactual(a.inputs,a.audit,a.source,a.database_root)
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",
                        encoding="utf-8")
    print("QWEN_CASEFOLD_COUNTERFACTUAL",json.dumps({
        k:v for k,v in result.items() if k!="cases"
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
