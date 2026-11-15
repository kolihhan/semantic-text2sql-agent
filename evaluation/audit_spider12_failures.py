"""CPU-only post-hoc audit of Spider value-evidence failures.

Never tunes the tested SQLs. Checks whether incorrect labels reflect true
relational mistakes or SQLite reference/data case sensitivity. Never promotes.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

from evaluation.qwen_value_evidence_spider12 import _dbpath, PROTOCOL, ARCHIVE_SHA
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.execution import SQLExecutionError, execute_readonly

EXPECTED_COHORT = "3f82cc607007be2c847fc3b8073c5d40a8e87b16294ef80dbaef18a6024f0e31"


def quote_identifier(value: str) -> str:
    if not value or "\x00" in value:
        raise ValueError("bad SQLite identifier")
    return '"' + value.replace('"', '""') + '"'


def literal_counts(database: Path, table: str, column: str, value: str) -> dict[str, int]:
    """Compare exact physical values to lowercase reference literals under SQLite."""
    catalog = DatabaseCatalog.from_sqlite(database)
    if table not in catalog.tables or column not in catalog.tables[table].columns:
        raise ValueError("identifier outside actual physical SQLite schema")
    name = f"{quote_identifier(table)}.{quote_identifier(column)}"
    query = f"SELECT COUNT(*) FROM {quote_identifier(table)} WHERE {name} = ?"
    with sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro",uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        exact = db.execute(query,(value,)).fetchone()[0]
        lower_exact = db.execute(query,(value.casefold(),)).fetchone()[0]
        nocase = db.execute(query + " COLLATE NOCASE", (value.casefold(),)).fetchone()[0]
    return {"value_exact":exact,"lowercase_exact":lower_exact,"lowercase_nocase":nocase}


def audit(report: dict, manifest: dict, source: list[dict], db_root: Path) -> dict:
    if (report.get("protocol")!=PROTOCOL or report.get("cohort_sha256")!=EXPECTED_COHORT
        or report.get("count")!=12 or report.get("scorable")!=12
        or report.get("baseline_ex")!=9 or report.get("evidence_ex")!=9
        or report.get("rescues")!=0 or report.get("regressions")!=0
        or manifest.get("cohort_sha256")!=EXPECTED_COHORT
        or manifest.get("archive_sha256")!=ARCHIVE_SHA):
        raise ValueError("non-frozen 12-case result manifest; refuse to diagnose")
    by_case = {x["case_id"]:x for x in manifest["cases"]}
    if len(by_case)!=12 or {x["case_id"] for x in report["cases"]}!=set(by_case):
        raise ValueError("physical 12-case coverage drift")
    observed = []
    for result in report["cases"]:
        if result["baseline"]["ex"] is not False or result["evidence"]["ex"] is not False:
            continue
        original = by_case[result["case_id"]]
        if result["db_id"]!=original["db_id"]:
            raise ValueError("case DB mismatch")
        db = _dbpath(db_root, result["db_id"])
        gold_sql = str(source[original["index"]]["query"])
        gold = execute_readonly(db,gold_sql,max_rows=50000)
        if gold.truncated:
            raise ValueError("gold reference truncated")
        arm_observations={}
        for name in ("baseline","evidence"):
            pred_sql = result[name]["sql"]
            scored=execute_readonly(db,pred_sql,max_rows=50000)
            if scored.truncated:
                raise ValueError("model reference truncated")
            # Not a replacement score, only diagnostics against original reference.
            arm_observations[name]={
                "sql":pred_sql,"row_count":len(scored.rows),
                "gold_count":len(gold.rows),
                "predicted_only":len(set(scored.rows)-set(gold.rows)),
                "gold_only":len(set(gold.rows)-set(scored.rows)),
            }
        physical_literals=[
            {"path":f"{h['table']}.{h['column']}","value":h["value"],
             **literal_counts(db,h["table"],h["column"],h["value"])}
            for h in original["value_hits"]
        ]
        observed.append({
            "case_id":result["case_id"],"db_id":result["db_id"],
            "question":original["question"],"reference_sql":gold_sql,
            "gold_row_count":len(gold.rows),
            "physical_literals":physical_literals,
            "arms":arm_observations,
        })
    if len(observed)!=3:
        raise ValueError("unexpected count of failed cases")
    unique_questions = len({
        " ".join(x["question"].casefold().replace("what are the names of","list the name of").split())
        for x in observed
    })
    return {
        "audit":"cpu-only-physical-sqlite-spider12-three-failures-v1",
        "cohort_sha256":EXPECTED_COHORT,
        "failed_case_count":len(observed),
        "distinct_database_count":len({x["db_id"] for x in observed}),
        "conclusion":"No value-evidence rescue; evaluate repeated questions and reference literal casing separately from true relational set mistakes.",
        "not_full_score_revision":True,"model_inference_ran":False,
        "cases":observed,
    }


def main():
    p=argparse.ArgumentParser()
    for a in ("final","manifest","source","db-root","output"):
        p.add_argument("--"+a,type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError("do not overwrite immutable audit")
    result=audit(
        json.loads(args.final.read_text(encoding="utf-8")),
        json.loads(args.manifest.read_text(encoding="utf-8")),
        json.loads(args.source.read_text(encoding="utf-8")),
        args.db_root,
    )
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("QWEN9B_SPIDER12_PHYSICAL_FAILURE_AUDIT",json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
