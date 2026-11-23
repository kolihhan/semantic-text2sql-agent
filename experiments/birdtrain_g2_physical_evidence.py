"""BIRD TRAIN G2: gold-blind check of genuinely NEW literal-to-column evidence.

No Qwen, no SQL references, no DEV100 gold, no reserved BIRD TRAIN DBs.
A physical value match is evidence of presence, not of semantic correctness.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3

from scripts.prepare_birdtrain_qwen9b_sft_source import fetch_database, fetch_manifest

TRAIN_SHA = "b6851fb0f2b5027833aff45a0b5e01326359b245be1c812edc66986d0df1d402"
VALID_SHA = "cc112305bc9de33a03cea704f2edecfdfb45e62cc953ae498cea27fbed93b506"
MAX_DBS, MAX_BYTES, MAX_COLUMNS = 8, 64 * 1024 * 1024, 80
MAX_PHRASES = 5
START_WORDS = set("How What Which Where When Who Why Find List Show Give Name Calculate Count Are The Of For In And By Does Is Do".split())


def source_rows(path: Path, digest: str) -> list[dict]:
    raw = path.read_bytes()
    assert sha256(raw).hexdigest() == digest, "historical BIRD TRAIN source mismatch"
    rows = []
    for line in raw.splitlines():
        item = json.loads(line)
        assert len(item["messages"]) == 3 and item["messages"][1]["role"] == "user"
        # The final assistant response contains original gold SQL: NEVER USE IT.
        user = item["messages"][1]["content"]
        assert user.startswith("Question: ") and "\nSchema context:" in user
        q = user.split("\nSchema context:", 1)[0][len("Question: "):]
        rows.append({"db": item["source_db_id"], "question": q, "id": item["source_index"]})
    return rows


def phrases(q: str) -> list[str]:
    quoted = re.findall(r'["“”‘’\x27]([^"“”‘’\x27]{3,55})["“”‘’\x27]', q)
    proper = re.findall(r'\b(?:[A-Z][a-z]{2,}\s+){1,3}[A-Z][a-z]{2,}\b', q)
    found = []
    for value in [*quoted, *proper]:
        value = " ".join(value.split())
        if 3 <= len(value) <= 55 and value.split()[0] not in START_WORDS:
            if value.casefold() not in {s.casefold() for s in found}:
                found.append(value)
    return found[:MAX_PHRASES]


def text_columns(con: sqlite3.Connection) -> list[tuple[str, str]]:
    tables = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    cols = []
    for table in tables:
        ident = '"' + table.replace('"','""') + '"'
        for r in con.execute("PRAGMA table_info(" + ident + ")"):
            affinity = (r[2] or "").upper()
            if not affinity or any(k in affinity for k in ("TEXT","CHAR","CLOB")):
                cols.append((table,r[1]))
    return cols


def exact_matches(con: sqlite3.Connection, phrase: str,
                  cols: list[tuple[str,str]]) -> tuple[list[str],int]:
    found, unknown = [], 0
    for table,col in cols:
        ident = lambda s: '"' + s.replace('"','""') + '"'
        ticks = [0]
        def progress():
            ticks[0] += 1
            return int(ticks[0] > 100)
        con.set_progress_handler(progress, 1000)
        try:
            sql = "SELECT 1 FROM " + ident(table) + " WHERE " + ident(col) + " = ? COLLATE NOCASE LIMIT 1"
            if con.execute(sql,(phrase,)).fetchone() is not None:
                found.append(table+"."+col)
        except sqlite3.OperationalError as e:
            if "interrupted" in str(e).lower():
                unknown += 1
            else:
                raise
        finally:
            con.set_progress_handler(None, 0)
    return found, unknown


def assess(q: str, con: sqlite3.Connection, cols: list[tuple[str,str]]) -> dict:
    ps = phrases(q)
    result = {"has_candidate":int(bool(ps)),"confirmed_unique":0,
              "ambiguous":0,"incomplete":0,"candidate_phrases":len(ps),"physical_hits":0}
    for p in ps:
        hits, unknown = exact_matches(con,p,cols)
        result["physical_hits"] += len(hits)
        if len(hits) == 1 and not unknown:
            result["confirmed_unique"] = 1
        if len(hits) > 1:
            result["ambiguous"] = 1
        if unknown:
            result["incomplete"] = 1
    if result["ambiguous"] or result["incomplete"]:
        result["confirmed_unique"] = 0
    return result


def run(src: Path, work: Path, out: Path) -> dict:
    records = source_rows(src/"train.jsonl", TRAIN_SHA) + source_rows(src/"validation.jsonl", VALID_SHA)
    assert len(records) == 306
    by_db = {}
    for row in records:
        by_db.setdefault(row["db"], []).append(row)
    assert len(by_db) == 20
    meta = fetch_manifest()  # publisher metadata only, never the two reserved physical DBs
    assert set(by_db) <= set(meta)
    eligible = [db for db in by_db if meta[db]["size"] <= MAX_BYTES]
    selected = sorted(eligible, key=lambda db:(meta[db]["size"],db))[:MAX_DBS]
    if len(selected) < 4:
        raise RuntimeError("not enough previously exposed physical DBs under resource cap")
    report = {
        "status":"CPU_ONLY_G2_SOURCE_BLIND",
        "source_rows":len(records),"historically_exposed_source_dbs":len(by_db),
        "physical_dbs":selected,"physical_db_count":len(selected),
        "physical_db_roster_sha256":sha256("\n".join(selected).encode()).hexdigest(),
        "new_Qwen_calls":0,"gold_sql_used":False,
        "reserved_unseen_DBs_accessed":False,"DEV100_accessed":False,"heldout300_accessed":False,
        "per_db":{},"counts":{},
        "meaning":"Exact SQLite literal-to-column observations are NOT verified NL semantic correctness; do not repeat FK/column descriptions already in baseline."
    }
    overall = Counter()
    for db in selected:
        dbfile = fetch_database(db, meta[db], work)
        con = sqlite3.connect("file:"+str(dbfile)+"?mode=ro", uri=True)
        con.execute("PRAGMA query_only=ON")
        cols = text_columns(con)
        stats = Counter({"questions":len(by_db[db]),"text_columns":len(cols)})
        if len(cols) > MAX_COLUMNS:
            stats["too_wide_unknown"] = len(by_db[db])
        else:
            for row in by_db[db]:
                stats.update(assess(row["question"],con,cols))
        con.close()
        report["per_db"][db] = dict(stats)
        for key in ("questions","has_candidate","confirmed_unique","ambiguous",
                    "incomplete","too_wide_unknown","candidate_phrases","physical_hits"):
            overall[key] += stats[key]
        print("G2",db,"questions",stats["questions"],
              "unique",stats["confirmed_unique"],flush=True)
    report["counts"] = dict(overall)
    report["new_evidence_question_fraction"] = (
        overall["confirmed_unique"]/overall["questions"] if overall["questions"] else 0)
    report["dbs_with_evidence"] = sum(v.get("confirmed_unique",0)>0 for v in report["per_db"].values())
    report["go_for_expensive_generation"] = (
        overall["confirmed_unique"]>=10
        and report["new_evidence_question_fraction"] >= 0.10
        and report["dbs_with_evidence"]>=3)
    report["decision"] = ("ELIGIBLE_FOR_PRE_REGISTRATION_NOT_EX_GAIN"
                          if report["go_for_expensive_generation"]
                          else "HOLD_NO_PROVEN_INCREMENTAL_EVIDENCE_COVERAGE")
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"counts":report["counts"],"fraction":report["new_evidence_question_fraction"],
                      "dbs_with_evidence":report["dbs_with_evidence"],"decision":report["decision"]},indent=2))
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--work",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    run(a.source,a.work,a.output)


if __name__=="__main__":
    main()
