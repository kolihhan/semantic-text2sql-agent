"""Qwen3.5-9B zero-row targeted semantic repair; label-blind, research-only."""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
import time
from urllib.request import Request, urlopen

from scripts.blind_physical_semantic_signals import SOURCE, N, MODEL_DIGEST, _SQL_EQ, _schema, _aliases, _identifier
from evaluation.run_bird import _resolve_database
from semantic_sql.execution import execute_readonly, SQLExecutionError
from semantic_sql.verifier import verify_sql_preflight

MODEL = "qwen3.5:9b-q4_K_M"
SOURCE_SHA = "0310b7ca2dfb6247234e2073f93e7a75106ee41ba55231f75d06d2ef68efe2ec"
PROTOCOL = "zero-row-qwen9b-single-repair-v1"
MAX_ROWS = 50000
MAX_CALLS = 30

def load(path):
    return json.loads(Path(path).read_text("utf-8"))

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

def digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def original75(data):
    if data.get("study") != SOURCE or data.get("status") != "frozen_unlabeled" or data.get("model_digest") != MODEL_DIGEST or len(data.get("cases", [])) != N:
        raise ValueError("frozen source identity mismatch")
    if any(set(r) != {"case_id", "db_id", "question", "selected_sql", "shard"} for r in data["cases"]):
        raise ValueError("unlabeled artifact contains unexpected fields")
    if Counter(int(r["shard"]) for r in data["cases"]) != {0:25, 1:25, 2:25, 3:25}:
        raise ValueError("original shard counts changed")
    selected = [r for r in data["cases"] if int(r["shard"]) < 3]
    if len({str(r["case_id"]) for r in selected}) != 75:
        raise ValueError("duplicate ids")
    return selected

def extract_sql(reply):
    if not isinstance(reply, str):
        return None
    s = reply.strip()
    fence = chr(96) * 3
    if s.startswith(fence):
        s = re.sub(r"^" + re.escape(fence) + r"(?:sql)?\s*", "", s, flags=re.I)
        s = re.sub(r"\s*" + re.escape(fence) + r"$", "", s)
    s = s.strip()
    if not re.match(r"^(SELECT|WITH)\b", s, re.I) or re.search(r"--|/\*|\*/|;.*\S", s):
        return None
    return s

def quote(s):
    return '"' + s.replace('"','""') + '"'

def evidence(db, sql):
    conn = sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
    try:
        conn.execute("PRAGMA query_only=ON")
        tables, cols, _ = _schema(conn)
        schema = "\n".join(f"{t}({', '.join(list(cols[k].values())[:40])})" for k,t in sorted(tables.items())[:35])[:5800]
        aliases = _aliases(sql, tables)
        witness = []
        for match in list(_SQL_EQ.finditer(sql))[:6]:
            v = match.group("value").replace("''", "'")
            col = _identifier(match.group("column"))
            alias = _identifier(match.group("alias") or "")
            if alias:
                tid = aliases.get(alias)
            else:
                found = [t for t in set(aliases.values()) if col in cols.get(t, {})]
                tid = found[0] if len(found) == 1 else None
            if not tid or col not in cols.get(tid, {}) or not 1 < len(v) <= 90:
                continue
            tn, cn = tables[tid], cols[tid][col]
            budget = [0]
            def limited():
                budget[0] += 1
                return int(budget[0] > 3000)
            conn.set_progress_handler(limited, 1000)
            try:
                found = conn.execute(
                    f"SELECT DISTINCT CAST({quote(cn)} AS TEXT) FROM {quote(tn)} "
                    f"WHERE LOWER(CAST({quote(cn)} AS TEXT)) LIKE ? LIMIT 4",
                    (v[:2].lower() + "%",)
                ).fetchall()
                samples = [str(x[0])[:70] for x in found if x[0] is not None]
            except sqlite3.Error:
                samples = []
            finally:
                conn.set_progress_handler(None, 0)
            witness.append({"column": f"{tn}.{cn}", "sql_literal": v, "same_prefix_db_values": samples})
        return schema, witness
    finally:
        conn.close()

def check_model():
    with urlopen("http://127.0.0.1:11434/api/tags", timeout=30) as response:
        models = json.load(response).get("models", [])
    if not any(m.get("name") == MODEL and m.get("digest") == MODEL_DIGEST for m in models):
        raise ValueError("Qwen9B model digest mismatch")

def one_qwen_call(question, sql, schema, observations):
    prompt = ("Question:\n" + question[:1600] + "\nReal SQLite schema:\n" + schema +
              "\nOriginal SQL (successfully executed but returned ZERO rows):\n" + sql[:4000] +
              "\nRead-only evidence from the original SQL bound columns:\n" +
              json.dumps(observations, ensure_ascii=False)[:2600])
    system = ("Return one SQLite SELECT/CTE SQL statement only. Fix a single filter/value-to-column/join mistake "
              "only if grounded in the question, schema and observations. Keep the requested projection and aggregation. "
              "If the evidence is insufficient, repeat the original query unchanged. No explanations or markdown.")
    request = json.dumps({"model": MODEL, "system": system, "prompt": prompt,
                          "think": False, "stream": False,
                          "options": {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 512}}).encode()
    start = time.monotonic()
    try:
        with urlopen(Request("http://127.0.0.1:11434/api/generate", data=request,
                             headers={"Content-Type":"application/json"}), timeout=540) as response:
            record = json.load(response)
        if record.get("model") != MODEL:
            return None, time.monotonic()-start, "MODEL_ID_MISMATCH"
        return extract_sql(record.get("response")), time.monotonic()-start, "OK"
    except Exception as ex:
        return None, time.monotonic()-start, "MODEL_FAILURE_" + type(ex).__name__

def infer(unlabeled, dbroot, output):
    data = load(unlabeled)
    cases = original75(data)
    check_model()
    records, calls, model_s = [], 0, 0.
    for r in sorted(cases, key=lambda x: int(x["case_id"])):
        sql = r["selected_sql"]
        row = {"case_id":str(r["case_id"]), "db_id":r["db_id"], "original_sha":sha256(sql.encode()).hexdigest(),
               "status":"ABSTAIN", "candidate_sql":None, "model_called":False}
        if not sql or not re.search(r"\bWHERE\b|\bHAVING\b", sql, re.I):
            row["status"] = "NO_FILTER"
        else:
            db = _resolve_database(dbroot, r["db_id"])
            if not verify_sql_preflight(db, sql).ok:
                row["status"] = "ORIGINAL_INVALID"
            else:
                try:
                    out = execute_readonly(db, sql, max_rows=MAX_ROWS)
                except SQLExecutionError:
                    row["status"] = "ORIGINAL_NOT_EXECUTABLE"
                else:
                    if out.truncated or out.rows:
                        row["status"] = "ORIGINAL_NONEMPTY"
                    elif calls >= MAX_CALLS:
                        row["status"] = "MODEL_BUDGET_LIMIT"
                    else:
                        schema, observations = evidence(db, sql)
                        candidate, sec, api = one_qwen_call(r["question"], sql, schema, observations)
                        calls += 1
                        model_s += sec
                        row["model_called"] = True
                        if api != "OK":
                            row["status"] = api
                        elif not candidate or candidate.strip().rstrip(";") == sql.strip().rstrip(";"):
                            row["status"] = "UNCHANGED_OR_BAD_OUTPUT"
                        elif not verify_sql_preflight(db, candidate).ok:
                            row["status"] = "CANDIDATE_PREFLIGHT_FAIL"
                        else:
                            try:
                                alt = execute_readonly(db, candidate, max_rows=MAX_ROWS)
                                if alt.rows and not alt.truncated:
                                    row["status"], row["candidate_sql"] = "FROZEN_CANDIDATE", candidate
                                else:
                                    row["status"] = "CANDIDATE_EMPTY_OR_TRUNCATED"
                            except SQLExecutionError:
                                row["status"] = "CANDIDATE_SQL_ERROR"
        records.append(row)
        print("EMPTY_REPAIR_CASE", row["case_id"], row["status"], flush=True)
    packet = {"status":"FROZEN_NO_GOLD", "protocol":PROTOCOL, "n":75, "model_digest":MODEL_DIGEST,
              "source_digest":digest(data), "calls":calls, "model_seconds":round(model_s,2),
              "no_gold":True, "max_calls":MAX_CALLS, "cases":records}
    save(output, packet)
    print("EMPTY_REPAIR_FROZEN", json.dumps({"n":75, "calls":calls, "proposals":sum(x["status"]=="FROZEN_CANDIDATE" for x in records)}), flush=True)

def score(inferred, unlabeled, labels, publisher, dbroot, output):
    frozen, source, lab = load(inferred), load(unlabeled), load(labels)
    selected = original75(source)
    if frozen.get("status")!="FROZEN_NO_GOLD" or frozen.get("protocol")!=PROTOCOL or frozen.get("source_digest")!=digest(source) or frozen.get("no_gold") is not True:
        raise ValueError("not a valid sealed source")
    byid={str(r["case_id"]):r for r in selected}
    sealed={str(r["case_id"]):r for r in lab["labels"] if int(r["shard"])<3}
    if len(sealed)!=75 or sum(bool(r["strict_ex"]) for r in sealed.values())!=27:
        raise ValueError("original baseline 27/75 mismatch")
    if sha256(Path(publisher).read_bytes()).hexdigest()!=SOURCE_SHA:
        raise ValueError("publisher checksum mismatch")
    pub=load(publisher)
    original={str(r.get("question_id") or r.get("id") or i):r for i,r in enumerate(pub)}
    totals=Counter()
    controls=[]
    db_results={}
    for row in frozen["cases"]:
        cid=str(row["case_id"])
        old=byid[cid]
        if old["db_id"]!=row["db_id"] or sha256(old["selected_sql"].encode()).hexdigest()!=row["original_sha"]:
            raise ValueError("original candidate identity changed")
        truth=original[cid]
        if truth["db_id"]!=row["db_id"] or truth["question"]!=old["question"]:
            raise ValueError("publisher identity mismatch")
        path=_resolve_database(dbroot, row["db_id"])
        gold=execute_readonly(path, truth["SQL"], max_rows=MAX_ROWS)
        def evaluate(sql):
            try:
                pred=execute_readonly(path, sql, max_rows=MAX_ROWS)
                if gold.truncated or pred.truncated:
                    return False, False
                return set(pred.rows)==set(gold.rows), Counter(pred.rows)==Counter(gold.rows)
            except SQLExecutionError:
                return False, False
        baseline_set,baseline_bag=evaluate(old["selected_sql"])
        if baseline_set!=bool(sealed[cid]["strict_ex"]):
            raise ValueError("original SQL strict score drift: "+cid)
        if row["status"]=="FROZEN_CANDIDATE":
            cand_set,cand_bag=evaluate(row["candidate_sql"])
            totals["proposals"]+=1
            totals["rescues"]+=int(not baseline_set and cand_set)
            totals["regressions"]+=int(baseline_set and not cand_set)
            totals["bag_rescues"]+=int(not baseline_bag and cand_bag)
            totals["bag_regressions"]+=int(baseline_bag and not cand_bag)
            controls.append({"case_id":cid, "db_id":row["db_id"], "baseline_set":baseline_set,
                             "candidate_set":cand_set,"baseline_bag":baseline_bag,"candidate_bag":cand_bag})
        else:
            cand_set,cand_bag=baseline_set,baseline_bag
        totals["base_set"]+=int(baseline_set)
        totals["base_bag"]+=int(baseline_bag)
        totals["new_set"]+=int(cand_set)
        totals["new_bag"]+=int(cand_bag)
        db_results.setdefault(row["db_id"], {"n":0,"base":0,"new":0})
        v=db_results[row["db_id"]]
        v["n"]+=1
        v["base"]+=int(baseline_set)
        v["new"]+=int(cand_set)
    improved=sorted(db for db,v in db_results.items() if v["new"]>v["base"])
    passed=(totals["new_set"]-totals["base_set"]>=3 and totals["regressions"]<=1 and
            len(improved)>=3 and totals["new_bag"]>=totals["base_bag"])
    report={"status":"EXPOSED_DEV75_DIAGNOSTIC_ONLY", "protocol":PROTOCOL,"n":75,
            "dbs":len(db_results), "model_digest":MODEL_DIGEST, "model_calls":frozen["calls"],
            "model_seconds":frozen["model_seconds"], "scores":dict(totals), "improved_db_ids":improved,
            "per_database":db_results,"changed_case_details":controls,
            "research_gate":"SCREEN_PASSED_NEEDS_NEW_DB" if passed else "REJECT",
            "no_default_pipeline_changes":True,
            "warning":"Historical exposed 75 original Qwen SQL, not independent DEV100 improvement."}
    save(output,report)
    print("EMPTY_REPAIR_STRICT_SCORE",json.dumps({"base_set":totals["base_set"],"new_set":totals["new_set"],
      "base_bag":totals["base_bag"],"new_bag":totals["new_bag"],"rescues":totals["rescues"],
      "regressions":totals["regressions"],"dbs_improved":len(improved),
      "calls":frozen["calls"],"gate":report["research_gate"]}),flush=True)

def main():
    p=argparse.ArgumentParser()
    cmd=p.add_subparsers(dest="cmd",required=True)
    for name,fields in (("infer",("unlabeled","database-root","output")),
                        ("score",("inferred","unlabeled","labels","publisher","database-root","output"))):
        sub=cmd.add_parser(name)
        for field in fields:
            sub.add_argument("--"+field,required=True,type=Path)
    args=p.parse_args()
    if args.cmd=="infer":
        infer(args.unlabeled,args.database_root,args.output)
    else:
        score(args.inferred,args.unlabeled,args.labels,args.publisher,args.database_root,args.output)

if __name__=="__main__":
    main()
