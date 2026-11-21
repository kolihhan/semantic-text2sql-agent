"""Locked DEV-derived normalization algorithm transported to historical BIRD TRAIN VAL64.

Three stages:
  1. Strip original model scores and publisher SQL from source question+model SQL.
  2. Run unchanged physical-DB normalization algorithm with NO labels.
  3. Only AFTER stage 2 freeze, execute patched SQL and publisher SQL readonly.

The four TRAIN validation databases are disjoint from BIRD DEV, but this older
TRAIN64 validation cohort was already exposed in previous research. Thus this
is cross-database *diagnostic replication*, NOT pristine independent promotion.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

from scripts.experimental_literal_normalization_replay import build_candidate
from scripts.prepare_birdtrain_qwen9b_sft_source import fetch_database, fetch_manifest
from semantic_sql.execution import SQLExecutionError, execute_readonly

MODEL_DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
VAL64_SHA = "cc112305bc9de33a03cea704f2edecfdfb45e62cc953ae498cea27fbed93b506"
VAL64_DBS = {"movie_3", "music_tracker", "superstore", "world"}
PROTECTED = {"simpson_episodes", "works_cycles"}
RUN_ID = 43892823922
N = 64
MAX_ROWS = 50000


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def write(path, x):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(x, indent=2, sort_keys=True, ensure_ascii=False)+"\n", encoding="utf-8")


def canonical_sha(x):
    return sha256(json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def prepare(source_path, baseline_path, out_unlabeled, out_gold):
    raw = Path(source_path).read_bytes()
    if sha256(raw).hexdigest() != VAL64_SHA:
        raise ValueError("publisher source VAL64 digest mismatch")
    source = [json.loads(s) for s in raw.decode().splitlines() if s.strip()]
    base = read(baseline_path)
    if not (len(source) == N and base.get("n") == N and base.get("status") == "complete"
            and base.get("model_digest") == MODEL_DIGEST
            and base.get("source_validation_sha256") == VAL64_SHA
            and len(base.get("cases", [])) == N):
        raise ValueError("original Qwen9B TRAIN64 baseline identity mismatch")
    lookup = {int(x["source_index"]): x for x in source}
    if len(lookup) != N or {x["source_db_id"] for x in source} != VAL64_DBS or PROTECTED & VAL64_DBS:
        raise ValueError("validation source physical DB identity changed")
    candidates, sealed = [], []
    for record in base["cases"]:
        source_index = int(record["source_index"])
        src = lookup[source_index]
        if record["db_id"] != src["source_db_id"] or record["prompt_sha256"] != src["input_sha256"]:
            raise ValueError("original Qwen source prompt/db mismatch")
        prompt = str(src["messages"][1]["content"])
        if "\nSchema context:\n" not in prompt or not prompt.startswith("Question: "):
            raise ValueError("original user question format drift")
        question = prompt.split("\nSchema context:\n", 1)[0].removeprefix("Question: ").strip()
        sql = str(record.get("guarded_sql") or "")
        candidates.append({"source_index":source_index, "db_id":src["source_db_id"],
                           "question":question, "sql":sql, "prompt_sha256":record["prompt_sha256"]})
        sealed.append({"source_index":source_index, "db_id":src["source_db_id"],
                       "original_ex":bool(record["guarded"]["ex"]),
                       "original_bag":bool(record["guarded"]["bag"]),
                       "original_execution_ok":bool(record["guarded"]["execution_ok"]),
                       "gold_sql":str(src["messages"][2]["content"])})
    if len({x["source_index"] for x in candidates}) != N:
        raise ValueError("duplicate source row")
    unlabeled={"status":"FROZEN_TRAIN64_QWEN_SQL_NO_GOLD","n":N,"validation_sha256":VAL64_SHA,
               "model_digest":MODEL_DIGEST,"dbs":sorted(VAL64_DBS),"cases":sorted(candidates,key=lambda x:x["source_index"])}
    labels={"status":"SEALED_TRAIN64_PUBLISHER_GOLD_AND_EX","n":N,"validation_sha256":VAL64_SHA,
            "cases":sorted(sealed,key=lambda x:x["source_index"])}
    write(out_unlabeled,unlabeled)
    write(out_gold,labels)
    print("TRANSFER_PREPARE_FROZEN",json.dumps({"unlabeled_sha":canonical_sha(unlabeled),"n":N,
                                                 "validation_dbs":sorted(VAL64_DBS)}),flush=True)


def validated_databases(root:Path):
    manifest=fetch_manifest()
    if not VAL64_DBS <= set(manifest) or PROTECTED & VAL64_DBS:
        raise ValueError("missing/potentially protected physical DB")
    paths={}
    for db in sorted(VAL64_DBS):
        paths[db]=fetch_database(db,manifest[db],root)
    return paths


def freeze_sql_candidates(unlabeled, db_root, output):
    data=read(unlabeled)
    if data.get("status")!="FROZEN_TRAIN64_QWEN_SQL_NO_GOLD" or data.get("n")!=N or data.get("model_digest")!=MODEL_DIGEST:
        raise ValueError("missing unlabeled original model cohort")
    if set(data.get("dbs",[]))!=VAL64_DBS or len(data.get("cases",[]))!=N:
        raise ValueError("validation disjoint database count drift")
    if any(set(x)!={"source_index","db_id","question","sql","prompt_sha256"} for x in data["cases"]):
        raise ValueError("unexpected gold/label leakage to freeze")
    paths=validated_databases(Path(db_root))
    conn={}
    results=[]
    try:
        for db,p in paths.items():
            dbconn=sqlite3.connect(p.resolve().as_uri()+"?mode=ro",uri=True)
            dbconn.execute("PRAGMA query_only=ON")
            conn[db]=dbconn
        for case in data["cases"]:
            original=case["sql"]
            candidate=build_candidate(original,conn[case["db_id"]],paths[case["db_id"]]) if original else {
                "status":"NO_SQL_CANDIDATE","sql":original,"replacements":[]}
            results.append({
                "source_index":case["source_index"],"db_id":case["db_id"],
                "original_sql_sha256":sha256(original.encode()).hexdigest(),
                "status":candidate["status"],"replacements":candidate["replacements"],
                "candidate_sql":candidate["sql"] if candidate["status"]=="FROZEN_NORMALIZATION_CANDIDATE" else None,
            })
    finally:
        for c in conn.values():c.close()
    payload={
        "status":"TRAIN64_NORMALIZATION_PROPOSALS_SEALED_BEFORE_GOLD",
        "n":N,"original_unlabeled_sha256":canonical_sha(data),
        "model_digest":MODEL_DIGEST,"physical_dbs":sorted(VAL64_DBS),
        "algorithm":"LOCKED_DEV_QWEN9B_SAME_COLUMN_UNIQUE_TRIM_NOCASE_V1",
        "no_gold_accessed":True,"no_qwen_inference":True,"cases":results,
    }
    write(output,payload)
    print("TRANSFER_FROZEN",json.dumps({"n":N,"proposed":sum(x["status"]=="FROZEN_NORMALIZATION_CANDIDATE" for x in results),
                                        "dbs":sorted(VAL64_DBS)}),flush=True)


def score_frozen(frozen, unlabeled, labels, db_root, output):
    proposals, original, sealed=read(frozen), read(unlabeled), read(labels)
    if (proposals.get("status")!="TRAIN64_NORMALIZATION_PROPOSALS_SEALED_BEFORE_GOLD"
            or proposals.get("original_unlabeled_sha256")!=canonical_sha(original)
            or proposals.get("no_gold_accessed") is not True or proposals.get("n")!=N):
        raise ValueError("candidate not sealed before gold")
    if sealed.get("status")!="SEALED_TRAIN64_PUBLISHER_GOLD_AND_EX" or sealed.get("n")!=N:
        raise ValueError("label source wrong")
    label_by_id={x["source_index"]:x for x in sealed["cases"]}
    original_by_id={x["source_index"]:x for x in original["cases"]}
    if len(label_by_id)!=N or len(original_by_id)!=N or len(proposals["cases"])!=N:
        raise ValueError("source coverage drift")
    paths=validated_databases(Path(db_root))
    rows=[]
    for p in proposals["cases"]:
        idx=p["source_index"]
        original=original_by_id[idx]
        gold=label_by_id[idx]
        if (original["db_id"]!=p["db_id"] or gold["db_id"]!=p["db_id"]
                or sha256(original["sql"].encode()).hexdigest()!=p["original_sql_sha256"]):
            raise ValueError("original SQL or question changed after freezing")
        row={"source_index":idx,"db_id":p["db_id"],"status":p["status"],
             "original_ex":gold["original_ex"],"original_bag":gold["original_bag"],
             "candidate_ex":None,"candidate_bag":None,"score_status":"UNCHANGED"}
        if p["status"]=="FROZEN_NORMALIZATION_CANDIDATE":
            try:
                pred=execute_readonly(paths[p["db_id"]],p["candidate_sql"],max_rows=MAX_ROWS)
                true=execute_readonly(paths[p["db_id"]],gold["gold_sql"],max_rows=MAX_ROWS)
                if pred.truncated or true.truncated:
                    row["score_status"]="CENSORED"
                else:
                    row["candidate_ex"]=set(pred.rows)==set(true.rows)
                    row["candidate_bag"]=Counter(pred.rows)==Counter(true.rows)
                    row["score_status"]="SCORED"
            except SQLExecutionError:
                row["score_status"]="SQL_EXECUTION_FAILED"
                row["candidate_ex"]=False
                row["candidate_bag"]=False
        rows.append(row)
    patches=[r for r in rows if r["status"]=="FROZEN_NORMALIZATION_CANDIDATE"]
    rescues=[r for r in patches if r["score_status"]=="SCORED" and r["candidate_ex"] and not r["original_ex"]]
    regressions=[r for r in patches if r["score_status"]=="SCORED" and not r["candidate_ex"] and r["original_ex"]]
    bag_rescues=[r for r in patches if r["score_status"]=="SCORED" and r["candidate_bag"] and not r["original_bag"]]
    bag_regressions=[r for r in patches if r["score_status"]=="SCORED" and not r["candidate_bag"] and r["original_bag"]]
    result={
        "status":"CROSS_DB_HISTORIC_EXPOSED_DIAGNOSTIC_NOT_PRISTINE",
        "n":N,"physical_db_n":len(VAL64_DBS),"physical_dbs":sorted(VAL64_DBS),
        "original_historical_strict_ex":sum(r["original_ex"] for r in rows),
        "original_historical_bag_ex":sum(r["original_bag"] for r in rows),
        "proposals":len(patches),"scored_proposals":sum(r["score_status"]=="SCORED" for r in patches),
        "rescues":len(rescues),"regressions":len(regressions),
        "bag_rescues":len(bag_rescues),"bag_regressions":len(bag_regressions),
        "net_strict_ex_delta":len(rescues)-len(regressions),
        "new_counterfactual_strict_ex":sum(r["original_ex"] for r in rows)+len(rescues)-len(regressions)
        if all(r["score_status"]=="SCORED" for r in patches) else None,
        "rescued_ids":[r["source_index"] for r in rescues],
        "regressed_ids":[r["source_index"] for r in regressions],
        "rescue_dbs":sorted({r["db_id"] for r in rescues}),
        "new_model_calls":0,"default_graph_unchanged":True,
        "promotion_verdict":"HOLD_NO_INDEPENDENT_VALIDATION",
        "method_warning":"Frozen algorithm transported from DEV, but older TRAIN64 physically DB-disjoint dataset has historically been seen; no new Qwen model calls. Source EX vs original model replay for modified predictions only.",
        "cases":rows,
    }
    if result["rescues"]<3 or len(result["rescue_dbs"])<3 or result["regressions"]>1 or result["bag_regressions"]>0:
        result["research_gate"]="REJECT_OR_HOLD_NO_CROSS_DB_PROOF"
    else:
        result["research_gate"]="CANDIDATE_ONLY_REQUIRES_NEW_PRISTINE_REPLICATION"
    write(output,result)
    print("CROSS_DATABASE_NORMALIZATION_REPLAY",json.dumps({
        k:result[k] for k in ("proposals","rescues","regressions","bag_rescues",
                             "bag_regressions","rescue_dbs","net_strict_ex_delta",
                             "new_counterfactual_strict_ex","research_gate")}),flush=True)


def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="mode",required=True)
    a=sub.add_parser("prepare")
    for n in ("source","baseline","unlabeled","labels"):
        a.add_argument("--"+n, type=Path,required=True)
    b=sub.add_parser("freeze")
    for n in ("unlabeled","database-root","output"):
        b.add_argument("--"+n,type=Path,required=True)
    c=sub.add_parser("score")
    for n in ("frozen","unlabeled","labels","database-root","output"):
        c.add_argument("--"+n,type=Path,required=True)
    args=p.parse_args()
    if args.mode=="prepare":prepare(args.source,args.baseline,args.unlabeled,args.labels)
    elif args.mode=="freeze":freeze_sql_candidates(args.unlabeled,args.database_root,args.output)
    else:score_frozen(args.frozen,args.unlabeled,args.labels,args.database_root,args.output)


if __name__=="__main__":main()
