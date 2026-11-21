"""Gold-blind 12-case same-Qwen9B Semantic Plan verifier experiment.

Historical BIRD DEV100 questions and guarded SQL are ALREADY EXPOSED: this is
diagnostic, not an independent final score. The freeze/infer/score jobs are
separate. Gold SQL and labels never enter the inference job.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import statistics
import time

DEV100_SOURCE_SHA = "0310b7ca2dfb6247234e2073f93e7a75106ee41ba55231f75d06d2ef68efe2ec"
DEV100_RAW_SHA = "ffd8018378ddb1a8794753e0a31cfc81862ff7318a5184c22f3dc4ce03a03feb"
QWEN_MODEL = "qwen3.5:9b-q4_K_M"
QWEN_DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
HISTORICAL_WORKFLOW = 37307108955
N = 12


def write_exclusive(p: Path, value: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2)


def _load_source(p: Path) -> list[dict]:
    raw = p.read_bytes()
    if sha256(raw).hexdigest() != DEV100_RAW_SHA:
        raise ValueError("Pinned BIRD source SHA mismatch")
    historical = raw.replace(b"\n", b"\r\n")
    if sha256(historical).hexdigest() != DEV100_SOURCE_SHA:
        raise ValueError("Historical CRLF BIRD source SHA mismatch")
    rows = json.loads(raw)
    if len(rows) != 1534:
        raise ValueError("Wrong pinned source universe")
    return rows


def _id(row: dict, i: int) -> str:
    return str(row.get("question_id") or row.get("id") or i)


def freeze(*, source: Path, shards: Path, output: Path) -> dict:
    rows = _load_source(source)
    src = {_id(row, i): row for i, row in enumerate(rows)}
    files = sorted(shards.glob("qwen9b-shard-*.json"))
    if len(files) != 4:
        raise ValueError("Missing original 4 historical Qwen9B shards")
    baseline = []
    statuses = Counter()
    for f in files:
        obj = json.loads(f.read_text())
        if (obj.get("source_sha256") != DEV100_SOURCE_SHA
            or obj.get("model_digest") != QWEN_DIGEST
            or obj.get("sample_size") != 25):
            raise ValueError("Historical Qwen9B model/source identity mismatch")
        statuses[obj.get("status")] += 1
        if obj.get("status") != "complete":
            continue  # exclude INVALID entire shard by original provenance, not scores
        for c in obj["cases"]:
            cid = str(c["case_id"])
            s = src[cid]
            if (str(s["db_id"]) != str(c["database_id"])
                or s["question"] != c["question"]):
                raise ValueError("Historical question/DB identity drift")
            sql = c["guarded"].get("final_sql") or ""
            baseline.append({
                "case_id": cid, "db_id": str(c["database_id"]),
                "question": s["question"], "evidence": s.get("evidence") or "",
                "historical_guarded_sql": sql,
                "historical_guarded_status": c["guarded"].get("status"),
            })
    if statuses != Counter({"complete": 3, "invalid": 1}) or len(baseline) != 75:
        raise ValueError(f"Historical shard provenance changed: {statuses} {len(baseline)}")
    # Fixed source/ID hash, never rank by historical SQL quality or EX.
    baseline.sort(key=lambda c: sha256(("scv12-source-v1|" + c["case_id"] + "|" + c["db_id"]).encode()).hexdigest())
    selected = baseline[:N]
    payload = {
        "protocol": "HISTORICALLY_EXPOSED_DEV100_SCV12_SOURCE_LOCKED_V1",
        "status": "FROZEN_GOLD_FREE",
        "selected_by": "SHA256(sc v12 fixed namespace | case ID | DB ID), three originally complete provenance shards",
        "historical_workflow": HISTORICAL_WORKFLOW,
        "model": QWEN_MODEL, "model_digest": QWEN_DIGEST,
        "source_sha256": DEV100_SOURCE_SHA, "sample_size": N,
        "source_reused_not_new_ood": True, "gold_sql_present": False,
        "selected": selected,
    }
    write_exclusive(output, payload)
    print("SOURCE_FREEZE_NO_LABELS", json.dumps({
        "n": N, "original_db_count": len({x["db_id"] for x in selected}),
        "ids": [x["case_id"] for x in selected],
    }), flush=True)
    return payload


def _model_digest(model: str) -> str:
    import urllib.request
    with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=30) as resp:
        body = json.load(resp)
    models = body.get("models", [])
    match = [m for m in models if m.get("name") == model or m.get("model") == model]
    if len(match) != 1:
        raise RuntimeError("Pinned Qwen model not present locally")
    return match[0]["digest"]


def _execute_sql(db: Path, sql: str):
    from semantic_sql.execution import execute_readonly, SQLExecutionError
    if not sql:
        return False, None, "missing_sql"
    try:
        result = execute_readonly(db, sql, max_rows=50000)
        return True, result, ""
    except SQLExecutionError as exc:
        return False, None, str(exc)[:350]


def infer(*, frozen: Path, db_root: Path, output: Path) -> dict:
    from semantic_sql.catalog import DatabaseCatalog
    from semantic_sql.contracts import SQLCandidate
    from semantic_sql.inference import run_guarded
    from semantic_sql.providers import OllamaProvider
    from semantic_sql.verifier import verify_sql_preflight
    from evaluation.run_bird import CountingProvider, _resolve_database
    from evaluation.schema_context import select_schema_context

    data = json.loads(frozen.read_text())
    if (data["protocol"] != "HISTORICALLY_EXPOSED_DEV100_SCV12_SOURCE_LOCKED_V1"
        or data["status"] != "FROZEN_GOLD_FREE"
        or data["sample_size"] != N or data["gold_sql_present"] is not False):
        raise ValueError("Invalid label-free experiment source")
    actual = _model_digest(QWEN_MODEL)
    if actual != QWEN_DIGEST:
        raise ValueError("Original Qwen3.5-9B model digest drift")

    results = {
        "protocol": data["protocol"], "status": "INCOMPLETE",
        "model": QWEN_MODEL, "model_digest": actual,
        "sample_size": N, "source_sha256": DEV100_SOURCE_SHA,
        "same_frozen_sql_all_arms": True, "reference_sql_opened": False,
        "historical_question_exposure": True,
        "cases": [],
    }
    write_exclusive(output, results)
    cache = {}
    for i, case in enumerate(data["selected"], 1):
        cid, dbid = case["case_id"], case["db_id"]
        if dbid not in cache:
            db = _resolve_database(db_root, dbid)
            cache[dbid] = (db, DatabaseCatalog.from_sqlite(db))
        db, catalog = cache[dbid]
        schema, _ = select_schema_context(catalog, db_root, dbid, case["question"])
        sql = case["historical_guarded_sql"]
        preflight = verify_sql_preflight(db, sql) if sql else None
        baseline_ok, base_result, error = _execute_sql(db, sql)
        result = {
            "case_id": cid, "db_id": dbid,
            "question_sha": sha256(case["question"].encode()).hexdigest(),
            "baseline_sql": sql, "baseline_preflight_ok": bool(preflight and preflight.ok),
            "baseline": {"final_sql": sql, "status": "ok" if baseline_ok else "execution_failed",
                         "execution_success": baseline_ok, "error": error,
                         "truncated": bool(base_result and base_result.truncated),
                         "model_calls": 0},
        }
        for arm, policy in (("always", "always"), ("selective", "selective")):
            if not preflight or not preflight.ok:
                result[arm] = {
                    "status": "not_attempted_original_preflight_invalid",
                    "final_sql": sql, "model_calls": 0, "latency_s": 0.0,
                    "stages": [], "truncated": False,
                }
                continue
            provider = CountingProvider(OllamaProvider(
                model=QWEN_MODEL, timeout_s=600.0,
                temperature=0.0, seed=42, num_predict=512,
            ))
            start = time.perf_counter()
            verdict = run_guarded(
                database=db, provider=provider,
                question=case["question"], evidence=case["evidence"] or None,
                schema_context=schema, initial_candidate=SQLCandidate(sql=sql),
                max_repairs=2, max_rows=50000,
                semantic_plan_verifier=True, semantic_plan_policy=policy,
            )
            elapsed = time.perf_counter() - start
            result[arm] = {
                "status": verdict.status,
                "final_sql": verdict.candidate.sql if verdict.candidate else None,
                "model_calls": provider.total_calls, "latency_s": round(elapsed, 4),
                "truncated": verdict.truncated, "error": verdict.message,
                "stages": [{"name": s.name, "summary": s.summary} for s in verdict.stages],
            }
        results["cases"].append(result)
        # Durable intermediate artifact for post-failure diagnosis; never interpret as complete.
        output.write_text(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
        print("SCV12_GOLD_BLIND", i, "/", N, cid,
              "baseline_preflight", result["baseline_preflight_ok"],
              "always", result["always"]["status"], "selective", result["selective"]["status"],
              "incremental_calls", result["always"]["model_calls"] + result["selective"]["model_calls"],
              flush=True)
    results["status"] = "COMPLETED_GOLD_BLIND_DECISIONS"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
    return results


def score(*, frozen: Path, blind: Path, source: Path, db_root: Path, output: Path) -> dict:
    from evaluation.run_bird import _resolve_database
    from semantic_sql.execution import execute_readonly, SQLExecutionError
    from collections import Counter as Bag
    rows = _load_source(source)
    labels = {_id(row,i):row for i,row in enumerate(rows)}
    f = json.loads(frozen.read_text())
    d = json.loads(blind.read_text())
    if (d.get("status") != "COMPLETED_GOLD_BLIND_DECISIONS"
        or d.get("reference_sql_opened") is not False
        or d.get("model_digest") != QWEN_DIGEST
        or len(d.get("cases", [])) != N
        or [c["case_id"] for c in d["cases"]] != [c["case_id"] for c in f["selected"]]):
        raise ValueError("Gold-blind decisions incomplete or source altered")
    scores = []
    for c,source_case in zip(d["cases"],f["selected"]):
        label = labels[c["case_id"]]
        if (label["question"] != source_case["question"]
            or str(label["db_id"]) != source_case["db_id"]
            or c["baseline_sql"] != source_case["historical_guarded_sql"]):
            raise ValueError("Question, database, or frozen baseline changed")
        db = _resolve_database(db_root, c["db_id"])
        try:
            gold = execute_readonly(db, str(label["SQL"]), max_rows=50000)
            scorable = not gold.truncated
        except SQLExecutionError:
            gold, scorable = None, False
        result={"case_id":c["case_id"],"db":c["db_id"],"scorable":scorable}
        for arm in ("baseline","always","selective"):
            candidate=c[arm]
            executed = candidate["status"] in {"ok","semantic_unverified"}
            try:
                pred=execute_readonly(db,candidate["final_sql"],max_rows=50000) if executed else None
            except (SQLExecutionError,TypeError):
                pred=None
            valid=bool(pred and scorable and not pred.truncated)
            set_ex=bool(valid and set(pred.rows) == set(gold.rows))
            bag_ex=bool(valid and Bag(pred.rows) == Bag(gold.rows))
            result[arm]={"set_ex":set_ex if scorable else None,
                         "bag_ex":bag_ex if scorable else None,
                         "status":candidate["status"],
                         "model_calls":candidate.get("model_calls",0)}
        scores.append(result)
    eligible=[c for c in scores if c["scorable"]]
    totals={}
    for arm in ("baseline","always","selective"):
        totals[arm] = {
            "set_correct":sum(c[arm]["set_ex"] for c in eligible),
            "bag_correct":sum(c[arm]["bag_ex"] for c in eligible),
            "model_calls_incremental":sum(c[arm]["model_calls"] for c in scores),
            "semantic_statuses":dict(Counter(c[arm]["status"] for c in scores)),
        }
        if arm!="baseline":
            totals[arm].update({
                "rescues":sum(c[arm]["set_ex"] and not c["baseline"]["set_ex"] for c in eligible),
                "regressions":sum(c["baseline"]["set_ex"] and not c[arm]["set_ex"] for c in eligible),
            })
            totals[arm]["net"] = totals[arm]["rescues"]-totals[arm]["regressions"]
    result={
        "protocol":"EXPOSED_DEV100_QWEN9B_TYPED_PLAN_12_PAIRED_SET_BAG_V1",
        "status":"COMPLETE_HISTORICAL_SOURCE_ONLY_NOT_INDEPENDENT",
        "n_selected":N,"n_scorable":len(eligible),
        "db_count":len({x["db"] for x in scores}),
        "model":QWEN_MODEL,"model_digest":QWEN_DIGEST,
        "same_original_qwen_sql_all_arms":True,
        "no_sql_training":True,
        "holdout_opened":False,
        "independence":"HISTORICALLY EXPOSED DEV100; this is only a diagnostic screen",
        "scores":totals,"cases":scores,
    }
    write_exclusive(output,result)
    print("SCV12_STRICT_RESULTS",json.dumps({k:v for k,v in result.items() if k!="cases"}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="phase",required=True)
    f=sub.add_parser("freeze");f.add_argument("--source",type=Path,required=True);f.add_argument("--shards",type=Path,required=True);f.add_argument("--out",type=Path,required=True)
    i=sub.add_parser("infer");i.add_argument("--frozen",type=Path,required=True);i.add_argument("--db-root",type=Path,required=True);i.add_argument("--out",type=Path,required=True)
    s=sub.add_parser("score");s.add_argument("--frozen",type=Path,required=True);s.add_argument("--blind",type=Path,required=True);s.add_argument("--source",type=Path,required=True);s.add_argument("--db-root",type=Path,required=True);s.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.phase=="freeze":freeze(source=args.source,shards=args.shards,output=args.out)
    elif args.phase=="infer":infer(frozen=args.frozen,db_root=args.db_root,output=args.out)
    else:score(frozen=args.frozen,blind=args.blind,source=args.source,db_root=args.db_root,output=args.out)


if __name__=="__main__":
    main()
