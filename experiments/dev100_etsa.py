"""Gold-blind matched-cohort ETSA experiment. Research-only; main service unchanged.

Both generation arms receive the same frozen G0 SQL and run only on the same
source-only physical evidence triggers. Neither sees publisher SQL or EX labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.contracts import SQLCandidate
from semantic_sql.direct import _strip_fence, generate_direct_sql
from semantic_sql.execution import SQLExecutionError, execute_readonly
from semantic_sql.inference import run_guarded
from semantic_sql.providers import OllamaProvider
from semantic_sql.verifier import verify_sql_preflight
from evaluation.run_bird import _resolve_database
from evaluation.schema_context import select_schema_context


MODEL_DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
MODEL = "qwen3.5:9b-q4_K_M"


def structural_risk(question: str, sql: str, catalog: DatabaseCatalog, row_count: int) -> dict | None:
    """A source-only *risk hypothesis*, never a semantic correctness verdict.

    Physical schema and observed result shape are available; publisher gold is not.
    No outcome-based case list or historical bad-SQL string matching is permitted.
    """
    q = question.casefold()
    s = sql.casefold()
    has = lambda pattern: re.search(pattern, s, flags=re.I) is not None
    asks = lambda pattern: re.search(pattern, q, flags=re.I) is not None
    hint: tuple[str, str] | None = None
    if asks(r"\b(how many|number of|count of|total number of)\b") and not has(r"\bcount\s*\(") and not asks(r"\b(percent|percentage|rate)\b"):
        hint = ("COUNT_MISSING", "Question requests a count but SQL has no COUNT expression")
    elif asks(r"\b(average|mean)\b") and not has(r"\bavg\s*\("):
        hint = ("AVERAGE_MISSING", "Question requests an average but SQL has no AVG expression")
    elif asks(r"\b(percentage|percent|ratio|growth rate)\b") and not has(r"/|\bavg\s*\("):
        hint = ("RATIO_MISSING", "Question requests a ratio or percentage but SQL has no visible division or AVG")
    elif asks(r"\b(most|highest|lowest|least|top|maximum|minimum)\b") and not has(r"\border\s+by\b|\bmax\s*\(|\bmin\s*\("):
        hint = ("RANK_MISSING", "Question requests an extremum but SQL lacks ORDER BY, MAX or MIN")
    elif asks(r"\b(for each|each|per)\b") and has(r"\b(count|sum|avg)\s*\(") and not has(r"\bgroup\s+by\b|\bover\s*\("):
        hint = ("PER_GROUP_MISSING", "Question asks for a per-entity result but SQL aggregates without GROUP BY or window")
    elif asks(r"\b(both|all of|every)\b") and has(r"\bin\s*\([^)]+,") and not has(r"\bhaving\b|\bintersect\b|\bexists\b"):
        hint = ("ALL_OF_MISSING", "Possible all-of relational constraint represented as any-of IN predicate")
    elif asks(r"\b(how many|total number)\b") and asks(r"\b(in total|overall|altogether)\b") and has(r"\bgroup\s+by\b") and row_count > 1:
        hint = ("GLOBAL_GROUP_MISMATCH", "Global scalar total requested but SQL groups into multiple rows")

    # Additional bounded physical evidence (not labels or a replacement answer).
    keys = [
        (str(a), str(b), str(c), str(d))
        for a, b, c, d in catalog.foreign_keys
    ][:24]
    if hint is None and has(r"\bjoin\b") and keys:
        sql_tables = set(re.findall(r'\b(?:from|join)\s+["\x60\[]?([a-z_][a-z0-9_]*)', s))
        relevant = [edge for edge in keys if edge[0].casefold() in sql_tables and edge[2].casefold() in sql_tables]
        if len(relevant) == 1:
            a, ac, b, bc = relevant[0]
            # Only flag when neither FK endpoint is used in the ON predicate;
            # alternate non-FK joins may still be valid, so it is a risk only.
            joined = re.search(r"\bjoin\b.+?\bon\b(.+?)(?:\bwhere\b|\bgroup\b|\border\b|\blimit\b|$)", s, re.S)
            on_sql = joined.group(1) if joined else ""
            if on_sql and not (re.search(r"\b"+re.escape(ac.casefold())+r"\b", on_sql) and re.search(r"\b"+re.escape(bc.casefold())+r"\b", on_sql)):
                hint = ("FK_JOIN_RISK", "Join predicate does not use the sole declared physical foreign-key pair for the joined tables")

    if hint is None:
        return None
    code, reason = hint
    return {
        "code": code,
        "reason": reason,
        "original_result_rows": row_count,
        "foreign_keys": [f"{a}.{b} -> {c}.{d}" for a, b, c, d in keys],
        "physical_table_count": len(catalog.tables),
    }


def resolves_risk(sql: str, witness: dict) -> bool:
    """Blind acceptance gate applied identically to G1 and G2."""
    s = sql.casefold()
    code = witness["code"]
    patterns = {
        "COUNT_MISSING": r"\bcount\s*\(",
        "AVERAGE_MISSING": r"\bavg\s*\(",
        "RATIO_MISSING": r"/|\bavg\s*\(",
        "RANK_MISSING": r"\border\s+by\b|\bmax\s*\(|\bmin\s*\(",
        "PER_GROUP_MISSING": r"\bgroup\s+by\b|\bover\s*\(",
        "ALL_OF_MISSING": r"\bhaving\b|\bintersect\b|\bexists\b",
    }
    if code == "GLOBAL_GROUP_MISMATCH":
        return re.search(r"\bgroup\s+by\b", s) is None and re.search(r"\bcount\s*\(", s) is not None
    if code == "FK_JOIN_RISK":
        return any(
            re.search(r"\b"+re.escape(line.split(".")[1].split(" -> ")[0].casefold())+r"\b", s) and
            re.search(r"\b"+re.escape(line.split(".")[-1].casefold())+r"\b", s)
            for line in witness.get("foreign_keys", [])
        )
    return bool(re.search(patterns[code], s))


@dataclass(frozen=True)
class ResearchContext:
    database: Path
    catalog: DatabaseCatalog
    g1: OllamaProvider
    g2: OllamaProvider


class ResearchState(TypedDict, total=False):
    question: str
    schema_context: str
    original_sql: str
    original_rows: int
    witness: dict | None
    routed: bool
    g1_candidate: str | None
    g2_candidate: str | None
    g1_sql: str
    g2_sql: str
    g1_selected: bool
    g2_selected: bool
    decisions: dict
    probe_wall_s: float


def inspect_evidence(state: ResearchState, runtime) -> dict:
    t0 = time.perf_counter()
    witness = structural_risk(
        state["question"], state["original_sql"],
        runtime.context.catalog, state["original_rows"],
    )
    return {
        "witness": witness,
        "routed": witness is not None,
        "probe_wall_s": time.perf_counter()-t0,
    }


def after_inspect(state: ResearchState) -> str:
    return "propose" if state.get("routed", False) else "abstain"


def abstain(state: ResearchState) -> dict:
    return {
        "g1_sql": state["original_sql"], "g2_sql": state["original_sql"],
        "g1_candidate": None, "g2_candidate": None,
        "g1_selected": False, "g2_selected": False,
        "decisions": {"g1": "no_witness", "g2": "no_witness"},
    }


def propose(state: ResearchState, runtime) -> dict:
    core = (
        "Question:\n" + state["question"] + "\n\n"
        "Schema:\n" + state["schema_context"] + "\n\n"
        "Original SQL:\n" + state["original_sql"] + "\n\n"
        "Return a single alternative read-only SQLite SELECT/CTE query, SQL only."
    )
    generic = runtime.context.g1.complete_text(
        system="Produce one independent SQL solution to the question; do not rely on a semantic diagnosis. SQL only.",
        user=core,
    )
    evidence = json.dumps(state["witness"], ensure_ascii=False)
    targeted = runtime.context.g2.complete_text(
        system="Produce one structurally distinct SQL solution addressing the source-grounded risk. Do not assume the risk proves the old SQL is wrong. SQL only.",
        user=core + "\n\nPhysical schema/result evidence and structural risk hypothesis:\n" + evidence,
    )
    return {"g1_candidate": _strip_fence(generic), "g2_candidate": _strip_fence(targeted)}


def blind_choose(original: str, proposed: str | None, witness: dict, database: Path) -> tuple[str, str]:
    if not proposed or proposed.strip().rstrip(";") == original.strip().rstrip(";"):
        return original, "identical_or_empty"
    if not resolves_risk(proposed, witness):
        return original, "risk_not_resolved"
    if not verify_sql_preflight(database, proposed).ok:
        return original, "failed_preflight"
    try:
        execution = execute_readonly(database, proposed, max_rows=50000)
    except SQLExecutionError:
        return original, "failed_execute"
    if execution.truncated:
        return original, "truncated"
    return proposed, "selected_resolved_risk"


def select(state: ResearchState, runtime) -> dict:
    witness = state["witness"]
    original = state["original_sql"]
    sql1, why1 = blind_choose(original, state.get("g1_candidate"), witness, runtime.context.database)
    sql2, why2 = blind_choose(original, state.get("g2_candidate"), witness, runtime.context.database)
    return {
        "g1_sql": sql1, "g2_sql": sql2,
        "g1_selected": sql1 != original, "g2_selected": sql2 != original,
        "decisions": {"g1": why1, "g2": why2},
    }


def graph():
    g = StateGraph(ResearchState, context_schema=ResearchContext)
    g.add_node("physical_evidence", inspect_evidence)
    g.add_node("abstain", abstain)
    g.add_node("propose", propose)
    g.add_node("blind_select", select)
    g.add_edge(START, "physical_evidence")
    g.add_conditional_edges("physical_evidence", after_inspect, {"propose":"propose", "abstain":"abstain"})
    g.add_edge("propose", "blind_select")
    g.add_edge("blind_select", END)
    g.add_edge("abstain", END)
    return g.compile()


def write_atomic(path: Path, record: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def model_usage(p: OllamaProvider) -> dict:
    return {"calls":len(p.usage_records), **p.usage_totals}


def run(args) -> None:
    frozen = json.loads(Path(args.input).read_text(encoding="utf-8"))
    assert frozen["model_digest"] == MODEL_DIGEST
    assert len(frozen["cases"]) == 100
    shard = int(args.shard)
    assert 0 <= shard < 4
    selected = [c for i,c in enumerate(frozen["cases"]) if i%4==shard]
    assert len(selected)==25
    outpath=Path(args.output)
    output={"status":"incomplete","model":MODEL,"digest":MODEL_DIGEST,
            "cohort_sha256":frozen["cohort_sha256"],"shard":shard,"cases":[]}
    previous={}
    if args.resume_from:
        old=json.loads(Path(args.resume_from).read_text(encoding="utf-8"))
        assert old["model"]==MODEL and old["digest"]==MODEL_DIGEST
        assert old["cohort_sha256"]==frozen["cohort_sha256"] and old["shard"]==shard
        expected={str(c["case_id"]):c for c in selected}
        for existing in old["cases"]:
            cid=str(existing["case_id"])
            assert cid in expected and cid not in previous, "resume case identity mismatch"
            assert existing["question"]==expected[cid]["question"]
            assert existing["database_id"]==expected[cid]["database_id"]
            previous[cid]=existing
    write_atomic(outpath,output)
    app=graph()
    cache={}
    for i,case in enumerate(selected):
        row={"case_id":str(case["case_id"]),"database_id":str(case["database_id"]),
             "question":case["question"]}
        restored=previous.get(row["case_id"])
        if restored and restored["status"]=="complete":
            output["cases"].append(restored)
            write_atomic(outpath,output)
            print(f"ETSA {shard} {i+1}/25 {row['case_id']} reused_completed",flush=True)
            continue
        try:
            db_id=case["database_id"]
            if db_id not in cache:
                db=_resolve_database(Path(args.database_root),db_id)
                cache[db_id]=(db,DatabaseCatalog.from_sqlite(db))
            db,catalog=cache[db_id]
            schema,_diag=select_schema_context(catalog,Path(args.database_root),db_id,case["question"])
            evidence=case.get("evidence") or None
            g0=OllamaProvider(model=MODEL,timeout_s=args.timeout_s,capture_usage=True)
            g1=OllamaProvider(model=MODEL,timeout_s=args.timeout_s,capture_usage=True,temperature=0,num_predict=512)
            g2=OllamaProvider(model=MODEL,timeout_s=args.timeout_s,capture_usage=True,temperature=0,num_predict=512)
            t0=time.perf_counter()
            first=generate_direct_sql(case["question"],schema,g0,external_evidence=evidence)
            baseline=run_guarded(database=db,provider=g0,question=case["question"],
                                 schema_context=schema,evidence=evidence,initial_candidate=first,
                                 max_repairs=2,max_rows=50000)
            g0_elapsed=time.perf_counter()-t0
            original=baseline.candidate.sql if baseline.candidate else first.sql
            original_ok=baseline.status=="ok" and not baseline.truncated
            row.update({"initial_sql":first.sql,"g0_sql":original,
                        "g0_status":baseline.status,"g0_elapsed_s":g0_elapsed})
            if original_ok:
                t1=time.perf_counter()
                state=app.invoke(
                    {"question":case["question"],"schema_context":schema,
                     "original_sql":original,"original_rows":len(baseline.rows)},
                    context=ResearchContext(db,catalog,g1,g2),
                )
                row.update({"routed":state["routed"],"witness":state["witness"],
                            "g1_candidate":state["g1_candidate"],"g2_candidate":state["g2_candidate"],
                            "g1_sql":state["g1_sql"],"g2_sql":state["g2_sql"],
                            "g1_selected":state["g1_selected"],"g2_selected":state["g2_selected"],
                            "decisions":state["decisions"],"graph_wall_s":time.perf_counter()-t1,
                            "probe_wall_s":state["probe_wall_s"]})
            else:
                row.update({"routed":False,"witness":None,"g1_sql":original,"g2_sql":original,
                            "g1_candidate":None,"g2_candidate":None,
                            "g1_selected":False,"g2_selected":False,
                            "decisions":{"g1":"baseline_not_valid","g2":"baseline_not_valid"},
                            "graph_wall_s":0.0,"probe_wall_s":0.0})
            row["usage"]={"shared_g0":model_usage(g0),"g1_extra":model_usage(g1),"g2_extra":model_usage(g2)}
            row["status"]="complete"
        except Exception as error:
            row["status"]="error"
            row["error"]=repr(error)
        output["cases"].append(row)
        write_atomic(outpath,output)
        print(f"ETSA {shard} {i+1}/25 {row['case_id']} status={row['status']} routed={row.get('routed')}",flush=True)
    output["status"]="complete" if all(c["status"]=="complete" for c in output["cases"]) else "invalid"
    write_atomic(outpath,output)
    if output["status"]!="complete":
        raise SystemExit("unscorable shard; inspect per-case errors")


def compare_results(a,b,bag:bool) -> bool:
    if a is None or b is None or a.truncated or b.truncated:
        return False
    try:
        return Counter(a.rows)==Counter(b.rows) if bag else set(a.rows)==set(b.rows)
    except TypeError:
        return False


def score(args) -> None:
    frozen=json.loads(Path(args.input).read_text(encoding="utf-8"))
    payloads=[json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(args.shards).glob("etsa-shard-*.json"))]
    assert len(payloads)==4 and all(p["status"]=="complete" for p in payloads)
    assert all(p["cohort_sha256"]==frozen["cohort_sha256"] and p["digest"]==MODEL_DIGEST for p in payloads)
    cases=[c for p in payloads for c in p["cases"]]
    assert len(cases)==100 and len({str(c["case_id"]) for c in cases})==100
    source=json.loads(Path(args.source).read_text(encoding="utf-8"))
    labels={str(row["question_id"]):row for row in source}
    baseline={str(c["case_id"]):c for c in frozen["cases"]}
    assert set(baseline)=={str(c["case_id"]) for c in cases}
    results=[]
    for c in cases:
        case_id=str(c["case_id"]);src=labels[case_id];orig=baseline[case_id]
        assert src["db_id"]==orig["database_id"]==c["database_id"]
        assert src["question"]==orig["question"]==c["question"]
        db=_resolve_database(Path(args.database_root),c["database_id"])
        try:
            gold=execute_readonly(db,src["SQL"],max_rows=50000)
        except SQLExecutionError:
            gold=None
        row={"case_id":case_id,"db_id":c["database_id"],"routed":c["routed"],
             "g1_selected":c["g1_selected"],"g2_selected":c["g2_selected"],
             "witness_code":c["witness"]["code"] if c["witness"] else None,
             "usage":c["usage"],"g0_elapsed_s":c["g0_elapsed_s"],
             "graph_wall_s":c["graph_wall_s"],"probe_wall_s":c["probe_wall_s"]}
        row["scorable"]=gold is not None and not gold.truncated
        for arm in ("g0","g1","g2"):
            try:
                pred=execute_readonly(db,c[arm+"_sql"],max_rows=50000) if c["g0_status"]=="ok" else None
            except SQLExecutionError:
                pred=None
            row[arm+"_set"]=compare_results(pred,gold,False)
            row[arm+"_bag"]=compare_results(pred,gold,True)
        results.append(row)
    assert len(results)==100
    metrics={}
    for arm in ("g0","g1","g2"):
        metrics[arm]={"strict_set_ex":sum(int(c[arm+"_set"]) for c in results),
                      "strict_bag_ex":sum(int(c[arm+"_bag"]) for c in results)}
        if arm!="g0":
            metrics[arm]["rescues"]=sum(int(not c["g0_set"] and c[arm+"_set"]) for c in results)
            metrics[arm]["regressions"]=sum(int(c["g0_set"] and not c[arm+"_set"]) for c in results)
    def total_usage(arm):
        records=[r["usage"][arm] for r in results]
        result={"calls":sum(r["calls"] for r in records)}
        for key in ("prompt_eval_count","eval_count","total_duration"):
            values=[r[key] for r in records]
            result[key]=sum(values) if all(type(v) is int for v in values) else None
        return result
    metrics["calls_and_tokens"]={
        "g0_shared":total_usage("shared_g0"),
        "g1_extra":total_usage("g1_extra"),
        "g2_extra":total_usage("g2_extra"),
    }
    metrics["routed"]=sum(int(c["routed"]) for c in results)
    metrics["g1_selected"]=sum(int(c["g1_selected"]) for c in results)
    metrics["g2_selected"]=sum(int(c["g2_selected"]) for c in results)
    metrics["g0_total_wall_s"]=sum(r["g0_elapsed_s"] for r in results)
    metrics["extra_graph_wall_s"]=sum(r["graph_wall_s"] for r in results)
    metrics["probe_wall_s"]=sum(r["probe_wall_s"] for r in results)
    metrics["actual_usd_cost"]=None
    write_atomic(Path(args.output),{"status":"complete","cohort":"historical exact DEV100",
                                   "model_digest":MODEL_DIGEST,"metrics":metrics,"cases":results,
                                   "caveat":"Historically exposed DEV100; paired architecture benchmark, NOT generalization"})
    print(json.dumps(metrics,indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("run","score"),required=True)
    p.add_argument("--input",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--database-root",required=True)
    p.add_argument("--source")
    p.add_argument("--shards")
    p.add_argument("--shard",type=int,default=0)
    p.add_argument("--resume-from",help="Existing identical shard artifact; reuse completed cases")
    p.add_argument("--timeout-s",type=float,default=600.0)
    args=p.parse_args()
    if args.mode=="run":
        run(args)
    else:
        if not args.source or not args.shards:
            p.error("--source and --shards required to score")
        score(args)


if __name__=="__main__":
    main()
