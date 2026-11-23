"""Source-blind, equal-call-budget DEV100 research: Direct+review / Guarded / Semantic LangGraph.

100 sealed first SQLs shared across arms. Baseline B used 27 repair calls.
Allocate exactly 27 extra model calls to each of A and C, without gold labels.
A and C review identical cases; only C receives physical semantic evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import time
from typing import TypedDict

from langgraph.graph import START, END, StateGraph

from evaluation.run_bird import _ollama_model_digest, _resolve_database
from evaluation.schema_context import select_schema_context
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.direct import _strip_fence
from semantic_sql.execution import SQLExecutionError, execute_readonly
from semantic_sql.providers import OllamaProvider
from semantic_sql.verifier import verify_sql_preflight

MODEL = "qwen3.5:9b-q4_K_M"
DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
SOURCE_SHA = "ffd8018378ddb1a8794753e0a31cfc81862ff7318a5184c22f3dc4ce03a03feb"
EXTRA = 27
INITIAL = 100


def read_shards(shards: Path, frozen: dict) -> dict[str, dict]:
    expected = {str(c["case_id"]): c for c in frozen["cases"]}
    assert len(expected) == INITIAL
    by_id: dict[str, dict] = {}
    for shard in range(4):
        obj = json.loads((shards / f"etsa-shard-{shard}.json").read_text())
        assert obj["status"] == "complete" and obj["shard"] == shard
        assert obj["digest"] == DIGEST and obj["cohort_sha256"] == frozen["cohort_sha256"]
        for case in obj["cases"]:
            key = str(case["case_id"])
            assert key in expected and key not in by_id and case["status"] == "complete"
            assert (case["database_id"], case["question"]) == (
                expected[key]["database_id"], expected[key]["question"])
            by_id[key] = case
    assert set(by_id) == set(expected)
    return by_id


def semantic_witness(question: str, sql: str, catalog: DatabaseCatalog) -> dict:
    """Priority is fixed by source-only features; no gold, EX, or case-ID targeting."""
    q, s = question.casefold(), sql.casefold()
    # Named entity quoted by the user but absent from any SQL filter.
    quoted = re.findall(r'["“]([^"”]{3,80})["”]', question)
    missing = [v for v in quoted if v.casefold().replace("'", "''") not in s and v.casefold() not in s]
    if missing:
        return {"kind": "NAMED_ENTITY_FILTER", "priority": 90,
                "detail": "Quoted entity absent from SQL predicates", "entity": missing[0]}
    if (re.search(r"\b(how many|number of)\b", q)
            and re.search(r"\bcount\s*\(\s*\*\s*\)", s)
            and re.search(r"\bjoin\b", s)):
        return {"kind": "JOIN_MULTIPLICITY", "priority": 85,
                "detail": "COUNT(*) across JOIN may count child rows, not requested entities",
                "foreign_keys": ["%s.%s -> %s.%s" % e for e in catalog.foreign_keys[:16]]}
    if (re.search(r"\b(each|per|for every)\b", q)
            and re.search(r"\b(count|sum|avg)\s*\(", s)
            and not re.search(r"\bgroup\s+by\b|\bover\s*\(", s)):
        return {"kind": "GROUP_GRAIN", "priority": 80,
                "detail": "Per-entity aggregation may require grouping"}
    if (re.search(r"\b(percent|percentage|ratio|rate of)\b", q)
            and not re.search(r"/|\bavg\s*\(", s)):
        return {"kind": "RATIO_MEASURE", "priority": 75,
                "detail": "Asked ratio/percentage but candidate lacks visible division/AVG"}
    if (re.search(r"\b(highest|lowest|top\s+\d+|most common|least common)\b", q)
            and not re.search(r"\border\s+by\b|\b(max|min)\s*\(", s)):
        return {"kind": "TOPK_ORDER", "priority": 70,
                "detail": "Question requires a ranking; query has no ORDER BY or extremum"}
    if (re.search(r"\b(both|all of|every)\b", q)
            and re.search(r"\bin\s*\([^)]*,", s)
            and not re.search(r"\bintersect\b|\bhaving\b|\bexists\b", s)):
        return {"kind": "ALL_OF", "priority": 65, "detail": "Possible all-of interpreted as any-of"}
    # Lower-confidence fallback is explicit, not falsely claimed as proof of error.
    joins = len(re.findall(r"\bjoin\b", s))
    complexity = joins * 2 + (2 if re.search(r"\bgroup\s+by\b", s) else 0)
    complexity += (1 if re.search(r"\bselect\b.*\bselect\b", s, re.S) else 0)
    return {"kind": "COMPLEX_SQL_UNCERTAIN", "priority": min(40, 10 + complexity),
            "detail": "No demonstrated semantic contradiction; bounded exploratory review"}


def plan(frozen: dict, shards: dict[str, dict], root: Path) -> dict:
    records = []
    cache = {}
    for c in frozen["cases"]:
        key, db_id = str(c["case_id"]), c["database_id"]
        row = shards[key]
        if db_id not in cache:
            db = _resolve_database(root, db_id)
            cache[db_id] = (db, DatabaseCatalog.from_sqlite(db))
        db, catalog = cache[db_id]
        base = row["initial_sql"]
        invalid = not verify_sql_preflight(db, base).ok
        witness = ({"kind": "COMPILE_REPAIR", "priority": 100,
                    "detail": "Initial query fails deterministic SQLite preflight"}
                   if invalid else semantic_witness(c["question"], base, catalog))
        records.append({"case_id": key, "database_id": db_id,
                        "question": c["question"], "initial_sql": base,
                        "evidence": c.get("evidence") or "", "witness": witness,
                        "invalid_initial": invalid})
    # Source-only ranking frozen before any reference SQL is downloaded.
    selected = sorted(records, key=lambda x: (-x["witness"]["priority"],
                                            sha256(x["case_id"].encode()).hexdigest()))[:EXTRA]
    selected_ids = {c["case_id"] for c in selected}
    assert len(selected_ids) == EXTRA
    return {"status": "frozen_gold_free", "model_digest": DIGEST,
            "cohort_sha256": frozen["cohort_sha256"], "equal_budget_total_calls": INITIAL+EXTRA,
            "extra_calls_per_arm": EXTRA,
            "cases": [{**c, "selected": c["case_id"] in selected_ids} for c in records]}


def resolved(candidate: str, witness: dict) -> bool:
    s = candidate.casefold()
    kind = witness["kind"]
    if kind == "NAMED_ENTITY_FILTER":
        return witness["entity"].casefold() in s
    if kind == "JOIN_MULTIPLICITY":
        return bool(re.search(r"\bcount\s*\(\s*distinct\b", s))
    if kind == "GROUP_GRAIN":
        return bool(re.search(r"\bgroup\s+by\b|\bover\s*\(", s))
    if kind == "RATIO_MEASURE":
        return bool(re.search(r"/|\bavg\s*\(", s))
    if kind == "TOPK_ORDER":
        return bool(re.search(r"\border\s+by\b|\b(max|min)\s*\(", s))
    if kind == "ALL_OF":
        return bool(re.search(r"\bintersect\b|\bhaving\b|\bexists\b", s))
    return True  # exploratory candidate still needs read-only preflight + execution


def choose(db: Path, old: str, alt: str, witness: dict) -> tuple[str, str]:
    if not alt or alt.strip().rstrip(";") == old.strip().rstrip(";"):
        return old, "unchanged"
    if not resolved(alt, witness):
        return old, "source_obligation_unresolved"
    if not verify_sql_preflight(db, alt).ok:
        return old, "invalid_readonly_or_compile"
    try:
        result = execute_readonly(db, alt, max_rows=50000)
        if result.truncated:
            return old, "truncated_result"
    except SQLExecutionError:
        return old, "failed_execution"
    return alt, "selected_gold_blind"


class GraphState(TypedDict, total=False):
    question: str
    schema: str
    evidence: str
    sql: str
    witness: dict
    candidate: str
    final_sql: str
    why: str


def _route(st: GraphState) -> str:
    return "syntax" if st["witness"]["kind"] == "COMPILE_REPAIR" else "semantic"


def _syntax(st: GraphState, runtime) -> dict:
    return {"candidate": _strip_fence(runtime.context["provider"].complete_text(
        system="Repair the SQLite compiler/preflight failure in the following SQL. Return one read-only SQL query only.",
        user=_prompt(st)))}


def _semantic(st: GraphState, runtime) -> dict:
    return {"candidate": _strip_fence(runtime.context["provider"].complete_text(
        system="Diagnose possible semantic mismatch using question, physical schema and witness. The witness is a hypothesis, not ground truth. Revise only if warranted. Return one read-only SQL query only.",
        user=_prompt(st) + "\nSource-only semantic obligation:\n" + json.dumps(st["witness"])))}


def _prompt(st: GraphState) -> str:
    return ("Question:\n" + st["question"] + "\nSchema:\n" + st["schema"]
            + "\nExternal evidence:\n" + st["evidence"]
            + "\nOriginal SQL:\n" + st["sql"] + "\nReturn SQL only.")


def _validate(st: GraphState, runtime) -> dict:
    sql, why = choose(runtime.context["database"], st["sql"], st["candidate"], st["witness"])
    return {"final_sql": sql, "why": why}


def build_graph():
    graph = StateGraph(GraphState, context_schema=dict)
    graph.add_node("syntax_repair", _syntax)
    graph.add_node("semantic_evidence", _semantic)
    graph.add_node("validate_and_fallback", _validate)
    graph.add_conditional_edges(START, _route,
        {"syntax": "syntax_repair", "semantic": "semantic_evidence"})
    graph.add_edge("syntax_repair", "validate_and_fallback")
    graph.add_edge("semantic_evidence", "validate_and_fallback")
    graph.add_edge("validate_and_fallback", END)
    return graph.compile()


def run(plan_data: dict, root: Path, shard: int, out: Path, *, timeout: float) -> None:
    assert plan_data["extra_calls_per_arm"] == EXTRA and 0 <= shard < 4
    assert _ollama_model_digest("http://localhost:11434", MODEL) == DIGEST
    selected = [c for c in plan_data["cases"] if c["selected"]]
    assert len(selected) == EXTRA
    graph = build_graph()
    cache = {}
    output = {"status": "incomplete", "shard": shard, "model_digest": DIGEST,
              "cohort_sha256": plan_data["cohort_sha256"], "cases": []}
    out.parent.mkdir(parents=True, exist_ok=True)
    for index, case in enumerate(selected):
        if index % 4 != shard:
            continue
        db_id = case["database_id"]
        if db_id not in cache:
            db = _resolve_database(root, db_id)
            cache[db_id] = (db, DatabaseCatalog.from_sqlite(db))
        db, catalog = cache[db_id]
        schema, _ = select_schema_context(catalog, root, db_id, case["question"])
        result = {"case_id": case["case_id"], "status": "incomplete"}
        try:
            # One generic call and one targeted LangGraph call, same case & tokens ceiling.
            generic = OllamaProvider(model=MODEL, timeout_s=timeout,
                                     temperature=0, num_predict=512, capture_usage=True)
            targeted = OllamaProvider(model=MODEL, timeout_s=timeout,
                                      temperature=0, num_predict=512, capture_usage=True)
            state = {"question": case["question"], "schema": schema, "sql": case["initial_sql"],
                     "evidence": case["evidence"], "witness": case["witness"]}
            t0 = time.monotonic()
            alt = _strip_fence(generic.complete_text(
                system="Independently reconsider this SQL to answer the question; produce a corrected read-only SQLite query, SQL only.",
                user=_prompt(state)))
            asql, awhy = choose(db, case["initial_sql"], alt, case["witness"])
            a_wall = time.monotonic()-t0
            t1 = time.monotonic()
            new = graph.invoke(state, context={"provider": targeted, "database": db})
            c_wall = time.monotonic()-t1
            assert len(generic.usage_records) == len(targeted.usage_records) == 1
            result.update(status="complete", a_sql=asql, c_sql=new["final_sql"],
                          a_decision=awhy, c_decision=new["why"],
                          a_wall_s=a_wall, c_wall_s=c_wall,
                          a_usage=generic.usage_records[0], c_usage=targeted.usage_records[0])
        except Exception as exc:
            result["status"] = "error"
            result["error"] = repr(exc)
        output["cases"].append(result)
        out.write_text(json.dumps(output,indent=2,ensure_ascii=False))
        print(f"EQUAL BUDGET shard={shard} case={case['case_id']} {result['status']}",flush=True)
    output["status"] = "complete" if all(c["status"] == "complete" for c in output["cases"]) else "invalid"
    out.write_text(json.dumps(output,indent=2,ensure_ascii=False))
    if output["status"] != "complete":
        raise SystemExit("failed shard; preserve artifact and recover only incomplete calls")


def _execute(db: Path, sql: str | None):
    if not sql:
        return None
    try:
        output = execute_readonly(db, sql, max_rows=50000)
        return None if output.truncated else output
    except SQLExecutionError:
        return None


def _correct(pred, gold, bag: bool) -> bool:
    if pred is None or gold is None:
        return False
    return (Counter(pred.rows) == Counter(gold.rows)) if bag else (set(pred.rows) == set(gold.rows))


def score(plan_data: dict, frozen: dict, old: dict[str,dict], results: Path,
          source: Path, root: Path) -> dict:
    assert sha256(source.read_bytes()).hexdigest() == SOURCE_SHA
    source_rows = {str(r["question_id"]): r for r in json.loads(source.read_text())}
    extras = {}
    for shard in range(4):
        data = json.loads((results/f"budget-shard-{shard}.json").read_text())
        assert data["status"] == "complete" and data["shard"] == shard
        assert data["cohort_sha256"] == frozen["cohort_sha256"] and data["model_digest"] == DIGEST
        for row in data["cases"]:
            assert row["case_id"] not in extras and row["status"] == "complete"
            extras[row["case_id"]] = row
    expected = {c["case_id"] for c in plan_data["cases"] if c["selected"]}
    assert len(expected) == EXTRA and expected == set(extras)
    assert plan_data["cohort_sha256"] == frozen["cohort_sha256"]
    cases = []
    for c in plan_data["cases"]:
        key = c["case_id"]
        old_case = old[key]
        gold_case = source_rows[key]
        assert gold_case["db_id"] == c["database_id"] and gold_case["question"] == c["question"]
        db = _resolve_database(root,c["database_id"])
        gold = _execute(db,gold_case["SQL"])
        assert gold is not None, "unscorable publisher SQL"
        extra = extras.get(key)
        paths = {"direct": c["initial_sql"],
                 "a_budget": extra["a_sql"] if extra else c["initial_sql"],
                 "workflow": old_case["g0_sql"] if old_case["g0_status"] == "ok" else None,
                 "graph": extra["c_sql"] if extra else c["initial_sql"]}
        row = {"case_id": key, "selected": bool(extra),
               "kind": c["witness"]["kind"], "results": {}}
        for arm, sql in paths.items():
            actual = _execute(db,sql)
            row["results"][arm] = {"set": _correct(actual,gold,False),
                                   "bag": _correct(actual,gold,True),
                                   "executed": actual is not None}
        cases.append(row)
    arms = ("direct","a_budget","workflow","graph")
    def changes(a,b,metric):
        return {"rescues":sum(not c["results"][a][metric] and c["results"][b][metric] for c in cases),
                "regressions":sum(c["results"][a][metric] and not c["results"][b][metric] for c in cases)}
    counts = {arm: {k: sum(c["results"][arm][k] for c in cases)
                    for k in ("set","bag","executed")} for arm in arms}
    tokens = {arm: {k:sum(x[arm+"_usage"][k] for x in extras.values())
                    if all(isinstance(x[arm+"_usage"].get(k),int) for x in extras.values()) else None
                    for k in ("prompt_eval_count","eval_count","total_duration")}
              for arm in ("a","c")}
    assert sum(int(r["usage"]["shared_g0"]["calls"]) for r in old.values()) == 127
    result={"status":"complete","cohort_sha256":frozen["cohort_sha256"],
            "model_digest":DIGEST,"cases":cases,
            "metrics":{"strict_ex":counts,
                "graph_vs_workflow_set":changes("workflow","graph","set"),
                "graph_vs_workflow_bag":changes("workflow","graph","bag"),
                "graph_vs_a_budget_set":changes("a_budget","graph","set"),
                "calls":{"direct":100,"a_budget":127,"workflow":127,"graph":127},
                "additional_tokens":tokens,
                "extra_a_wall_s":sum(v["a_wall_s"] for v in extras.values()),
                "extra_c_wall_s":sum(v["c_wall_s"] for v in extras.values()),
                "actual_usd_cost":None}}
    assert len(cases)==100
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=["plan","run","score"],required=True)
    p.add_argument("--frozen",type=Path,required=True)
    p.add_argument("--shards",type=Path)
    p.add_argument("--plan",type=Path)
    p.add_argument("--results",type=Path)
    p.add_argument("--source",type=Path)
    p.add_argument("--db-root",type=Path,required=True)
    p.add_argument("--shard",type=int,default=0)
    p.add_argument("--timeout-s",type=float,default=900)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    frozen=json.loads(args.frozen.read_text())
    assert frozen["status"]=="frozen_gold_free" and frozen["model_digest"]==DIGEST
    assert len(frozen["cases"])==INITIAL and len({c["case_id"] for c in frozen["cases"]})==INITIAL
    if args.mode == "plan":
        result=plan(frozen,read_shards(args.shards,frozen),args.db_root)
    elif args.mode == "run":
        run(json.loads(args.plan.read_text()),args.db_root,args.shard,args.output,timeout=args.timeout_s)
        return
    else:
        result=score(json.loads(args.plan.read_text()),frozen,read_shards(args.shards,frozen),
                     args.results,args.source,args.db_root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result.get("metrics",{"selected":sum(c["selected"] for c in result["cases"]),
            "syntax":sum(c["selected"] and c["invalid_initial"] for c in result["cases"])}),indent=2))


if __name__=="__main__":
    main()
