"""Original fixed-Qwen BIRD TRAIN VAL64 gold-blind physical literal counterexamples.

One narrow mechanism, frozen BEFORE model-outcome physical scoring:
- original Qwen SQL has WHERE col = 'literal';
- model user question/external evidence explicitly contains that literal;
- physical SQLite column has no exact match, but exactly ONE canonical
  Unicode/case/punctuation equivalent stored string;
- SQL is a simple single statement with unambiguous resolved column.
This proposes one read-only alternative SQL without any LLM call.
Publisher reference SQL/outcome is ONLY touched after proposal is immutable.

Research validation is historical, source-DB-separated from prior TRAIN16
physical fanout experiment, NOT unseen prospective BIRD DEV or Kaggle result.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from evaluation.bounded_shard7_probe import bounded_outcome
from evaluation.kcache_protocol import MODEL_DIGEST, sample_correctness
from evaluation.physical_literal_counterexample import propose_literal_counterexample

TRAIN_VAL64_SHA = "cc112305bc9de33a03cea704f2edecfdfb45e62cc953ae498cea27fbed93b506"
PROTOCOL = "BIRD-TRAIN-VAL64-ORIGINAL-FIXED-QWEN9B-UNIQUE-PHYSICAL-LITERAL-COUNTEREXAMPLE-V1"
DBS = ("movie_3", "music_tracker", "superstore", "world")
TIMEOUT = 8.
GATE = {
    "min_guarded_changed": 3,
    "min_guarded_rescues": 2,
    "max_guarded_regressions": 0,
    "min_guarded_rescue_databases": 2,
    "max_original_bounded_score_disagreement": 0,
}


def original_rows(path: Path) -> list[dict]:
    data = path.read_bytes()
    if sha256(data).hexdigest() != TRAIN_VAL64_SHA:
        raise ValueError("original BIRD TRAIN-VAL64 source SHA drift")
    rows = [json.loads(s) for s in data.decode("utf-8").splitlines() if s.strip()]
    if len(rows) != 64 or len({r["source_index"] for r in rows}) != 64:
        raise ValueError("must use every original 64 publisher source question exactly once")
    if tuple(sorted({r["source_db_id"] for r in rows})) != DBS:
        raise ValueError("unexpected original four database source identities")
    for r in rows:
        m = r["messages"]
        if ([x["role"] for x in m] != ["system", "user", "assistant"]
                or sha256(m[1]["content"].encode()).hexdigest() != r["input_sha256"]):
            raise ValueError("publisher model prompt/label linkage drift")
    return rows


def original_model(path: Path) -> dict:
    obj = json.loads(path.read_text("utf-8"))
    if (obj.get("status") != "complete" or obj.get("n") != 64
            or obj.get("source_validation_sha256") != TRAIN_VAL64_SHA
            or obj.get("model_digest") != MODEL_DIGEST
            or obj.get("direct_ex") != 27 or obj.get("guarded_ex") != 31
            or len(obj.get("cases", [])) != 64):
        raise ValueError("original real Qwen9B source/score/base digest drift")
    return obj


def worker(source: Path, model: Path, database: Path, shard: int) -> dict:
    if shard not in range(4):
        raise ValueError("invalid four-database BIRD TRAIN source worker")
    rows = original_rows(source)
    baseline = original_model(model)
    db_name = DBS[shard]
    if database.name != db_name + ".sqlite" or not database.is_file():
        raise ValueError("wrong original physical SQLite file")
    index = {r["source_index"]: r for r in rows}
    original_cases = {c["source_index"]: c for c in baseline["cases"]}
    if len(original_cases) != 64 or set(index) != set(original_cases):
        raise ValueError("original real Qwen cases do not cover exact source")
    chosen = [r for r in rows if r["source_db_id"] == db_name]
    if len(chosen) != 16:
        raise ValueError("original database cohort must contain exactly 16 cases")
    results = []
    for row in chosen:
        c = original_cases[row["source_index"]]
        user = row["messages"][1]["content"]
        if (c["db_id"] != db_name
                or c["prompt_sha256"] != sha256(user.encode()).hexdigest()):
            raise ValueError("original model candidate prompt identity drift")
        outcomes = {}
        for arm in ("direct", "guarded"):
            sql = c[arm + "_sql"]
            t0 = perf_counter()
            # Critically, propose() has NO original publisher gold SQL/outcome.
            proposal = propose_literal_counterexample(database, user, sql)
            elapsed = round(perf_counter()-t0,4)
            result = {
                "changed": proposal.changed,
                "reason": proposal.reason,
                "check_wall_s": elapsed,
                "original_sql_sha256": sha256(sql.encode()).hexdigest(),
                "proposed_sql_sha256": sha256(proposal.sql.encode()).hexdigest(),
                "original_correct_immutable_label": bool(c[arm]["ex"]),
                "original_executable_immutable_label": bool(c[arm]["execution_ok"]),
                "base_table": proposal.base_table if proposal.changed else None,
                "column": proposal.column if proposal.changed else None,
            }
            if proposal.changed:
                # Score AFTER immutable source-only proposal. Bounded 8s
                # SQLite protocol can differ from original unbounded original
                # VAL64 label; we explicitly censor rather than silently mix.
                baseline_out = bounded_outcome(database, sql, TIMEOUT)
                proposed_out = bounded_outcome(database, proposal.sql, TIMEOUT)
                gold = bounded_outcome(database, row["messages"][2]["content"], TIMEOUT)
                if not gold["valid"]:
                    result.update({"gold_scorable": False,
                                   "base_scorable": baseline_out["valid"],
                                   "proposed_scorable": proposed_out["valid"]})
                else:
                    base_correct = sample_correctness(baseline_out, gold)
                    new_correct = sample_correctness(proposed_out, gold)
                    result.update({
                        "gold_scorable": True,
                        "base_scorable": baseline_out["valid"],
                        "proposed_scorable": proposed_out["valid"],
                        "base_bounded_correct": base_correct,
                        "proposed_bounded_correct": new_correct,
                        "original_bounded_label_disagrees": (
                            base_correct != bool(c[arm]["ex"])
                        ),
                    })
            outcomes[arm] = result
        results.append({
            "source_index": row["source_index"],
            "db_id": db_name,
            "direct": outcomes["direct"],
            "guarded": outcomes["guarded"],
        })
        print("GOLDBLIND_PHYSICAL_LITERAL_QWEN64",db_name,len(results),"/16",
              "direct_changed",outcomes["direct"]["changed"],
              "guarded_changed",outcomes["guarded"]["changed"],flush=True)
    return {
        "status":"PHYSICAL_COUNTEREXAMPLE_SOURCE64_WORKER_COMPLETE",
        "protocol":PROTOCOL,
        "source_sha256":TRAIN_VAL64_SHA,
        "model_digest":MODEL_DIGEST,
        "db_name":db_name,
        "shard":shard,
        "physical_SQL_seconds":TIMEOUT,
        "n":len(results),
        "new_original_Qwen_calls":0,
        "no_training":True,
        "DEV500_used":False,
        "HOLDOUT300_used":False,
        "cases":results,
    }


def aggregate(parts: list[dict], source: Path) -> dict:
    src=original_rows(source)
    if len(parts)!=4 or sorted(p.get("shard") for p in parts)!=list(range(4)):
        raise ValueError("missing or duplicated original physical source worker")
    cases=[]
    for p in parts:
        if (p.get("status")!="PHYSICAL_COUNTEREXAMPLE_SOURCE64_WORKER_COMPLETE"
                or p.get("protocol")!=PROTOCOL
                or p.get("source_sha256")!=TRAIN_VAL64_SHA
                or p.get("model_digest")!=MODEL_DIGEST
                or p.get("db_name")!=DBS[p["shard"]]
                or p.get("physical_SQL_seconds")!=TIMEOUT
                or p.get("n")!=16 or len(p.get("cases",[]))!=16
                or p.get("new_original_Qwen_calls")!=0
                or p.get("DEV500_used") is not False
                or p.get("HOLDOUT300_used") is not False):
            raise ValueError("original model/source/benchmark protocol drift")
        cases.extend(p["cases"])
    expected={r["source_index"] for r in src}
    if len(cases)!=64 or {x["source_index"] for x in cases}!=expected:
        raise ValueError("missing or duplicated original validation64 cases")
    arm_reports={}
    gate_blockers=[]
    for arm in ("direct","guarded"):
        counters=Counter()
        by_reason=Counter()
        rescues=set()
        loss=set()
        for row in cases:
            x=row[arm]
            by_reason[x["reason"]]+=1
            if not x["changed"]:
                continue
            counters["changed"]+=1
            if x.get("gold_scorable") is not True:
                counters["gold_censored"]+=1
                continue
            if x.get("original_bounded_label_disagrees"):
                counters["original_score_mismatch"]+=1
                continue
            if x["proposed_bounded_correct"] and not x["base_bounded_correct"]:
                counters["rescues"]+=1
                rescues.add(row["db_id"])
            elif x["base_bounded_correct"] and not x["proposed_bounded_correct"]:
                counters["regressions"]+=1
                loss.add(row["db_id"])
            else:
                counters["unchanged_correctness"]+=1
            if not x["proposed_scorable"]:
                counters["proposed_nonexecuting"]+=1
        if counters["gold_censored"] or counters["original_score_mismatch"]:
            gate_blockers.append(f"{arm}_censored_or_incomparable")
        arm_reports[arm]={
            "original_correct_out_of_64": 27 if arm=="direct" else 31,
            "changed":counters["changed"],
            "gold_censored":counters["gold_censored"],
            "original_score_mismatch":counters["original_score_mismatch"],
            "rescues":counters["rescues"],
            "regressions":counters["regressions"],
            "net":counters["rescues"]-counters["regressions"],
            "rescue_databases":len(rescues),
            "regression_databases":len(loss),
            "unchanged_correctness":counters["unchanged_correctness"],
            "proposed_nonexecuting":counters["proposed_nonexecuting"],
            "reasons":dict(sorted(by_reason.items())),
            "source_local_treatment_correct_out_of_64":(
                (27 if arm=="direct" else 31)
                + counters["rescues"] - counters["regressions"]
                if not counters["gold_censored"] and not counters["original_score_mismatch"]
                else None
            ),
        }
    g=arm_reports["guarded"]
    passes=(
        not gate_blockers
        and g["changed"]>=GATE["min_guarded_changed"]
        and g["rescues"]>=GATE["min_guarded_rescues"]
        and g["regressions"]<=GATE["max_guarded_regressions"]
        and g["rescue_databases"]>=GATE["min_guarded_rescue_databases"]
        and g["original_score_mismatch"]<=GATE["max_original_bounded_score_disagreement"]
    )
    return {
        "status":"COMPLETE_FIXED_QWEN64_PHYSICAL_LITERAL_SCREEN",
        "protocol":PROTOCOL,
        "model_digest":MODEL_DIGEST,
        "source_sha256":TRAIN_VAL64_SHA,
        "original_real_model_n":64,
        "source_dbs":list(DBS),
        "source_count":64,
        "physical_SQL_seconds":TIMEOUT,
        "real_model_calls":0,
        "real_model_training_steps":0,
        "DEV500_used":False,
        "HOLDOUT300_used":False,
        "goldblind_rule_code_uses_no_reference_SQL":True,
        "direct":arm_reports["direct"],
        "guarded":g,
        "positive_mechanism_gate":passes,
        "predeclared_gate":GATE,
        "gate_blockers":gate_blockers,
        "claim_boundary":(
            "Original physical BIRD TRAIN VAL64 Qwen3.5-9B predictions, not new "
            "model generations, not DEV500 nor heldout generalization. Only a "
            "conservative one-literal deterministic alternative was proposed "
            "using real SQLite values, no gold. Source-publisher EX may contain "
            "reference conflicts; bounded 8-second replay may differ from "
            "historical original unbounded source. Never use viewed outcomes "
            "to tune this rule. Positive screen still needs untouched source."
        ),
    }


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("score","aggregate"))
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--model",type=Path)
    p.add_argument("--database",type=Path)
    p.add_argument("--shard",type=int)
    p.add_argument("--inputs",type=Path,nargs="+")
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():
        raise FileExistsError("no overwriting immutable original-source evidence")
    if a.mode=="score":
        if a.model is None or a.database is None or a.shard is None:
            p.error("score needs original model, physical DB, shard")
        result=worker(a.source,a.model,a.database,a.shard)
    else:
        if not a.inputs:
            p.error("aggregate needs exactly four original workers")
        result=aggregate([json.loads(x.read_text()) for x in a.inputs],a.source)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("ORIGINAL_FIXED_QWEN_VAL64_UNIQUE_LITERAL_COUNTEREXAMPLE",
          json.dumps({k:v for k,v in result.items() if k!="cases"},sort_keys=True),
          flush=True)


if __name__=="__main__":
    main()
