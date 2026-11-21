"""Diagnostic only: six fixed-Qwen9B repairs on exposed, frozen original SQL."""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import time
from evaluation import qwen9b_typed_plan_exposed12 as core
from evaluation.qwen9b_typed_plan_exposed75 import _validate_full

HINTS = {
    "169": "Growth rate requested: inspect whether SQL computes a single numerical ratio rather than displaying numerator and denominator separately.",
    "298": "Percentage of molecules requested: check molecule-level counting versus atom rows multiplied by a JOIN.",
    "994": "Maximum TOTAL constructor points requested: check aggregation by constructor before ranking.",
}
CONTROL_ID = "755"
PROTO = "EXPOSED_QWEN9B_ONE_SHOT_REPAIR_SIX_V1"

def select(frozen):
    f = json.loads(frozen.read_text())
    _validate_full(f)
    rows = {str(c["case_id"]):c for c in f["selected"]}
    required = list(HINTS) + [CONTROL_ID]
    if any(i not in rows for i in required):
        raise ValueError("historical frozen 75 missing case IDs: "+str([i for i in required if i not in rows]))
    pool = [c for c in f["selected"] if str(c["case_id"]) not in required and c.get("historical_guarded_sql")]
    pool.sort(key=lambda c: sha256(("unlabeled-control|"+str(c["case_id"])).encode()).hexdigest())
    result = [rows[x] for x in required] + pool[:2]
    assert len(result)==6 and len({x["case_id"] for x in result})==6
    return result

def sql_only(response):
    s=response.strip()
    fence=chr(96)*3
    if s.startswith(fence) and s.endswith(fence):
        s="\n".join(s.splitlines()[1:-1]).strip()
    return s.rstrip(";").strip()+";"

def infer(frozen, db_root, out):
    from semantic_sql.catalog import DatabaseCatalog
    from semantic_sql.providers import OllamaProvider
    from semantic_sql.verifier import verify_sql_preflight
    from evaluation.run_bird import _resolve_database
    from evaluation.schema_context import select_schema_context
    cases=select(frozen)
    if core._model_digest(core.QWEN_MODEL)!=core.QWEN_DIGEST:
        raise ValueError("model digest drift")
    provider=OllamaProvider(model=core.QWEN_MODEL,timeout_s=600,temperature=0,seed=42,num_predict=768)
    payload={"protocol":PROTO,"status":"PARTIAL","no_reference_sql_in_inference":True,
             "model_digest":core.QWEN_DIGEST,"source_sha256":core.DEV100_SOURCE_SHA,"cases":[]}
    core.write_exclusive(out,payload)
    cache={}
    for case in cases:
        cid=str(case["case_id"]); dbid=case["db_id"]; original=case["historical_guarded_sql"] or ""
        if dbid not in cache:
            db=_resolve_database(db_root,dbid)
            cache[dbid]=(db,DatabaseCatalog.from_sqlite(db))
        db,catalog=cache[dbid]
        before=bool(original and verify_sql_preflight(db,original).ok)
        final=original
        decision="initial_invalid"
        call_s=0.0
        if before:
            schema,_=select_schema_context(catalog,db_root,dbid,case["question"])
            hint=HINTS.get(cid,"No specific semantic contradiction established; preserve original SQL unless a direct mismatch is proven.")
            user=("Question: "+case["question"]+"\nEvidence: "+(case.get("evidence") or "none")
                  +"\nSchema: "+str(schema)[:14000]+"\nOriginal SQL: "+original
                  +"\nSpecific issue to check: "+hint
                  +"\nReturn ONE executable SQLite SELECT only. If SQL is already correct, repeat it exactly. Preserve other conditions.")
            start=time.monotonic()
            answer=provider.complete_text(system="Review SQL semantics. Output SQL only, no reasoning or prose.",
                                          user=user)
            call_s=round(time.monotonic()-start,2)
            proposed=sql_only(answer)
            valid=bool(verify_sql_preflight(db,proposed).ok)
            if valid:
                final=proposed
                decision="same" if proposed.strip().rstrip(";")==original.strip().rstrip(";") else "repaired"
            else:
                decision="invalid_fallback"
        payload["cases"].append({
            "case_id":cid,"db_id":dbid,"original_sql":original,"final_sql":final,
            "decision":decision,"model_calls":int(before),"latency_s":call_s,
            "hinted":cid in HINTS,"question_sha256":sha256(case["question"].encode()).hexdigest()
        })
        out.write_text(json.dumps(payload,indent=2,sort_keys=True))
        print("REPAIR_SIX_BLIND",cid,decision,"calls",int(before),flush=True)
    payload["status"]="SEALED_GOLD_BLIND_SIX"
    out.write_text(json.dumps(payload,indent=2,sort_keys=True))
    print("REPAIR_SIX_BLIND_SEALED",len(payload["cases"]),flush=True)

def score(frozen, blind, source, db_root, out):
    from semantic_sql.execution import execute_readonly, SQLExecutionError
    from evaluation.run_bird import _resolve_database
    cases=select(frozen)
    b=json.loads(blind.read_text())
    if (b.get("status")!="SEALED_GOLD_BLIND_SIX" or b.get("protocol")!=PROTO
        or b.get("model_digest")!=core.QWEN_DIGEST
        or [str(c["case_id"]) for c in cases]!=[c["case_id"] for c in b["cases"]]):
        raise ValueError("cannot open reference before full blind inference")
    labels={core._id(row,i):row for i,row in enumerate(core._load_source(source))}
    results=[]
    for c,d in zip(cases,b["cases"]):
        label=labels[d["case_id"]]
        if label["question"]!=c["question"] or str(label["db_id"])!=d["db_id"]:
            raise ValueError("question DB drift")
        db=_resolve_database(db_root,d["db_id"])
        def execute(sql):
            try:
                value=execute_readonly(db,sql,max_rows=50000)
                return None if value.truncated else value
            except (SQLExecutionError,ValueError,TypeError):
                return None
        gold=execute(str(label["SQL"]))
        original=execute(d["original_sql"])
        repaired=execute(d["final_sql"])
        def metrics(pred):
            return {"set_ex":bool(gold is not None and pred is not None and set(gold.rows)==set(pred.rows)),
                    "bag_ex":bool(gold is not None and pred is not None and Counter(gold.rows)==Counter(pred.rows))}
        results.append({"case_id":d["case_id"],"db":d["db_id"],"scorable":gold is not None,
                        "decision":d["decision"],"hinted":d["hinted"],
                        "original":metrics(original),"one_shot":metrics(repaired)})
    usable=[x for x in results if x["scorable"]]
    report={"protocol":PROTO,"status":"EXPLORATORY_COMPLETE","n":len(results),
            "n_scorable":len(usable),
            "calls":sum(x["model_calls"] for x in b["cases"]),
            "model_latency_s":round(sum(x["latency_s"] for x in b["cases"]),2),
            "original_set":sum(x["original"]["set_ex"] for x in usable),
            "one_shot_set":sum(x["one_shot"]["set_ex"] for x in usable),
            "original_bag":sum(x["original"]["bag_ex"] for x in usable),
            "one_shot_bag":sum(x["one_shot"]["bag_ex"] for x in usable),
            "rescues":sum(x["one_shot"]["set_ex"] and not x["original"]["set_ex"] for x in usable),
            "regressions":sum(x["original"]["set_ex"] and not x["one_shot"]["set_ex"] for x in usable),
            "cases":results,"historically_exposed":True,"performance_promotion_allowed":False,
            "gold_opened_only_after_sealed_inference":True,"sql_training":False,"holdout_untouched":True}
    core.write_exclusive(out,report)
    print("REPAIR_SIX_PAIRED_STRICT_EX",json.dumps(report,sort_keys=True),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=("infer","score"))
    p.add_argument("--frozen",type=Path,required=True)
    p.add_argument("--db-root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--blind",type=Path)
    p.add_argument("--source",type=Path)
    a=p.parse_args()
    if a.phase=="infer":infer(a.frozen,a.db_root,a.out)
    else:
        if a.blind is None or a.source is None:raise ValueError("score needs blind + source")
        score(a.frozen,a.blind,a.source,a.db_root,a.out)
if __name__=="__main__":main()
