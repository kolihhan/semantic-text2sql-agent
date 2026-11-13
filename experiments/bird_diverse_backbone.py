"""BIRD SQL candidate *backbone* diversity: XiYan 7B vs Qwen2.5-Coder 7B.

Same public BIRD DEV questions, same lexical table selection, same complete
M-Schema, same template and evidence, different independently trained backbone.
Reuse frozen XiYan A0 SQL from run 43372535238; no reference labels, old
A1 proposals or A0 correctness labels are accessible in model generation.
Measure candidate-set oracle BEFORE attempting a selector, on this exploratory
cohort only. No leaderboard, blind generalization or automatic promotion claim.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from evaluation.bird_loader import load_bird_json
from evaluation.schema_context import _description_rows, select_schema_context
from experiments.xiyan_dgsl_frozen100 import generate_xiyan,resolve_database,score_query
from experiments.xiyan_mschema_frozen100 import build_mschema
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.providers import OllamaProvider

CODER="qwen2.5-coder:7b-instruct-q4_K_M"
EXPECTED_COHORT="bird-fk-bridge-aggregate-v1"


def extract_blinded_baseline(full: dict) -> dict:
    """Strip ALL prior evaluation fields and previous alternative answers."""
    if full.get("schema")!="bird-fk-bridge-aggregate-v1":
        raise ValueError("require specific previous paired100 experiment")
    rows=full.get("cases")
    if not isinstance(rows,list) or len(rows)!=100:
        raise ValueError("need 100 archived source question IDs")
    safe=[]
    for row in rows:
        sql=(row.get("sql") or {}).get("a0")
        if not isinstance(sql,str) or not sql.strip():
            raise ValueError("missing XiYan A0 SQL")
        safe.append({"case_id":str(row["case_id"]),
                     "database_id":str(row["database_id"]),
                     "xiyan_a0_sql":sql.strip()})
    if len({r["case_id"] for r in safe})!=100:
        raise ValueError("duplicate archived questions")
    return {"schema":"bird-diverse-backbone-blind-baseline-v1",
            "cohort_sha256":full["cohort_sha256"],
            "source_artifact_sha256":sha256(json.dumps(full,sort_keys=True).encode()).hexdigest(),
            "cases":sorted(safe,key=lambda r:int(r["case_id"]))}


def run_shard(*,source:Path,root:Path,blinded_baseline:Path,shard:int,
              model:str=CODER)->dict:
    if shard not in range(4):
        raise ValueError("4 CPU shards required")
    data=json.loads(blinded_baseline.read_text(encoding="utf-8"))
    if data.get("schema")!="bird-diverse-backbone-blind-baseline-v1":
        raise ValueError("unsafe baseline: historical outcome/alternatives may leak")
    rows=data["cases"]
    if len(rows)!=100 or len({r["case_id"] for r in rows})!=100 or any(set(r)!={"case_id","database_id","xiyan_a0_sql"} for r in rows):
        raise ValueError("blinded baseline must contain only IDs, physical DB and XiYan SQL")
    cases, gold_labels=load_bird_json(source)
    lookup={str(x.case_id):x for x in cases}
    target=[r for idx,r in enumerate(rows) if idx%4==shard]
    if len(target)!=25 or any(r["case_id"] not in lookup for r in target):
        raise ValueError("bad source/baseline cohort")
    provider=OllamaProvider(model=model,timeout_s=900)
    cache={}
    generated=[]
    for idx,row in enumerate(target,1):
        case=lookup[row["case_id"]]
        if case.database_id!=row["database_id"]:
            raise ValueError("wrong physical DB")
        if case.database_id not in cache:
            db=resolve_database(root,case.database_id)
            catalog=DatabaseCatalog.from_sqlite(db)
            descriptions={(t,c):d for t,c,d in _description_rows(root,case.database_id)}
            cache[case.database_id]=(db,catalog,descriptions)
        db,catalog,descriptions=cache[case.database_id]
        _, selection=select_schema_context(catalog,root,case.database_id,case.question)
        schema=build_mschema(catalog=catalog,database_id=case.database_id,
                             selected_tables=selection["selected_tables"],
                             descriptions=descriptions)
        started=perf_counter()
        candidate=generate_xiyan(question=case.question,schema_context=schema,
                                 evidence=case.evidence.strip() or None,provider=provider)
        generated.append({"case_id":row["case_id"],"database_id":case.database_id,
                          "xiyan_a0_sql":row["xiyan_a0_sql"],"coder_sql":candidate,
                          "changed":row["xiyan_a0_sql"].strip()!=candidate.strip(),
                          "coder_latency_s":round(perf_counter()-started,2)})
        print(f"CODER_DIVERSE_GENERATED shard={shard} {idx}/25",flush=True)

    # Reference SQL never enters model prompt, or even per-case scoring until
    # every inference of a shard has completed.
    gold={str(x.case_id):x.sql for x in gold_labels}
    out=[]
    for row in generated:
        db=cache[row["database_id"]][0]
        score_a0=score_query(db,row["xiyan_a0_sql"],gold[row["case_id"]])
        score_a1=score_query(db,row["coder_sql"],gold[row["case_id"]])
        out.append({**row,"ex_a0":bool(score_a0["frozen_execution_match"]),
                    "ex_coder":bool(score_a1["frozen_execution_match"]),
                    "valid_a0":bool(score_a0["execution_success"]),
                    "valid_coder":bool(score_a1["execution_success"])})
    return {"schema":"bird-diverse-backbone-shard-v1","shard":shard,
            "model":model,"cohort_sha256":data["cohort_sha256"],
            "gold_visible_in_generation":False,"cases":out}


def aggregate(shards:list[dict])->dict:
    if len(shards)!=4 or sorted(s.get("shard") for s in shards)!=[0,1,2,3]:
        raise ValueError("4 distinct shard outputs required")
    if any(s.get("schema")!="bird-diverse-backbone-shard-v1" or
           s.get("gold_visible_in_generation") is not False for s in shards):
        raise ValueError("invalid shard / gold exposure")
    if len({s.get("model") for s in shards})!=1 or len({s.get("cohort_sha256") for s in shards})!=1:
        raise ValueError("model/cohort drift")
    rows=[r for s in shards for r in s["cases"]]
    if len(rows)!=100 or len({r["case_id"] for r in rows})!=100:
        raise ValueError("100 distinct cases required")
    base=sum(bool(r["ex_a0"]) for r in rows)
    coder=sum(bool(r["ex_coder"]) for r in rows)
    rescues=sum(not r["ex_a0"] and r["ex_coder"] for r in rows)
    regressions=sum(r["ex_a0"] and not r["ex_coder"] for r in rows)
    if base!=64 or coder-base!=rescues-regressions:
        raise ValueError("paired arithmetic or frozen source baseline changed")
    return {"schema":"bird-diverse-backbone-aggregate-v1",
            "count":100,"backbone":shards[0]["model"],
            "cohort_sha256":shards[0]["cohort_sha256"],
            "xiyan_correct":base,"coder_correct":coder,
            "wrong_to_correct":rescues,"correct_to_wrong":regressions,
            "candidate_oracle_correct":base+rescues,
            "oracle_gain_vs_xiyan":rescues,
            "changed_sql":sum(bool(r["changed"]) for r in rows),
            "automatic_selector_ran":False,
            "training_ran":False,
            "note":"Same exposed exploratory BIRD DEV cohort as bridge ablation; test on a NEW holdout before promotion. Local SQLite set-of-tuples, not official blind metric.",
            "cases":sorted(rows,key=lambda r:int(r["case_id"]))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-source",type=Path)
    p.add_argument("--blind-output",type=Path)
    p.add_argument("--source",type=Path)
    p.add_argument("--database-root",type=Path)
    p.add_argument("--blinded-baseline",type=Path)
    p.add_argument("--shard-index",type=int)
    p.add_argument("--model",default=CODER)
    p.add_argument("--inputs",type=Path,nargs="+")
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    if a.baseline_source:
        if a.blind_output is None or a.blind_output.exists():
            p.error("need new non-existing blind-output")
        a.blind_output.write_text(json.dumps(extract_blinded_baseline(
            json.loads(a.baseline_source.read_text(encoding="utf-8"))),
            ensure_ascii=False,indent=2,sort_keys=True)+"\n")
        return
    if a.output is None or a.output.exists():
        p.error("need new output path")
    if a.inputs:
        value=aggregate([json.loads(x.read_text(encoding="utf-8")) for x in a.inputs])
        print("BIRD_DIVERSE_BACKBONE",json.dumps({k:v for k,v in value.items() if k!="cases"},
                                                sort_keys=True),flush=True)
    else:
        if any(x is None for x in (a.source,a.database_root,a.blinded_baseline,a.shard_index)):
            p.error("missing generation inputs")
        value=run_shard(source=a.source,root=a.database_root,blinded_baseline=a.blinded_baseline,
                        shard=a.shard_index,model=a.model)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n")


if __name__=="__main__":
    main()
