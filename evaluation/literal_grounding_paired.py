"""Matched original Qwen-9B guarded SQL vs verified quoted-value grounding.

Only source/SQLite evidence selects 40 fresh, value-positive BIRD DEV questions;
all known 400 previously explored question IDs are excluded. Reference SQL
used *after* both model outputs are frozen, never as a prompt or selector.
The baseline uses the original lexical schema + guarded repair. No new
semantic-revision model call and no change to the default service.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from evaluation.bird_loader import load_bird_json
from evaluation.schema_context import select_schema_context
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.execution import SQLExecutionError, execute_readonly
from semantic_sql.inference import run_guarded
from semantic_sql.literal_evidence import append_literal_evidence, exact_literal_evidence, quoted_literals
from semantic_sql.providers import OllamaProvider

SEED="bird-qwen9b-verified-literals-fresh-20261113-v1"
MODEL="qwen3.5:9b-q4_K_M"


def load_excluded(paths: list[Path]) -> set[str]:
    if len(paths) != 4:
        raise ValueError("exclude frozen100 plus three prior question-disjoint100 cohorts")
    ids:set[str]=set()
    for path in paths:
        obj=json.loads(path.read_text(encoding="utf-8"))
        cases=obj.get("cases")
        if not isinstance(cases,list):
            raise ValueError("prior manifest lacks cases")
        current=[str(row["case_id"]) for row in cases]
        if len(current)!=100 or len(set(current))!=100:
            raise ValueError("prior cohort not a unique100")
        if ids.intersection(current):
            raise ValueError("overlapping previous cohorts: cannot verify isolation")
        ids.update(current)
    if len(ids)!=400:
        raise ValueError("not 400 previously evaluated BIRD questions")
    return ids


def choose_grounded_cohort(cases, *, excluded:set[str], root:Path,
                           count:int=40, seed:str=SEED):
    if len(excluded)!=400 or count!=40:
        raise ValueError("frozen 400 exclusion and new 40 sample required")
    cases_by_id={str(c.case_id):c for c in cases}
    if len(cases_by_id)!=len(cases):
        raise ValueError("duplicated BIRD case IDs")
    eligible=[c for c in cases if str(c.case_id) not in excluded and quoted_literals(c.question)]
    eligible.sort(key=lambda c:(sha256(f"{seed}:{c.case_id}".encode()).hexdigest(),str(c.case_id)))
    cache={}
    selected=[]
    for c in eligible:
        if c.database_id not in cache:
            from evaluation.run_bird import _resolve_database
            db=_resolve_database(root,c.database_id)
            cache[c.database_id]=DatabaseCatalog.from_sqlite(db)
        hits=exact_literal_evidence(cache[c.database_id],c.question)
        if not hits:
            continue
        selected.append({"case_id":str(c.case_id),"database_id":str(c.database_id),
                         "matching_literal_hits":len(hits)})
        if len(selected)>=count:
            break
    if len(selected)!=count:
        raise ValueError(f"need {count} fresh, verified literal-hit cases; only {len(selected)} available")
    return selected


def _matched(db:Path, sql:str, gold:str, max_rows:int=50_000) -> bool:
    try:
        p=execute_readonly(db,sql,max_rows=max_rows)
        g=execute_readonly(db,gold,max_rows=max_rows)
    except SQLExecutionError:
        return False
    return not (p.truncated or g.truncated) and set(p.rows)==set(g.rows)


def run_shard(*, source:Path, root:Path, manifest:Path,
              model:str, shard:int) -> dict:
    if shard not in range(4):
        raise ValueError("four original-system shards")
    saved=json.loads(manifest.read_text())
    ids=[str(x["case_id"]) for x in saved["selected"]]
    if len(ids)!=40 or len(set(ids))!=40 or saved.get("excluded_count")!=400:
        raise ValueError("invalid frozen independent manifest")
    by_case, labels=load_bird_json(source)
    questions={str(c.case_id):c for c in by_case}
    chosen=ids[shard::4]
    if len(chosen)!=10: raise ValueError("expected ten cases per shard")
    cache={}
    provider=OllamaProvider(model=model,timeout_s=900)
    frozen=[]
    for n,cid in enumerate(chosen,1):
        c=questions[cid]
        if c.database_id not in cache:
            from evaluation.run_bird import _resolve_database
            db=_resolve_database(root,c.database_id)
            cache[c.database_id]=(db,DatabaseCatalog.from_sqlite(db))
        db,catalog=cache[c.database_id]
        baseline_schema,_=select_schema_context(catalog,root,c.database_id,c.question)
        grounded_schema=append_literal_evidence(catalog,c.question,
                                                 schema_context=baseline_schema)
        if grounded_schema==baseline_schema:
            raise ValueError("selected question no longer yields literal evidence")
        kwargs={"database":db,"provider":provider,"question":c.question,
                "evidence":c.evidence.strip() or None,"max_repairs":2,
                "max_rows":50_000,"semantic_revision":False}
        baseline=run_guarded(**kwargs,schema_context=baseline_schema)
        enhanced=run_guarded(**kwargs,schema_context=grounded_schema)
        frozen.append({
            "case_id":cid,"database_id":c.database_id,
            "baseline_sql":baseline.candidate.sql if baseline.candidate else "",
            "grounded_sql":enhanced.candidate.sql if enhanced.candidate else "",
            "baseline_status":baseline.status,"grounded_status":enhanced.status,
            "baseline_repairs":baseline.candidate.attempt if baseline.candidate else 0,
            "grounded_repairs":enhanced.candidate.attempt if enhanced.candidate else 0,
            "context_delta_bytes":len(grounded_schema)-len(baseline_schema),
            "changed_sql":bool(baseline.candidate and enhanced.candidate and
                               baseline.candidate.sql!=enhanced.candidate.sql),
        })
        print(f"LITERAL_PAIRED shard={shard} {n}/10 case_id={cid}",flush=True)
    # Reference SQL is accessed only after every baseline/grounded candidate.
    gold={str(c.case_id):c.sql for c in labels}
    scored=[]
    for row in frozen:
        db=cache[row["database_id"]][0]
        ex={"baseline":_matched(db,row["baseline_sql"],gold[row["case_id"]]),
            "grounded":_matched(db,row["grounded_sql"],gold[row["case_id"]])}
        scored.append({**row,"ex":ex})
    return {"schema":"qwen9b-exact-value-grounding-shard/v1","shard":shard,
            "model":model,"manifest_sha256":sha256(manifest.read_bytes()).hexdigest(),
            "gold_used_during_generation":False,"cases":scored}


def aggregate(shards:list[dict]) -> dict:
    if len(shards)!=4 or sorted(x.get("shard") for x in shards)!=[0,1,2,3]:
        raise ValueError("four shards required")
    if any(x.get("schema")!="qwen9b-exact-value-grounding-shard/v1" or
           x.get("gold_used_during_generation") is not False for x in shards):
        raise ValueError("invalid protocol")
    if len({x["model"] for x in shards})!=1 or len({x["manifest_sha256"] for x in shards})!=1:
        raise ValueError("model/selection drift")
    rows=[r for s in shards for r in s["cases"]]
    if len(rows)!=40 or len({r["case_id"] for r in rows})!=40:
        raise ValueError("40 independent paired cases required")
    baseline=sum(r["ex"]["baseline"] for r in rows)
    grounded=sum(r["ex"]["grounded"] for r in rows)
    gains=sum(not r["ex"]["baseline"] and r["ex"]["grounded"] for r in rows)
    regressions=sum(r["ex"]["baseline"] and not r["ex"]["grounded"] for r in rows)
    if gains-regressions!=grounded-baseline:
        raise ValueError("paired math invalid")
    return {"schema":"qwen9b-exact-value-grounding-paired40/v1",
            "sample_size":40,"baseline_correct":baseline,
            "grounded_correct":grounded,"wrong_to_correct":gains,
            "correct_to_wrong":regressions,"net_gain":gains-regressions,
            "changed_sql":sum(r["changed_sql"] for r in rows),
            "promote":False,
            "caveat":"Source- and value-grounded selection of 40 new public BIRD DEV questions; not random DEV accuracy or hidden evaluation; no model SFT.",
            "cases":sorted(rows,key=lambda r:int(r["case_id"]))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path)
    p.add_argument("--database-root",type=Path)
    p.add_argument("--excluded",nargs="+",type=Path)
    p.add_argument("--manifest",type=Path)
    p.add_argument("--shard-index",type=int)
    p.add_argument("--model",default=MODEL)
    p.add_argument("--inputs",nargs="+",type=Path)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("refuse to overwrite result")
    if a.inputs:
        result=aggregate([json.loads(x.read_text()) for x in a.inputs])
    elif a.excluded:
        if not a.source or not a.database_root:
            p.error("preselect requires source and database root")
        cases,_labels=load_bird_json(a.source)
        ex=load_excluded(a.excluded)
        cohort=choose_grounded_cohort(cases,excluded=ex,root=a.database_root)
        result={"schema":"qwen9b-literal-preselect/v1","selected":cohort,"excluded_count":len(ex),
                "selection_seed":SEED,"labels_used":False}
    else:
        if any(x is None for x in (a.source,a.database_root,a.manifest,a.shard_index)):
            p.error("paired shard requires source, database root, manifest, shard-index")
        result=run_shard(source=a.source,root=a.database_root,
                         manifest=a.manifest,model=a.model,shard=a.shard_index)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,sort_keys=True)+"\n")
    print("LITERAL_RESULT",json.dumps({k:v for k,v in result.items() if k not in ("cases","selected")},sort_keys=True))


if __name__=="__main__":
    main()
