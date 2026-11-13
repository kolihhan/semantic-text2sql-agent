"""Fresh question-disjoint BIRD40 paired original Qwen9B Guarded vs count/grain v3.

No training. Exclude all 500 historically sampled BIRD DEV identities.
Preregister 20 count-intent and 20 control questions by stable seeded hash;
max three questions per database per stratum. No evaluation answers are used
to choose the 40 questions or revise SQL. One shared original Qwen output,
then deterministic opt-in v3. Gold execution-match is measured only after
candidate decisions. 8 CPU inference shards, no default-path changes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import urllib.request

from evaluation.bird_loader import load_bird_json
from evaluation.run_bird import _resolve_database, _ollama_model_digest
from evaluation.schema_context import select_schema_context
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.semantic_shape_v3 import evidence_gated_count_projection, _COUNT_INTENT
from semantic_sql.contracts import SQLCandidate
from semantic_sql.execution import SQLExecutionError, execute_readonly
from semantic_sql.inference import run_guarded
from semantic_sql.providers import OllamaProvider

MODEL="qwen3.5:9b-q4_K_M"
MODEL_DIGEST="56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
SEED="sql-v3-count-fresh40-paired-20261113-v1"
MODEL_SEED=42
COHORT_COUNTS=(100,100,100,100,40,20,40)
QUOTED=re.compile(r"""['"\x60][^'"\x60]{2,80}['"\x60]""")


class FixedModelProvider(OllamaProvider):
    """Use repeatable decoding; paired arms share exactly one generation."""
    def _chat(self, *, system: str, user: str) -> str:
        data={
            "model":self.model,"stream":False,"think":False,
            "options":{"temperature":0,"seed":MODEL_SEED},
            "messages":[{"role":"system","content":system},{"role":"user","content":user}],
        }
        req=urllib.request.Request(
            self.base_url.rstrip("/")+"/api/chat",
            data=json.dumps(data).encode(),headers={"Content-Type":"application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req,timeout=self.timeout_s) as resp:
            return str(json.loads(resp.read())["message"]["content"])


class FailIfCalled:
    def complete_text(self, *, system: str, user: str) -> str:
        raise AssertionError("second arm must NEVER call Qwen model")


def source_cohort_ids(paths: list[Path]) -> set[str]:
    if len(paths)!=7:
        raise ValueError("need seven previous frozen cohorts (100 x4, 40, 20, 40)")
    result=set()
    for path,n in zip(paths,COHORT_COUNTS,strict=True):
        payload=json.loads(path.read_text())
        rows=payload.get("cases") if isinstance(payload.get("cases"),list) else payload.get("selected")
        if not isinstance(rows,list) or len(rows)!=n:
            raise ValueError(f"invalid frozen case count {path.name}: expected {n}")
        ids=[str(row["case_id"]) for row in rows]
        if len(set(ids))!=n or result.intersection(ids):
            raise ValueError("duplicate or overlapping original DEV question IDs")
        result.update(ids)
    if len(result)!=500:
        raise AssertionError("missing original exclusion cohorts")
    return result


def manifest(*, source:Path, excluded:list[Path]) -> dict:
    seen=source_cohort_ids(excluded)
    cases,_reference_sql_not_read=load_bird_json(source)
    if len({str(x.case_id) for x in cases})!=len(cases):
        raise ValueError("duplicate source BIRD identities")
    available=sorted(
        (x for x in cases if str(x.case_id) not in seen),
        key=lambda x:(sha256(f"{SEED}:{x.case_id}".encode()).hexdigest(),str(x.case_id)),
    )
    selected={"count_intent":[], "control":[]}
    source_counts={"count_intent":Counter(), "control":Counter()}
    for item in available:
        group="count_intent" if _COUNT_INTENT.search(item.question) else "control"
        if len(selected[group])>=20 or source_counts[group][item.database_id]>=3:
            continue
        selected[group].append({
            "case_id":str(item.case_id),
            "database_id":item.database_id,
            "group":group,
        })
        source_counts[group][item.database_id]+=1
        if all(len(x)==20 for x in selected.values()):
            break
    if any(len(x)!=20 or len(source_counts[k])<7 for k,x in selected.items()):
        raise ValueError("not enough new database-diverse cases in strata")
    chosen=selected["count_intent"]+selected["control"]
    ids={r["case_id"] for r in chosen}
    if len(ids)!=40 or ids & seen:
        raise AssertionError("overlap in selected new cases")
    return {
        "schema":"sql-semantic-v3-new40-preregistered/v1",
        "selection_seed":SEED,"source_sha256":sha256(source.read_bytes()).hexdigest(),
        "model":MODEL,"model_seed":MODEL_SEED,
        "excluded_identity_count":500,
        "group_counts":{"count_intent":20,"control":20},
        "source_database_count":{k:len(v) for k,v in source_counts.items()},
        "gold_used_in_selection":False,
        "selected":chosen,
    }


def _score(db:Path, sql:str, gold:str) -> bool:
    try:
        pred=execute_readonly(db,sql,max_rows=50_000)
        truth=execute_readonly(db,gold,max_rows=50_000)
        return not (pred.truncated or truth.truncated) and set(pred.rows)==set(truth.rows)
    except (SQLExecutionError,ValueError):
        return False


def score_shard(*, source:Path, database_root:Path, manifest_path:Path, shard:int,
                model:str=MODEL) -> dict:
    if shard not in range(8) or model!=MODEL:
        raise ValueError("exact frozen 8-shard Qwen9B evaluation required")
    cfg=json.loads(manifest_path.read_text())
    if (cfg.get("schema")!="sql-semantic-v3-new40-preregistered/v1"
        or cfg.get("excluded_identity_count")!=500
        or cfg.get("gold_used_in_selection") is not False
        or cfg.get("selection_seed")!=SEED
        or cfg.get("source_sha256")!=sha256(source.read_bytes()).hexdigest()
        or len(cfg.get("selected",[]))!=40):
        raise ValueError("wrong manifest or changed source")
    ids={str(x["case_id"]) for x in cfg["selected"]}
    if len(ids)!=40:
        raise ValueError("duplicate frozen manifest ID")
    cases,labels=load_bird_json(source)
    by_id={str(c.case_id):c for c in cases}
    picked=cfg["selected"][shard::8]
    if len(picked)!=5:
        raise ValueError("each shard must have 5 unique frozen questions")
    digest=_ollama_model_digest("http://127.0.0.1:11434",model)
    if digest!=MODEL_DIGEST:
        raise ValueError("model digest drift; not the original pinned Qwen3.5-9B")
    provider=FixedModelProvider(model=model,timeout_s=900)
    db_cache={}
    frozen=[]
    for item in picked:
        cid=item["case_id"]
        case=by_id[cid]
        if case.database_id!=item["database_id"]:
            raise ValueError("database mismatch in fixed manifest")
        if case.database_id not in db_cache:
            db=_resolve_database(database_root,case.database_id)
            db_cache[case.database_id]=(db,DatabaseCatalog.from_sqlite(db))
        db,catalog=db_cache[case.database_id]
        context,_diag=select_schema_context(catalog,database_root,case.database_id,case.question)
        baseline=run_guarded(
            database=db,provider=provider,question=case.question,
            schema_context=context,evidence=case.evidence.strip() or None,
            max_repairs=2,max_rows=50_000,semantic_v2=False,
        )
        if baseline.status=="ok" and baseline.candidate is not None:
            # This is an independent deterministic projection correction,
            # not another LLM call or a gold-conditioned selector.
            decision=evidence_gated_count_projection(
                database=db,question=case.question,sql=baseline.candidate.sql,
            )
            changed=decision.changed
            why=decision.reason
            alt_sql=decision.sql
            alt_status="ok"
        else:
            changed=False
            why="non_ok_baseline_preserved"
            alt_sql=baseline.candidate.sql if baseline.candidate else ""
            alt_status=baseline.status
        frozen.append({
            **item,
            "baseline_sql":baseline.candidate.sql if baseline.candidate else "",
            "baseline_status":baseline.status,
            "baseline_repairs":baseline.candidate.attempt if baseline.candidate else 0,
            "v3_sql":alt_sql,"v3_status":alt_status,
            "v3_changed":changed,"v3_reason":why,
        })
        print("SQL_SEMANTIC_V3_FRESH40_GENERATED",shard,len(frozen),cid,
              "changed="+str(changed),flush=True)
    # All model generations and deterministic proposals for this shard have
    # completed. Only now access the gold/reference SQL from BIRD.
    gold={str(x.case_id):x.sql for x in labels}
    for row in frozen:
        db=db_cache[row["database_id"]][0]
        row["baseline_ex"]=_score(db,row["baseline_sql"],gold[row["case_id"]])
        row["v3_ex"]=_score(db,row["v3_sql"],gold[row["case_id"]])
    return {
        "schema":"sql-semantic-v3-new40-shard/v1",
        "shard":shard,
        "model":MODEL,"seed":MODEL_SEED,
        "manifest_sha256":sha256(manifest_path.read_bytes()).hexdigest(),
        "gold_used_during_generation_or_selection":False,
        "cases":frozen,
    }


def aggregate(inputs:list[dict]) -> dict:
    if len(inputs)!=8 or sorted(i.get("shard") for i in inputs)!=list(range(8)):
        raise ValueError("all 8 unique shards needed")
    if any(x.get("schema")!="sql-semantic-v3-new40-shard/v1"
           or x.get("gold_used_during_generation_or_selection") is not False
           or x.get("model")!=MODEL or x.get("seed")!=MODEL_SEED for x in inputs):
        raise ValueError("missing frozen 9B inference provenance")
    if len({x["manifest_sha256"] for x in inputs})!=1:
        raise ValueError("shard manifest drift")
    rows=[r for shard in inputs for r in shard["cases"]]
    if len(rows)!=40 or len({x["case_id"] for x in rows})!=40:
        raise ValueError("incomplete or duplicate paired cohort")
    groups=defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    if {k:len(v) for k,v in groups.items()}!={"count_intent":20,"control":20}:
        raise ValueError("incorrect balanced strata")
    def metrics(records):
        base=sum(bool(r["baseline_ex"]) for r in records)
        new=sum(bool(r["v3_ex"]) for r in records)
        correct=sum(not r["baseline_ex"] and r["v3_ex"] for r in records)
        regress=sum(r["baseline_ex"] and not r["v3_ex"] for r in records)
        assert new-base==correct-regress
        return {"count":len(records),"baseline_correct":base,"v3_correct":new,
                "changed":sum(bool(r["v3_changed"]) for r in records),
                "rescued":correct,"regressed":regress,"net_gain":new-base,
                "reasons":dict(Counter(str(r["v3_reason"]).split(":")[0] for r in records))}
    return {
        "schema":"sql-semantic-v3-new40-aggregate/v1",
        "model":MODEL,"manifest_sha256":inputs[0]["manifest_sha256"],
        "paired_shared_baseline_generation":True,
        "overall":metrics(rows),
        "by_group":{k:metrics(v) for k,v in sorted(groups.items())},
        "no_training":True,
        "promote":False,
        "claim_boundary":"40 new question-disjoint public BIRD DEV problems; not BIRD hidden, "
                         "not independently DB-disjoint and no confidence interval "
                         "or production promotion from small post-hoc result.",
        "cases":sorted(rows,key=lambda x:int(x["case_id"])),
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path)
    p.add_argument("--database-root",type=Path)
    p.add_argument("--exclude",nargs="+",type=Path)
    p.add_argument("--manifest",type=Path)
    p.add_argument("--shard",type=int)
    p.add_argument("--inputs",nargs="+",type=Path)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError("do not overwrite a frozen source artifact")
    if a.inputs:
        output=aggregate([json.loads(p.read_text()) for p in a.inputs])
    elif a.exclude:
        if a.source is None: p.error("manifest generation needs --source")
        output=manifest(source=a.source,excluded=a.exclude)
    else:
        if a.source is None or a.database_root is None or a.manifest is None or a.shard is None:
            p.error("shard needs --source --database-root --manifest --shard")
        output=score_shard(source=a.source,database_root=a.database_root,
                           manifest_path=a.manifest,shard=a.shard)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(output,indent=2,ensure_ascii=False,sort_keys=True)+"\n")
    print("SQL_SEMANTIC_V3_FRESH40_STAGE",json.dumps(
        {k:v for k,v in output.items() if k not in ("selected","cases")},sort_keys=True
    ),flush=True)


if __name__=="__main__":
    main()
