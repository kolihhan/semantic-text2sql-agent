"""A0 M-Schema vs A4 relational-grounded M-Schema on unseen Spider dev DBs.

Cross-dataset, model-matched, same-100-query, zero-training experiment.
Dataset source: HAL-9001/spider-databases official Spider 1.0 rehost.
Cohort sampled by salted question hash + balanced database round-robin BEFORE
reading reference SQL. All gold scoring happens only after the model generated
both A0 and A4 candidates for its entire shard.

Local metric: project read-only SQLite set-of-tuples execution matching;
not official Spider Test Suite Accuracy, and Spider dev labels are public.
"""
from __future__ import annotations

from collections import Counter,defaultdict
from dataclasses import dataclass
from hashlib import sha256
import argparse
import json
from pathlib import Path
from time import perf_counter
import urllib.request

from evaluation.schema_context import select_schema_context
from experiments.xiyan_dgsl_frozen100 import generate_xiyan,resolve_database,score_query
from experiments.xiyan_mschema_frozen100 import build_mschema
from experiments.relational_grounding_v2 import enrich_mschema
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.providers import OllamaProvider

MODEL="hf.co/wanhin/XiYanSQL-QwenCoder-7B-2504-gguf:Q4_K_M"
SEED="spider-source-shift-relational-grounding-20261113-frozen-v1"


@dataclass
class FixedSeedOllamaProvider(OllamaProvider):
    """Only generation randomness changes from older exploratory experiments.

    Both Spider arms use the SAME temp=0 + seed protocol, so paired effects
    cannot be explained by unequal random decoding configurations.
    """
    def _chat(self,*,system:str,user:str)->str:
        req=urllib.request.Request(
            self.base_url.rstrip("/")+"/api/chat",
            method="POST",headers={"Content-Type":"application/json"},
            data=json.dumps({
                "model":self.model,"stream":False,"think":False,
                "options":{"temperature":0,"seed":42},
                "messages":[{"role":"system","content":system},
                            {"role":"user","content":user}],
            }).encode(),
        )
        with urllib.request.urlopen(req,timeout=self.timeout_s) as reply:
            return str(json.load(reply)["message"]["content"])


def frozen_selection(rows:list[dict],*,count:int=100,seed:str=SEED)->list[dict]:
    if count<1 or len(rows)<count:
        raise ValueError("insufficient Spider source rows")
    per_db=defaultdict(list)
    seen=set()
    for index,row in enumerate(rows):
        db=str(row.get("db_id") or "")
        question=str(row.get("question") or "").strip()
        if not db or not question:
            raise ValueError("invalid source Spider question")
        cid=f"{db}:{index}"
        if cid in seen:raise ValueError("duplicate Spider case")
        seen.add(cid)
        # Intentionally do not read row['query'] or any gold annotation.
        per_db[db].append({
            "case_id":cid,"row_index":index,"database_id":db,"question":question,
        })
    if len(per_db)<10:
        raise ValueError("need >=10 distinct unseen schema domains")
    for db,items in per_db.items():
        items.sort(key=lambda x:(sha256(f"{seed}:{x['case_id']}".encode()).hexdigest(),x["case_id"]))
    order=sorted(per_db,key=lambda d:sha256(f"{seed}:database:{d}".encode()).hexdigest())
    chosen=[]
    for step in range(max(len(group) for group in per_db.values())):
        for db in order:
            if len(per_db[db])<=step:continue
            chosen.append(per_db[db][step])
            if len(chosen)==count:
                return chosen
    raise ValueError("source too small for requested cohort")


def run_shard(*,source:Path,database_root:Path,shard_index:int,
              shards:int=4,count:int=100,model:str=MODEL)->dict:
    if shards!=4 or not 0<=shard_index<shards or count!=100:
        raise ValueError("pre-registered Spider protocol requires 4x25")
    raw=json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw,list):raise ValueError("Spider dev.json must be a JSON list")
    cohort=frozen_selection(raw,count=count)
    cohort_sha=sha256("\n".join(row["case_id"] for row in cohort).encode()).hexdigest()
    questions=cohort[shard_index::shards]
    if len(questions)!=25:raise ValueError("not exactly 25 selected cases")
    provider=FixedSeedOllamaProvider(model=model,timeout_s=900)
    cache={}
    generated=[]
    for i,record in enumerate(questions,1):
        database_id=record["database_id"]
        if database_id not in cache:
            db=resolve_database(database_root,database_id)
            cat=DatabaseCatalog.from_sqlite(db)
            cache[database_id]=(db,cat)
        db,cat=cache[database_id]
        start=perf_counter()
        _,selected=select_schema_context(cat,database_root,database_id,record["question"])
        original=build_mschema(
            catalog=cat,database_id=database_id,
            selected_tables=selected["selected_tables"],descriptions={})
        grounded,diag=enrich_mschema(
            base_mschema=original,database=db,
            selected_tables=selected["selected_tables"],
            question=record["question"])
        if not grounded.startswith(original+"\n"):
            raise AssertionError("grounding dropped original schema columns")
        base=generate_xiyan(
            question=record["question"],schema_context=original,
            evidence=None,provider=provider)
        enhanced=generate_xiyan(
            question=record["question"],schema_context=grounded,
            evidence=None,provider=provider)
        generated.append({
            **record,"sql":{"a0":base,"a4":enhanced},
            "a0_schema_sha256":sha256(original.encode()).hexdigest(),
            "a4_grounding":diag,
            "latency_s":round(perf_counter()-start,3),
            "gold_consulted_before_generation":False,
        })
        print(f"SPIDER_PAIRED_GENERATED shard={shard_index} case={i}/25 id={record['case_id']}",flush=True)
    # Strict separation: access public dev SQL labels only AFTER all 50
    # candidates for this shard have been generated and frozen.
    scored=[]
    unscorable=[]
    for row in generated:
        gold=raw[row["row_index"]]["query"]
        db=cache[row["database_id"]][0]
        try:
            scores={
                arm:score_query(db,sql,gold)
                for arm,sql in row["sql"].items()
            }
        except (RuntimeError,ValueError,TypeError) as exc:
            unscorable.append({"case_id":row["case_id"],"reason":type(exc).__name__})
            continue
        scored.append({**row,"scores":scores})
    return {
        "schema":"spider-source-shift-a0-a4-shard-v1",
        "dataset":"Spider dev (public, NOT hidden test)",
        "metric":"project-local read-only SQLite set-of-tuples EX",
        "model":model,"seed":SEED,"temperature":0,"ollama_seed":42,
        "cohort_sha256":cohort_sha,"shard":shard_index,"requested":25,
        "scored":len(scored),"unscorable":unscorable,
        "cases":scored,
    }


def aggregate(shards:list[dict])->dict:
    if len(shards)!=4 or sorted(r.get("shard") for r in shards)!=[0,1,2,3]:
        raise ValueError("need four uniquely indexed shards")
    if len({r["cohort_sha256"] for r in shards})!=1:
        raise ValueError("cohort hash mismatch")
    for s in shards:
        if s["schema"]!="spider-source-shift-a0-a4-shard-v1" or s["requested"]!=25:
            raise ValueError("unexpected shard protocol")
    rows=[c for shard in shards for c in shard["cases"]]
    ids=[c["case_id"] for c in rows]
    if len(ids)!=len(set(ids)):raise ValueError("duplicate case IDs")
    total=len(rows)
    if not total or sum(len(s["unscorable"]) for s in shards)+total!=100:
        raise ValueError("incomplete 100-case cohort")
    baseline=sum(bool(c["scores"]["a0"]["frozen_execution_match"]) for c in rows)
    enriched=sum(bool(c["scores"]["a4"]["frozen_execution_match"]) for c in rows)
    wins=sum(not c["scores"]["a0"]["frozen_execution_match"]
             and c["scores"]["a4"]["frozen_execution_match"] for c in rows)
    losses=sum(c["scores"]["a0"]["frozen_execution_match"]
               and not c["scores"]["a4"]["frozen_execution_match"] for c in rows)
    if enriched-baseline!=wins-losses:
        raise AssertionError("paired arithmetic mismatch")
    dbs={c["database_id"] for c in rows}
    return {
        "schema":"spider-source-shift-a0-a4-aggregate-v1",
        "sample_size":total,"requested_cases":100,"database_count":len(dbs),
        "unscorable":[x for shard in shards for x in shard["unscorable"]],
        "cohort_sha256":shards[0]["cohort_sha256"],
        "model":shards[0]["model"],"temperature":0,"ollama_seed":42,
        "a0_correct":baseline,"a4_correct":enriched,
        "wrong_to_correct":wins,"correct_to_wrong":losses,
        "net_gain":wins-losses,
        "total_literal_hits":sum(c["a4_grounding"]["literal_hits"] for c in rows),
        "total_fk_declarations":sum(c["a4_grounding"]["verified_fk_declarations"] for c in rows),
        "average_pair_latency_s":round(sum(c["latency_s"] for c in rows)/total,2),
        "cases":sorted(rows,key=lambda row:row["case_id"]),
        "caveat":"Spider source-shift, public DEV labels; local set-of-tuples EX not official test-suite accuracy; study selection and repeated testing bias remain.",
    }


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path)
    p.add_argument("--database-root",type=Path)
    p.add_argument("--shard-index",type=int)
    p.add_argument("--model",default=MODEL)
    p.add_argument("--inputs",type=Path,nargs="+")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("refuse to overwrite prior evidence")
    if a.inputs:
        result=aggregate([json.loads(p.read_text(encoding="utf-8")) for p in a.inputs])
        print("SPIDER_A4_PAIRED_RESULT",json.dumps({k:v for k,v in result.items() if k!="cases"},sort_keys=True),flush=True)
    else:
        if any(v is None for v in (a.source,a.database_root,a.shard_index)):
            p.error("requires --source --database-root --shard-index")
        result=run_shard(source=a.source,database_root=a.database_root,
                         shard_index=a.shard_index,model=a.model)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,sort_keys=True,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
