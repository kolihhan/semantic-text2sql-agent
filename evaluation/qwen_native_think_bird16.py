"""BIRD DEV16: native thinking vs no-thinking, same pinned Qwen3.5-9B Guarded.

No new semantic prompt, no SQL training, no alternate model. Both arms use
identical question, schema, evidence, deterministic sampling, token budget,
read-only preflight/execution, and up to 2 repairs. Gold is read ONLY after
both arms freeze their SQL. Raw model reasoning is never persisted.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from time import perf_counter
from urllib.request import Request, urlopen

from evaluation.run_bird import _resolve_database, _ollama_model_digest
from evaluation.schema_context import select_schema_context
from evaluation.qwen_native_think_gate import MODEL, DIGEST
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.execution import execute_readonly, SQLExecutionError
from semantic_sql.inference import run_guarded

SOURCE_SHA = "ffd8018378ddb1a8794753e0a31cfc81862ff7318a5184c22f3dc4ce03a03feb"
SEED = "qwen9b-native-reasoning-bird16-new-20261114-v1"
N_SHARDS = 8
TOTAL = 16
PREVIOUS_16 = {
    "632", "706", "1477", "629", "635", "1140", "1097", "1112",
    "1156", "195", "721", "1422", "391", "155", "821", "851",
}
PREVIOUS_24 = {
    "1465", "576", "1348", "539", "156", "525", "1469", "1413",
    "113", "628", "499", "1324", "1000", "981", "307", "769",
    "244", "1174", "489", "141", "1186", "748", "529", "954",
}
ARM_KEYS = ("no_think", "native_think")


def _id(row: dict, index: int) -> str:
    return str(row.get("question_id") or row.get("id") or index)


def _source(path: Path) -> list[dict]:
    if sha256(path.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError("BIRD source bytes changed")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if len(rows) != 1534:
        raise ValueError("unexpected BIRD DEV size")
    return rows


def _group(question: str) -> str:
    return ("analytic" if re.search(
        r"\b(how many|total|sum|average|mean|highest|lowest|most|least|"
        r"ratio|percentage|percent|top|ranking|ranked|per|each)\b",
        question.casefold(),
    ) else "other")


def build_manifest(source: Path, frozen: Path) -> dict:
    rows = _source(source)
    prior = json.loads(frozen.read_text(encoding="utf-8"))
    if prior.get("status") != "complete" or prior.get("sample_size") != 100:
        raise ValueError("100 historical BIRD exclusions unavailable")
    excluded = {str(x["case_id"]) for x in prior["cases"]} | PREVIOUS_16 | PREVIOUS_24
    if len(excluded) != 140:
        raise ValueError("prior cohort exclusions overlap or changed")
    chosen = {"analytic": [], "other": []}
    per_db = {"analytic": Counter(), "other": Counter()}
    for cid, row in sorted(
        ((_id(x,i),x) for i,x in enumerate(rows) if _id(x,i) not in excluded),
        key=lambda item: (sha256(f"{SEED}:{item[0]}".encode()).hexdigest(),item[0]),
    ):
        category = _group(str(row["question"]))
        dbid = str(row["db_id"])
        if len(chosen[category]) >= 8 or per_db[category][dbid] >= 2:
            continue
        chosen[category].append({
            "case_id": cid, "db_id": dbid,
            "question": str(row["question"]), "evidence": str(row.get("evidence") or ""),
            "group": category,
        })
        per_db[category][dbid] += 1
        if all(len(x) == 8 for x in chosen.values()):
            break
    if any(len(v)!=8 or len(per_db[k])<4 for k,v in chosen.items()):
        raise ValueError("insufficient database-diverse semantic cohorts")
    # Interleave analytic and other for equal difficulty per CPU shard.
    flat = [item for i in range(8) for item in (chosen["analytic"][i],chosen["other"][i])]
    if len({r["case_id"] for r in flat}) != TOTAL:
        raise ValueError("case identity overlap")
    key = sha256(json.dumps(
        [(r["case_id"],r["db_id"],r["question"],r["group"]) for r in flat],
        ensure_ascii=False,separators=(",",":"),
    ).encode()).hexdigest()
    return {
        "protocol":"qwen-native-reasoning-bird16-v1",
        "source_sha256":SOURCE_SHA,"model_id":MODEL,"model_digest":DIGEST,
        "seed":SEED,"cohort_sha256":key,"excluded_prior_ids":140,
        "gold_used_for_selection":False,"gold_used_for_generation":False,
        "max_new_tokens_both_arms":4096,"temperature":0,"model_seed":42,
        "selected":flat,
        "boundary":"Public BIRD DEV, question-disjoint only from exact frozen100 + schema16 + physical24; "
                   "other prior project experiments and pretraining may overlap.",
    }


class NativeQwenProvider:
    def __init__(self, *, think: bool):
        self.think=think
        self.calls=0
        self.thinking_chars=0
        self.output_tokens=0

    def complete_text(self, *, system: str, user: str) -> str:
        payload={
            "model":MODEL,"stream":False,"think":self.think,
            "options":{"temperature":0,"seed":42,"num_predict":4096},
            "messages":[{"role":"system","content":system},
                        {"role":"user","content":user}],
        }
        req=Request(
            "http://127.0.0.1:11434/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type":"application/json"},
            method="POST",
        )
        with urlopen(req,timeout=1800) as resp:
            data=json.load(resp)
        msg=data.get("message")
        if not isinstance(msg,dict):
            raise RuntimeError("Qwen returned malformed message")
        self.calls+=1
        self.thinking_chars+=len(str(msg.get("thinking") or ""))
        self.output_tokens+=int(data.get("eval_count") or 0)
        return str(msg.get("content") or "")


def run_shard(source: Path, dbroot: Path, manifest: Path, shard: int) -> dict:
    if shard not in range(N_SHARDS):
        raise ValueError("invalid shard")
    data = _source(source)
    cfg = json.loads(manifest.read_text(encoding="utf-8"))
    if (cfg.get("protocol") != "qwen-native-reasoning-bird16-v1"
        or cfg.get("source_sha256") != SOURCE_SHA or cfg.get("model_digest") != DIGEST
        or cfg.get("model_id") != MODEL or cfg.get("max_new_tokens_both_arms") != 4096
        or cfg.get("gold_used_for_selection") is not False
        or cfg.get("gold_used_for_generation") is not False
        or len(cfg.get("selected",[])) != TOTAL):
        raise ValueError("unrecognized frozen cohort")
    modelhash=_ollama_model_digest("http://127.0.0.1:11434",MODEL)
    if modelhash!=DIGEST:
        raise ValueError("pinned Qwen model changed")
    cohort=cfg["selected"]
    chosen=cohort[2*shard:2*(shard+1)]
    if len(chosen)!=2 or {r["group"] for r in chosen}!={"analytic","other"}:
        raise ValueError("unbalanced shard")
    by_id={_id(r,i):r for i,r in enumerate(data)}
    databases={}
    cases=[]
    for row in chosen:
        raw=by_id[row["case_id"]]
        dbid=row["db_id"]
        if str(raw["db_id"])!=dbid or raw["question"]!=row["question"]:
            raise ValueError("question identity changed")
        if dbid not in databases:
            path=_resolve_database(dbroot,dbid)
            databases[dbid]=(path,DatabaseCatalog.from_sqlite(path))
        path,cat=databases[dbid]
        schema,diag=select_schema_context(cat,dbroot,dbid,row["question"])
        cases.append({**row,"schema_sha256":diag["schema_context_sha256"],
                      "_schema":schema,"arms":{}})
    # Generate two Qwen arms in fixed order; no reference SQL enters model prompts.
    for item in cases:
        path,_=databases[item["db_id"]]
        for name,think in (("no_think",False),("native_think",True)):
            provider=NativeQwenProvider(think=think)
            start=perf_counter()
            result=run_guarded(
                database=path,provider=provider,question=item["question"],
                schema_context=item["_schema"],
                evidence=item["evidence"].strip() or None,
                max_repairs=2,max_rows=50_000,
            )
            ok=result.status=="ok" and not result.truncated
            if think and provider.thinking_chars<=0:
                raise RuntimeError("native-think model returned no reasoning trace")
            item["arms"][name]={
                "sql":result.candidate.sql if result.candidate else "",
                "execution_success":ok,"_rows":result.rows if ok else (),
                "status":result.status,
                "model_calls":provider.calls,
                "thinking_chars":provider.thinking_chars,
                "eval_tokens":provider.output_tokens,
                "repairs":result.candidate.attempt if result.candidate else -1,
                "latency_s":round(perf_counter()-start,3),
            }
            print("QWEN_THINK_FROZEN_GENERATED",shard,item["case_id"],name,flush=True)
    for item in cases:
        item.pop("_schema")
        path,_=databases[item["db_id"]]
        try:
            gold=execute_readonly(path,by_id[item["case_id"]]["SQL"],max_rows=50_000)
            if gold.truncated:
                raise ValueError("truncated gold")
        except (SQLExecutionError,ValueError) as exc:
            item.update(reference_scorable=False,reference_error=type(exc).__name__)
            for arm in ARM_KEYS:
                item["arms"][arm].pop("_rows")
                item["arms"][arm].update(ex=None,bag=None)
            continue
        item.update(reference_scorable=True,reference_error=None)
        for arm in ARM_KEYS:
            record=item["arms"][arm]
            rows=record.pop("_rows")
            record["ex"]=bool(record["execution_success"] and set(rows)==set(gold.rows))
            record["bag"]=bool(record["execution_success"] and Counter(rows)==Counter(gold.rows))
    return {
        "protocol":cfg["protocol"],"shard":shard,
        "model_digest":modelhash,"cohort_sha256":cfg["cohort_sha256"],
        "cases":cases,
    }


def aggregate(parts: list[dict]) -> dict:
    if len(parts)!=N_SHARDS or sorted(p["shard"] for p in parts)!=list(range(N_SHARDS)):
        raise ValueError("eight unique paired shards required")
    first=parts[0]
    if any(p["protocol"]!="qwen-native-reasoning-bird16-v1"
           or p["model_digest"]!=DIGEST
           or p["cohort_sha256"]!=first["cohort_sha256"]
           or len(p["cases"])!=2 for p in parts):
        raise ValueError("model, cohort or protocol drift")
    rows=[r for p in parts for r in p["cases"]]
    if len(rows)!=TOTAL or len({r["case_id"] for r in rows})!=TOTAL:
        raise ValueError("missing/duplicate BIRD cases")
    def metrics(subset: list[dict]) -> dict:
        valid=[r for r in subset if r["reference_scorable"]]
        return {
            "n":len(subset),"valid_gold":len(valid),"invalid_gold":len(subset)-len(valid),
            "no_think_ex":sum(bool(r["arms"]["no_think"]["ex"]) for r in valid),
            "native_think_ex":sum(bool(r["arms"]["native_think"]["ex"]) for r in valid),
            "no_think_bag":sum(bool(r["arms"]["no_think"]["bag"]) for r in valid),
            "native_think_bag":sum(bool(r["arms"]["native_think"]["bag"]) for r in valid),
            "rescues":sum(not r["arms"]["no_think"]["ex"] and r["arms"]["native_think"]["ex"] for r in valid),
            "regressions":sum(r["arms"]["no_think"]["ex"] and not r["arms"]["native_think"]["ex"] for r in valid),
            "oracle_ex":sum(r["arms"]["no_think"]["ex"] or r["arms"]["native_think"]["ex"] for r in valid),
            "changed_sql":sum(r["arms"]["no_think"]["sql"]!=r["arms"]["native_think"]["sql"] for r in subset),
            "mean_latency_no_think":round(sum(r["arms"]["no_think"]["latency_s"] for r in subset)/len(subset),2),
            "mean_latency_native_think":round(sum(r["arms"]["native_think"]["latency_s"] for r in subset)/len(subset),2),
            "thinking_chars_total":sum(r["arms"]["native_think"]["thinking_chars"] for r in subset),
        }
    return {
        "protocol":first["protocol"],"model":MODEL,"model_digest":DIGEST,
        "cohort_sha256":first["cohort_sha256"],
        "overall":metrics(rows),
        "groups":{key:metrics([r for r in rows if r["group"]==key]) for key in ("analytic","other")},
        "cases":rows,
        "no_training":True,"gold_blind_generation":True,
        "oracle_is_hindsight_only":True,
        "no_chain_of_thought_saved":True,
        "boundary":"Public DEV question-disjoint only from known 140 prior cases; new model reasoning mode, not external-generalization claim",
    }


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path)
    p.add_argument("--database-root",type=Path)
    p.add_argument("--reference",type=Path,default=Path("runs/bird-paired-dev-v5/paired.json"))
    p.add_argument("--manifest",type=Path)
    p.add_argument("--shard",type=int)
    p.add_argument("--inputs",type=Path,nargs="+")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError("refusing to overwrite results")
    if a.inputs:
        result=aggregate([json.loads(x.read_text()) for x in a.inputs])
        print("QWEN_NATIVE_THINK_BIRD16",json.dumps({k:v for k,v in result.items() if k!="cases"},sort_keys=True),flush=True)
    elif a.shard is not None:
        if a.source is None or a.database_root is None or a.manifest is None:
            p.error("paired mode requires source, database-root, manifest")
        result=run_shard(a.source,a.database_root,a.manifest,a.shard)
    else:
        if a.source is None:
            p.error("manifest requires source")
        result=build_manifest(a.source,a.reference)
        print("QWEN_NATIVE_THINK_PREREGISTERED",
              json.dumps({k:v for k,v in result.items() if k!="selected"},sort_keys=True),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")


if __name__=="__main__":
    main()
