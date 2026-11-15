"""Source-shift test: same Qwen9B Guarded versus physical SQLite values on Spider DEV.

The previous BIRD DEV attempt stopped before inference: all 11 databases were
already represented in frozen100. This tests a *different physical corpus*.
Only conditional paired SQLite execution equality is claimed; this is NOT
official Spider exact match, BIRD EX, or model-generalization proof.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

from evaluation.qwen_value_evidence_bird20 import (
    DIGEST, MODEL, _evaluate, enhanced_context, value_hits,
)
from evaluation.run_bird import _ollama_model_digest
from evaluation.schema_context import select_schema_context
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.execution import SQLExecutionError, execute_readonly

PROTOCOL = "qwen9b-real-db-value-evidence-spider-dev12-v1"
ARCHIVE_SHA = "00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b"
SEED = "spider-dev-value-grounding-20261115-prospective-v1"
N_SHARDS = 3


def _hash(text: str) -> str:
    return sha256(f"{SEED}:{text}".encode()).hexdigest()


def _dbpath(root: Path, dbid: str) -> Path:
    if not dbid or not all(c.isalnum() or c=="_" for c in dbid):
        raise ValueError("invalid physical Spider DB ID")
    path = root / dbid / f"{dbid}.sqlite"
    if not path.is_file():
        raise FileNotFoundError(f"no physical source DB {dbid}")
    return path


def freeze_spider_cohort(train: list[dict], dev: list[dict], db_root: Path) -> dict:
    """Select 6 DB × 2 questions from question + actual DB values only."""
    train_dbs = {str(x["db_id"]) for x in train}
    source_dbs = {str(x["db_id"]) for x in dev}
    if source_dbs & train_dbs:
        raise ValueError("Spider TRAIN/DEV physical database overlap")
    dbs = defaultdict(list)
    indexed = {}
    for idx, raw in enumerate(dev):
        dbid = str(raw["db_id"])
        question = str(raw.get("question") or "").strip()
        if not question:
            raise ValueError("missing question")
        if dbid not in indexed:
            indexed[dbid] = DatabaseCatalog.from_sqlite(_dbpath(db_root,dbid))
        hits = value_hits(indexed[dbid],question)
        if not hits:
            continue
        dbs[dbid].append({
            "case_id":f"{dbid}:{idx}", "index":idx,
            "db_id":dbid,"question":question,"value_hits":hits,
        })
    eligible = sorted((dbid for dbid,rows in dbs.items() if len(rows)>=2),
                      key=lambda dbid:_hash("db:"+dbid))
    if len(eligible)<6:
        raise ValueError(f"only {len(eligible)} physical DBs have two grounded questions; stop before inference")
    chosen=[]
    for dbid in eligible[:6]:
        chosen += sorted(dbs[dbid],key=lambda r:_hash("case:"+r["case_id"]))[:2]
    cohort_sha=sha256(json.dumps(chosen,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    return {
        "protocol":PROTOCOL, "cohort_sha256":cohort_sha,
        "archive_sha256":ARCHIVE_SHA,
        "model":MODEL, "model_digest":DIGEST,
        "count":12,"db_count":6,"selected_from":"Spider DEV, not Spider TRAIN",
        "gold_blind_selection":True,
        "conditional_on_existing_question_matched_values":True,
        "source_shift_from_BIRD_DEV":True,
        "original_sql_training":False,
        "not_official_Spider_EM_or_BIRD_EX":True,
        "cases":chosen,
    }


def run_shard(manifest: Path, train_file: Path, dev_file: Path, root: Path, shard: int) -> dict:
    if shard not in range(N_SHARDS):
        raise ValueError("invalid shard index")
    cfg=json.loads(manifest.read_text(encoding="utf-8"))
    if (cfg.get("protocol")!=PROTOCOL or cfg.get("model_digest")!=DIGEST
        or cfg.get("archive_sha256")!=ARCHIVE_SHA or cfg.get("count")!=12
        or len(cfg.get("cases",[]))!=12 or cfg.get("db_count")!=6):
        raise ValueError("source/cohort preflight changed")
    train=json.loads(train_file.read_text(encoding="utf-8"))
    dev=json.loads(dev_file.read_text(encoding="utf-8"))
    actual=freeze_spider_cohort(train,dev,root)
    if actual!=cfg:
        raise ValueError("physically pinned Spider source/manifest drift")
    digest=_ollama_model_digest("http://127.0.0.1:11434",MODEL)
    if digest!=DIGEST:
        raise ValueError("Qwen model digest changed")
    chosen=cfg["cases"][shard*4:(shard+1)*4]
    outputs=[]
    for row in chosen:
        db=_dbpath(root,row["db_id"])
        catalog=DatabaseCatalog.from_sqlite(db)
        schema,_=select_schema_context(catalog,root,row["db_id"],row["question"])
        plain=_evaluate(db,row["question"],schema,"")
        grounded=_evaluate(db,row["question"],enhanced_context(schema,row["value_hits"]),"")
        outputs.append({
            "case_id":row["case_id"],"db_id":row["db_id"],"index":row["index"],
            "baseline":plain,"evidence":grounded,
            "hit_count":len(row["value_hits"]),
        })
        print("SPIDER_VALUE_GENERATION_FROZEN",shard,row["case_id"],flush=True)
    # Read all labels only after both Qwen candidates on every shard task frozen.
    for row in outputs:
        db=_dbpath(root,row["db_id"])
        try:
            expected=execute_readonly(db,str(dev[row["index"]]["query"]),max_rows=50000)
            scorable=not expected.truncated
        except (SQLExecutionError,ValueError):
            scorable=False
            expected=None
        row["reference_scorable"]=scorable
        for arm in ("baseline","evidence"):
            pred=row[arm]
            got=pred.pop("rows")
            pred["ex"]=bool(pred["executable"] and set(map(tuple,got))
                            ==set(map(tuple,expected.rows))) if scorable else None
            pred["bag"]=bool(pred["executable"] and Counter(map(tuple,got))
                             == Counter(map(tuple,expected.rows))) if scorable else None
    return {"protocol":PROTOCOL,"shard":shard,"cohort_sha256":cfg["cohort_sha256"],
            "model_digest":DIGEST,"cases":outputs}


def aggregate(shards: list[dict], *, expected_digest: str=DIGEST) -> dict:
    if len(shards)!=N_SHARDS or sorted(s["shard"] for s in shards)!=list(range(N_SHARDS)):
        raise ValueError("three shards required")
    if len({s.get("cohort_sha256") for s in shards})!=1 or any(
        s.get("protocol")!=PROTOCOL or s.get("model_digest")!=expected_digest
        or len(s.get("cases",[]))!=4 for s in shards
    ):
        raise ValueError("model/cohort drift")
    rows=[r for s in sorted(shards,key=lambda d:d["shard"]) for r in s["cases"]]
    if len(rows)!=12 or len({r["case_id"] for r in rows})!=12 or len({r["db_id"] for r in rows})!=6:
        raise ValueError("source database/row count drift")
    scorable=[r for r in rows if r["reference_scorable"]]
    rescue=[r for r in scorable if r["baseline"]["ex"] is False and r["evidence"]["ex"] is True]
    regres=[r for r in scorable if r["baseline"]["ex"] is True and r["evidence"]["ex"] is False]
    old=sum(r["baseline"]["ex"] is True for r in scorable)
    new=sum(r["evidence"]["ex"] is True for r in scorable)
    if new-old!=len(rescue)-len(regres):
        raise AssertionError("paired scoring transitions mismatch")
    old_bag=sum(r["baseline"]["bag"] is True for r in scorable)
    new_bag=sum(r["evidence"]["bag"] is True for r in scorable)
    gain_dbs=len({r["db_id"] for r in rescue})
    return {
        "protocol":PROTOCOL, "count":12,"scorable":len(scorable),"dbs":6,
        "cohort_sha256":shards[0]["cohort_sha256"],
        "model_digest":expected_digest,
        "baseline_ex":old,"evidence_ex":new,
        "baseline_bag":old_bag,"evidence_bag":new_bag,
        "rescues":len(rescue),"regressions":len(regres),
        "improved_db_count":gain_dbs,
        "baseline_latency_s":round(sum(r["baseline"]["latency_s"] for r in rows),1),
        "evidence_latency_s":round(sum(r["evidence"]["latency_s"] for r in rows),1),
        "positive_screen_only":len(scorable)==12 and new-old>=2
            and len(regres)==0 and gain_dbs>=2 and new_bag>=old_bag,
        "promotion_allowed":False, "no_sql_training":True,
        "not_official_spider_exact_match_or_BIRD_EX":True,
        "caveat":"Spider DEV 6 physical DBs disjoint from Spider TRAIN and BIRD DEV; value-eligible and source-visible; no GAIA/BIRD claims or production promotion.",
        "cases":rows,
    }


def main():
    p=argparse.ArgumentParser()
    for key in ("train","dev","db-root","manifest","output"):
        p.add_argument("--"+key,type=Path)
    p.add_argument("--shard",type=int)
    p.add_argument("--inputs",type=Path,nargs="+")
    a=p.parse_args()
    if a.output is None or a.output.exists():
        raise FileExistsError("new nonexisting output required")
    if a.inputs:
        report=aggregate([json.loads(f.read_text(encoding="utf-8")) for f in a.inputs])
        print("SPIDER_QWEN9B_VALUE_SOURCE_SHIFT_FINAL",json.dumps({k:v for k,v in report.items() if k!="cases"},sort_keys=True),flush=True)
    elif a.manifest:
        report=run_shard(a.manifest,a.train,a.dev,a.db_root,a.shard)
    else:
        if any(x is None for x in (a.train,a.dev,a.db_root)):
            p.error("preflight requires train/dev/db-root")
        report=freeze_spider_cohort(json.loads(a.train.read_text()),json.loads(a.dev.read_text()),a.db_root)
        print("SPIDER_VALUE_SOURCE_SHIFT_FROZEN",json.dumps({k:v for k,v in report.items() if k!="cases"},sort_keys=True),flush=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
