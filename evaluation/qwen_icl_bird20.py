"""Gold-blind retrieved two-shot exemplars for the original Qwen3.5-9B Guarded SQL.

One hypothesis: real cross-database, high-quality BIRD TRAIN demonstrations
may improve the semantic output/selection behavior that deterministic SQL
verification misses. No SQL training, reranker, model swap or deployment.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from math import log, sqrt
from pathlib import Path
import re
from time import monotonic

from evaluation.run_bird import _ollama_model_digest, _resolve_database
from evaluation.schema_context import select_schema_context
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.execution import SQLExecutionError, execute_readonly
from semantic_sql.inference import run_guarded
from semantic_sql.providers import OllamaProvider

SOURCE_SHA = "ffd8018378ddb1a8794753e0a31cfc81862ff7318a5184c22f3dc4ce03a03feb"
MODEL = "qwen3.5:9b-q4_K_M"
DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
TRAIN_REPO = "birdsql/bird23-train-filtered"
PROTOCOL = "qwen9b-crossdb-icl-bird-dev20-exploratory-v1"
SEED = "qwen9b-two-crossdb-retrieval-20261115-before-gold"
N_SHARDS = 5
MAX_ROWS = 50_000
STOP = {
    "a","an","and","are","at","be","by","can","for","from","has","have",
    "how","in","is","it","of","on","or","the","their","these","to","was",
    "were","what","when","where","which","who","with","all","please",
}


def _hash(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _words(value: str) -> set[str]:
    return {w.rstrip("s") for w in re.findall(r"[a-z0-9]+",value.lower())
            if w not in STOP and len(w.rstrip("s"))>1}


def build_retrieval(train: list[dict]):
    """Training question only determines similarity; SQL labels never rank."""
    if not isinstance(train,list) or len(train)<2:
        raise ValueError("missing training reference examples")
    rows=[]
    docfreq=Counter()
    for index,item in enumerate(train):
        db=str(item.get("db_id") or "")
        question=str(item.get("question") or "").strip()
        sql=str(item.get("SQL") or "").strip()
        if not db or not question or not sql:
            raise ValueError("incomplete filtered BIRD TRAIN example")
        tokens=_words(question)
        if not tokens:
            continue
        rows.append({"db_id":db,"question":question,"SQL":sql,
                     "train_index":index,"tokens":tokens})
        docfreq.update(tokens)
    if len({x["db_id"] for x in rows})<2:
        raise ValueError("fewer than two eligible training databases")
    weights={word:log((len(rows)+1)/(count+1))+1 for word,count in docfreq.items()}
    return rows,weights


def generate_examples(index,question: str,current_db: str,*,k: int=2) -> list[dict]:
    """Pick k question-similar demos, each from a DIFFERENT training DB."""
    rows,weights=index
    query_tokens=_words(question)
    ident=" ".join(question.lower().split())
    candidates=[]
    for row in rows:
        if row["db_id"]==current_db or " ".join(row["question"].lower().split())==ident:
            continue
        matches=query_tokens & row["tokens"]
        if not matches:
            continue
        # High-IDF similarity normalized by training-question length.
        score=sum(weights[t] for t in matches)/sqrt(max(1,len(row["tokens"])))
        candidates.append((-score,_hash(f"{SEED}:{row['db_id']}:{row['train_index']}"),row))
    candidates.sort(key=lambda x:(x[0],x[1]))
    chosen=[]
    databases=set()
    for _,__,row in candidates:
        if row["db_id"] in databases:
            continue
        databases.add(row["db_id"])
        chosen.append({key:row[key] for key in ("db_id","question","SQL","train_index")})
        if len(chosen)==k:
            break
    if len(chosen)!=k:
        raise ValueError(f"insufficient cross-DB semantic examples: {len(chosen)}/{k}")
    return chosen


def case_id(row: dict, index: int) -> str:
    return str(row.get("question_id") or row.get("id") or index)


def select_cases(dev: list[dict],*,excluded: set[str]) -> list[dict]:
    """Freeze heldout by ID, question and database before labels or model outcomes."""
    groups=defaultdict(list)
    for i,row in enumerate(dev):
        cid=case_id(row,i)
        db=str(row.get("db_id") or "")
        question=str(row.get("question") or "").strip()
        if not cid or not db or not question:
            raise ValueError("invalid BIRD DEV source")
        if cid not in excluded:
            groups[db].append({"case_id":cid,"db_id":db,"question":question,
                               "evidence":str(row.get("evidence") or "")})
    eligible=[db for db,xs in groups.items() if len(xs)>=2]
    if len(eligible)<10:
        raise ValueError("need 10 database-distributed BIRD DEV samples")
    databases=sorted(eligible,key=lambda db:_hash(f"{SEED}:db:{db}"))[:10]
    selection=[]
    for db in databases:
        choices=sorted(groups[db],key=lambda r:_hash(f"{SEED}:case:{r['case_id']}"))
        selection+=choices[:2]
    if len(selection)!=20 or len({x["case_id"] for x in selection})!=20:
        raise ValueError("invalid frozen 20 question IDs")
    return selection


class PromptWithExamples:
    """Add context only to the first SQL generation call, not its repairs."""
    def __init__(self,inner,examples: list[dict]):
        if len(examples)!=2 or len({r["db_id"] for r in examples})!=2:
            raise ValueError("two distinct training-DB examples required")
        self.inner=inner
        self.examples=examples
        self.calls=0

    def complete_text(self,*,system:str,user:str)->str:
        self.calls+=1
        if self.calls==1:
            snippets="\n\n".join(
                f"Example from an unrelated database {i}:\n"
                f"User question: {e['question']}\n"
                f"Correct SQL for THAT database: {e['SQL']}"
                for i,e in enumerate(self.examples,1)
            )
            user=(
                "Use the examples ONLY as patterns for interpreting the user's "
                "requested result and relational conditions. Their database schemas "
                "and physical identifiers are DIFFERENT: never copy a table, column "
                "or literal from an example into the current task.\n\n"
                +snippets+"\n\nCURRENT DATABASE TASK:\n"+user
            )
        return self.inner.complete_text(system=system,user=user)


def make_manifest(source:Path,reference:Path,train:list[dict],train_revision:str)->dict:
    if sha256(source.read_bytes()).hexdigest()!=SOURCE_SHA:
        raise ValueError("BIRD DEV original LF bytes changed")
    if len(train)!=6601 or not re.fullmatch(r"[0-9a-f]{40}",train_revision):
        raise ValueError("filtered TRAIN source count/version changed")
    dev=json.loads(source.read_text(encoding="utf-8"))
    if len(dev)!=1534:
        raise ValueError("BIRD DEV source count changed")
    historic=json.loads(reference.read_text(encoding="utf-8"))
    if historic.get("status")!="complete" or len(historic.get("cases",[]))!=100:
        raise ValueError("frozen BIRD100 exclusion set changed")
    exclude={str(row["case_id"]) for row in historic["cases"]}
    if len(exclude)!=100:
        raise ValueError("duplicate baseline cases")
    selected=select_cases(dev,excluded=exclude)
    index=build_retrieval(train)
    for row in selected:
        row["examples"]=generate_examples(index,row["question"],row["db_id"],k=2)
    cohort=_hash(json.dumps([(x["case_id"],x["db_id"],
                               [(y["db_id"],y["train_index"]) for y in x["examples"]])
                             for x in selected],separators=(",",":")))
    return {
        "protocol":PROTOCOL, "source_sha256":SOURCE_SHA,
        "train_repo":TRAIN_REPO,"train_revision":train_revision,
        "train_count":len(train),"model":MODEL,"model_digest":DIGEST,
        "excluded_frozen100":len(exclude),
        "cohort_sha256":cohort,
        "selected":selected,"count":len(selected),
        "gold_blind_selection":True,"no_SQL_training":True,
        "unseen_questions_only_not_unseen_databases":True,
    }


def _run(db:Path,question:str,schema:str,evidence:str,provider)->dict:
    start=monotonic()
    result=run_guarded(
        database=db,provider=provider,question=question,schema_context=schema,
        evidence=evidence or None,max_repairs=2,max_rows=MAX_ROWS,
    )
    valid=result.status=="ok" and not result.truncated
    return {
        "sql":result.candidate.sql if result.candidate else "",
        "status":result.status,
        "execution_success":valid,"rows":result.rows if valid else (),
        "latency_s":round(monotonic()-start,2),
        "repair_attempts":result.candidate.attempt if result.candidate else 0,
    }


def run_shard(source:Path,manifest:Path,root:Path,shard:int)->dict:
    if shard not in range(N_SHARDS):
        raise ValueError("invalid shard index")
    if sha256(source.read_bytes()).hexdigest()!=SOURCE_SHA:
        raise ValueError("BIRD DEV input bytes changed")
    cfg=json.loads(manifest.read_text(encoding="utf-8"))
    if (cfg.get("protocol")!=PROTOCOL or cfg.get("source_sha256")!=SOURCE_SHA
        or cfg.get("count")!=20 or len(cfg.get("selected",[]))!=20
        or cfg.get("model_digest")!=DIGEST
        or cfg.get("gold_blind_selection") is not True):
        raise ValueError("preregistered cohort/model drift")
    digest=_ollama_model_digest("http://127.0.0.1:11434",MODEL)
    if digest!=DIGEST:
        raise ValueError("wrong physical Qwen9B digest")
    dev=json.loads(source.read_text(encoding="utf-8"))
    by_id={case_id(row,i):row for i,row in enumerate(dev)}
    selected=cfg["selected"][shard*4:(shard+1)*4]
    if len(selected)!=4:
        raise ValueError("incomplete shard")
    outcomes=[]
    for row in selected:
        cid,dbid=row["case_id"],row["db_id"]
        original=by_id[cid]
        if original["db_id"]!=dbid or original["question"]!=row["question"]:
            raise ValueError("frozen question/DB drift")
        db=_resolve_database(root,dbid)
        cat=DatabaseCatalog.from_sqlite(db)
        schema,_=select_schema_context(cat,root,dbid,row["question"])
        common=dict(model=MODEL,base_url="http://127.0.0.1:11434",
                    timeout_s=900,temperature=0,seed=42,num_predict=1024)
        base=_run(db,row["question"],schema,row["evidence"],OllamaProvider(**common))
        prompted=PromptWithExamples(OllamaProvider(**common),row["examples"])
        enhanced=_run(db,row["question"],schema,row["evidence"],prompted)
        outcomes.append({
            "case_id":cid,"db_id":dbid,"examples_count":len(row["examples"]),
            "schema_sha256":_hash(schema),
            "baseline":base,"icl":enhanced,
        })
        print("QWEN_ICL_BIRD20_CANDIDATES_FROZEN",shard,cid,flush=True)
    # Gold SQL is FIRST ACCESSED only after both arms finish generating every
    # candidate in the shard; never flows into retrieval or model prompts.
    for row in outcomes:
        source_row=by_id[row["case_id"]]
        db=_resolve_database(root,row["db_id"])
        try:
            gold=execute_readonly(db,str(source_row["SQL"]),max_rows=MAX_ROWS)
            if gold.truncated:raise ValueError("gold truncated")
            scorable=True
        except (SQLExecutionError,ValueError):
            scorable=False
            gold=None
        row["reference_scorable"]=scorable
        for arm in ("baseline","icl"):
            raw=row[arm]
            rows=raw.pop("rows")
            raw["ex"]=bool(scorable and raw["execution_success"] and
                          set(map(tuple,rows))==set(map(tuple,gold.rows))) if scorable else None
            raw["bag"]=bool(scorable and raw["execution_success"] and
                           Counter(map(tuple,rows))==Counter(map(tuple,gold.rows))) if scorable else None
    return {"protocol":PROTOCOL,"shard":shard,"cohort_sha256":cfg["cohort_sha256"],
            "model_digest":digest,"cases":outcomes}


def aggregate(shards:list[dict],*,expected_digest:str=DIGEST)->dict:
    if len(shards)!=N_SHARDS or sorted(s["shard"] for s in shards)!=list(range(N_SHARDS)):
        raise ValueError("need 5 unique complete shards")
    if len({s.get("cohort_sha256") for s in shards})!=1 or any(
        s.get("protocol")!=PROTOCOL or s.get("model_digest")!=expected_digest
        or len(s.get("cases",[]))!=4 for s in shards
    ):
        raise ValueError("cohort/protocol/model drift")
    cases=[r for s in sorted(shards,key=lambda d:d["shard"]) for r in s["cases"]]
    if len(cases)!=20 or len({r["case_id"] for r in cases})!=20 or len({r["db_id"] for r in cases})!=10:
        raise ValueError("missing cases or databases")
    valid=[r for r in cases if r["reference_scorable"]]
    rescues=sum(r["baseline"]["ex"] is False and r["icl"]["ex"] is True for r in valid)
    regressions=sum(r["baseline"]["ex"] is True and r["icl"]["ex"] is False for r in valid)
    original=sum(r["baseline"]["ex"] is True for r in valid)
    enhanced=sum(r["icl"]["ex"] is True for r in valid)
    if enhanced-original!=rescues-regressions:
        raise AssertionError("EX vs paired rescue/regression accounting mismatch")
    gained={r["db_id"] for r in valid if r["baseline"]["ex"] is False and r["icl"]["ex"] is True}
    base_bag=sum(r["baseline"]["bag"] is True for r in valid)
    new_bag=sum(r["icl"]["bag"] is True for r in valid)
    return {
        "protocol":PROTOCOL,"model_digest":expected_digest,
        "cohort_sha256":shards[0]["cohort_sha256"],
        "requested":20,"reference_scorable":len(valid),"databases":10,
        "baseline_ex":original,"icl_ex":enhanced,
        "baseline_bag":base_bag,"icl_bag":new_bag,
        "rescues":rescues,"regressions":regressions,
        "rescued_databases":len(gained),
        "baseline_latency_s":round(sum(r["baseline"]["latency_s"] for r in cases),1),
        "icl_latency_s":round(sum(r["icl"]["latency_s"] for r in cases),1),
        "positive_screen_only":rescues-regressions>=3 and regressions<=1
                             and len(gained)>=3 and new_bag>=base_bag,
        "promotion_allowed":False,
        "no_training":True,"same_model":MODEL,"gold_blind_retrieval":True,
        "caveat":"Exploratory public BIRD DEV question-disjoint from historical100, DBs overlap other experiments; filtered public TRAIN examples may appear in model pretraining; no generalized BIRD100 or production uplift implied.",
        "cases":cases,
    }


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path)
    p.add_argument("--reference",type=Path,default=Path("runs/bird-paired-dev-v5/paired.json"))
    p.add_argument("--train",type=Path)
    p.add_argument("--train-revision",type=str)
    p.add_argument("--manifest",type=Path)
    p.add_argument("--db-root",type=Path)
    p.add_argument("--shard",type=int)
    p.add_argument("--inputs",type=Path,nargs="+")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("refuse overwrite experiment output")
    if a.inputs:
        result=aggregate([json.loads(q.read_text()) for q in a.inputs])
        print("QWEN_ICL_BIRD20_FINAL",json.dumps({k:v for k,v in result.items() if k!="cases"},sort_keys=True),flush=True)
    elif a.manifest:
        if a.source is None or a.db_root is None or a.shard is None:
            p.error("shard requires source/DB/manifest/shard")
        result=run_shard(a.source,a.manifest,a.db_root,a.shard)
    else:
        if a.source is None or a.train is None or not a.train_revision:
            p.error("manifest requires source/train/train-revision")
        result=make_manifest(a.source,a.reference,json.loads(a.train.read_text()),a.train_revision)
        print("QWEN_ICL_PRE_REGISTERED",json.dumps({k:v for k,v in result.items() if k!="selected"},sort_keys=True),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
