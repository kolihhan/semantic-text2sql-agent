"""Two frozen, gold-blind selectors on the identical 4000 original Qwen9B K8 SQLs.

Hypothesis: exact SQL-program duplicates from K8 stochastic samples overweight
one answer in result-set plurality. Compare unweighted distinct-program-vote
vs original eight-sample result-vote; no generated SQL edits, no new Qwen calls.
Conservative SQL normalization collapses whitespace only OUTSIDE quotations,
case outside quotation, and a single final semicolon; no SQL AST rewrites.

No SQL answer/reference, selected candidate correctness, or publisher metadata
is available anywhere in choose_index(). Selector decisions are frozen BEFORE
publisher outcomes are opened. This does not create any missing candidates.
"""
from __future__ import annotations
from collections import Counter
from hashlib import sha256, sha1
import json
from pathlib import Path
import random
import sqlite3
import time

FIXED_MODEL="qwen3.5:9b-q4_K_M"
FIXED_MODEL_DIGEST="56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
CACHE_PROTOCOL="FROZEN-BIRD-DEV500-QWEN35-9B-KCACHE-V1"
DEV_SOURCE_SHA="ffd8018378ddb1a8794753e0a31cfc81862ff7318a5184c22f3dc4ce03a03feb"
MANIFEST_GIT_BLOB="692fc84b881746d32f70469d03c7026fcc67291a"
DEV_ZIP_SHA="cdd6d19faeb45a23970b98d3ef6c40a87987c95459c2cf12076897a60cf5a630"
PROTOCOL="FROZEN-500-ORIGINAL-K8-EXACT-DISTINCT-PROGRAM-VOTE-V1"
SHARDS=32
SAMPLES_PER_CASE=8
MIN_PROGRAM_COLLAPSE_AMBIGUITY=80
MIN_AMBIGUITY_DB_SPREAD=5
MIN_PROMOTION_EX_NET=25
MIN_PROMOTION_DB_CI_LOWER=0.0
SEED=20261118
MAX_ROWS=50000
SQL_SECONDS=8.


def git_blob_sha(raw:bytes)->str:
    return sha1(f"blob {len(raw)}\0".encode()+raw).hexdigest()


def read_frozen_manifest(path:Path)->list[str]:
    raw=path.read_bytes()
    if git_blob_sha(raw)!=MANIFEST_GIT_BLOB:
        raise ValueError("original DEV500 and protected final manifest blob drift")
    data=json.loads(raw)
    dev=data.get("dev_case_ids")
    final=data.get("holdout_case_ids")
    if (data.get("source_sha256")!=DEV_SOURCE_SHA or
        data.get("dev_size")!=500 or data.get("holdout_size")!=300 or
        not isinstance(dev,list) or len(dev)!=500 or len(set(dev))!=500 or
        not isinstance(final,list) or len(final)!=300 or
        set(dev)&set(final)):
        raise ValueError("invalid frozen original DEV500 IDs / reserved final")
    return [str(i) for i in dev]


def seed_for(case_id:str,index:int)->int:
    return int(sha256(f"generation-v1:{case_id}:{index}".encode()).hexdigest()[:8],16)


def conservative_program(sql:str)->str:
    """Do not normalize quoted literal contents; do not infer SQL equivalence."""
    raw=sql.strip()
    if raw.endswith(";"):
        raw=raw[:-1].rstrip()
    out=[]
    state=None
    pending=False
    i=0
    while i<len(raw):
        c=raw[i]
        if state:
            out.append(c)
            if c==state:
                if i+1<len(raw) and raw[i+1]==state:
                    i+=1;out.append(raw[i])
                else:
                    state=None
        elif c in ("'",'"',chr(96)):
            if pending and out:out.append(" ")
            pending=False
            out.append(c)
            state=c
        elif c.isspace():
            pending=True
        else:
            if pending and out:out.append(" ")
            pending=False
            out.append(c.lower() if c.isascii() else c)
        i+=1
    return "".join(out)


def validate_cache(cache:dict,shard:int)->list[dict]:
    if (cache.get("status")!="CANDIDATE_CACHE_COMPLETE"
        or cache.get("protocol")!=CACHE_PROTOCOL
        or cache.get("source_sha256")!=DEV_SOURCE_SHA
        or cache.get("manifest_git_blob_sha1")!=MANIFEST_GIT_BLOB
        or cache.get("model")!=FIXED_MODEL
        or cache.get("model_digest")!=FIXED_MODEL_DIGEST
        or cache.get("split")!="dev" or cache.get("k")!=8
        or cache.get("shard")!=shard or cache.get("shards")!=SHARDS
        or cache.get("num_predict")!=384
        or cache.get("temperature")!=0.7
        or cache.get("no_gold_sql_in_cache") is not True
        or cache.get("no_repair") is not True):
        raise ValueError("original fixed-Qwen K8 cache source or sampling protocol drift")
    cases=cache["cases"]
    if len(cases) not in (15,16) or len({r["case_id"] for r in cases})!=len(cases):
        raise ValueError("cache shard case count or unique IDs drift")
    for row in cases:
        sample=row.get("samples")
        if row.get("k")!=8 or not isinstance(sample,list) or len(sample)!=8:
            raise ValueError("original 8 Qwen SQL candidates not complete")
        for k,s in enumerate(sample):
            if (s["index"]!=k or s["seed"]!=seed_for(row["case_id"],k)
                or s["num_predict"]!=384 or not isinstance(s["sql"],str)):
                raise ValueError("original Qwen sample identity or seed drift")
    return cases


def read_all_caches(directory:Path,allowed_dev_ids:list[str])->list[dict]:
    all_rows=[]
    for shard in range(SHARDS):
        p=directory/f"k8-dev500-candidate-cache-{shard}.json"
        data=json.loads(p.read_text("utf-8"))
        for row in validate_cache(data,shard):
            all_rows.append({**row,"original_shard":shard})
    ids=[str(x["case_id"]) for x in all_rows]
    if len(ids)!=500 or len(set(ids))!=500 or set(ids)!=set(allowed_dev_ids):
        raise ValueError("not 500 unique source-frozen original DEV questions")
    return all_rows


def choose_index(sqls:list[str],outcomes:list[dict],*,dedup_program:bool)->dict:
    """Never sees a publisher gold SQL, score label, EX, target row or oracle."""
    if len(sqls)!=8 or len(outcomes)!=8:
        raise ValueError("K8 candidates only")
    indices=[]
    seen=set()
    for i,(sql,out) in enumerate(zip(sqls,outcomes)):
        if dedup_program:
            program=conservative_program(sql)
            if program in seen:continue
            seen.add(program)
        if out.get("valid") and out.get("signature"):
            indices.append((i,out["signature"]))
    if not indices:
        return {"index":0,"votes":0,"groups":0,"valid":0}
    counter=Counter(sig for _,sig in indices)
    maximum=max(counter.values())
    winner=next(i for i,sig in indices if counter[sig]==maximum)
    return {"index":winner,"votes":maximum,"groups":len(counter),"valid":len(indices)}


def _safe_db(dbroot:Path,dbid:str)->Path:
    if "/" in dbid or ".." in dbid or "\\" in dbid:
        raise ValueError("unsafe DB ID")
    file=dbroot/dbid/f"{dbid}.sqlite"
    if not file.is_file():
        raise FileNotFoundError("missing original physical BIRD DEV sqlite: "+dbid)
    return file


def _bounded_fingerprint(db:Path,sql:str)->dict:
    from semantic_sql.execution import (
        SQLExecutionError,_open_readonly_connection,validate_readonly_sql
    )
    if not sql.strip():
        return {"valid":False,"reason":"empty","signature":None,
                "set_hash":None,"bag_hash":None}
    stop=time.monotonic()+SQL_SECONDS
    conn=None
    try:
        query=validate_readonly_sql(sql)
        conn,_=_open_readonly_connection(db)
        conn.set_progress_handler(lambda:int(time.monotonic()>stop),2000)
        cur=conn.execute(query)
        cols=tuple(x[0] for x in (cur.description or ()))
        rows=cur.fetchmany(MAX_ROWS+1)
        if len(rows)>MAX_ROWS:
            raise ValueError("row_cap")
        from collections import Counter as C
        lines=[json.dumps(list(row),ensure_ascii=False,sort_keys=True,default=str,
                          separators=(",",":")) for row in rows]
        distinct=sorted(set(lines))
        bags=sorted(C(lines).items())
        set_hash=sha256(json.dumps(distinct).encode()).hexdigest()
        bag_hash=sha256(json.dumps(bags).encode()).hexdigest()
        signature=sha256(json.dumps([len(cols),set_hash]).encode()).hexdigest()
        return {"valid":True,"reason":"ok","signature":signature,
                "set_hash":set_hash,"bag_hash":bag_hash}
    except (sqlite3.Error,SQLExecutionError,ValueError,OSError) as exc:
        reason=("query_timeout" if time.monotonic()>stop else
                type(exc).__name__)
        return {"valid":False,"reason":reason,"signature":None,
                "set_hash":None,"bag_hash":None}
    finally:
        if conn is not None:
            conn.set_progress_handler(None,0)
            conn.close()


def score_original_candidates(db:Path,sqls:list[str])->list[dict]:
    """Memoized but fully physical, read-only SQL execution; no gold."""
    outcomes={}
    result=[]
    for sql in sqls:
        # Cache conservative duplicate programs only, never combine
        # different SQL programs just because of spelling similarity.
        key=conservative_program(sql)
        if key not in outcomes:
            outcomes[key]=_bounded_fingerprint(db,sql)
        result.append(outcomes[key])
    return result


def dev_rows(source:Path)->dict[str,dict]:
    raw=source.read_bytes()
    if sha256(raw).hexdigest()!=DEV_SOURCE_SHA:
        raise ValueError("original source bytes not pinned")
    data=json.loads(raw)
    if len(data)!=1534:
        raise ValueError("original DEV pool size mismatch")
    index={str(r["question_id"]):r for r in data}
    if len(index)!=1534:
        raise ValueError("source duplicate question IDs")
    return index


def bootstrap(records:list[dict],key_old:str,key_new:str,*,cluster:bool,
              rounds:int=4000)->dict:
    if not records:
        raise ValueError("no original source paired records")
    rng=random.Random(SEED)
    by_db={}
    deltas=[]
    for r in records:
        v=float(bool(r[key_new]))-float(bool(r[key_old]))
        by_db.setdefault(r["db_id"],[]).append(v)
        deltas.append(v)
    groups=sorted(by_db)
    sims=[]
    for i in range(rounds):
        if cluster:
            selected=[rng.choice(groups) for _ in groups]
            draw=[v for db in selected for v in by_db[db]]
        else:
            draw=[rng.choice(deltas) for _ in deltas]
        sims.append(sum(draw)/len(draw))
    sims.sort()
    return {"mean":sum(deltas)/len(deltas),
            "lower95":sims[int(rounds*0.025)],
            "upper95":sims[int(rounds*0.975)-1],
            "n":len(records),"database_clusters":len(groups),
            "unit":"database" if cluster else "question"}


def aggregate_census(cases:list[dict],score_summaries:dict[str,dict]|None=None)->dict:
    n=len(cases)
    duplicate=[]
    entropy=[]
    by_db=set()
    for row in cases:
        programs=[conservative_program(s["sql"]) for s in row["samples"]]
        uniq=len(set(programs))
        if uniq<8:
            duplicate.append(row)
        if score_summaries:
            old=score_summaries.get(str(row["case_id"]))
            if (old and old.get("unique_valid_result_sets",0)>=2 and uniq<8):
                entropy.append(row);by_db.add(row["db_id"])
    return {
        "n":n,"duplicate_program_questions":len(duplicate),
        "fully_collapsed_8_of_8_cases":sum(
            len(set(conservative_program(s["sql"]) for s in row["samples"]))==1
            for row in cases),
        "mean_unique_SQL_programs_per_case":sum(
            len(set(conservative_program(s["sql"]) for s in row["samples"]))
            for row in cases)/n,
        "score_metadata_available":len(score_summaries or {}),
        "cases_with_duplication_and_disagreeing_SQL_results":len(entropy),
        "database_clusters_in_ambiguous_duplications":len(by_db),
        "source_only_positive_gate":(
            len(entropy)>=MIN_PROGRAM_COLLAPSE_AMBIGUITY and
            len(by_db)>=MIN_AMBIGUITY_DB_SPREAD
        ),
        "model_calls":0,"DEV500_reinference":False,
        "HOLDOUT300_answers_accessed":False
    }
