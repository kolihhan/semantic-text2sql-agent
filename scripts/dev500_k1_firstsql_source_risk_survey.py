"""BIRD DEV500 frozen first-Qwen-SQL semantic risk census: NO new inference.

Stage SOURCE sees original 32 Qwen K8 caches (uses index=0 only), a
projection of public question/difficulty/db WITHOUT publisher SQL, and frozen
V2 manifest. Source-only risk flags are NOT correctness/root-cause labels.

Stage SCORE sees an immutable SOURCE artifact and 8 original physical workers
(K1 and K8 correctness labels); no publisher reference SQL or new SQLite eval.
No tuning, original prompt changes, QLoRA or HOLDOUT300 access.

The goal is a real K1-right and K1-wrong control population with source-only
features, not a hand-chosen incorrect-only table.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re

SRC_SHA="ffd8018378ddb1a8794753e0a31cfc81862ff7318a5184c22f3dc4ce03a03feb"
MODEL_SHA="56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
CACHE_PROTO="FROZEN-BIRD-DEV500-QWEN35-9B-KCACHE-V1"
WORKER_PROTO="BIRD-DEV500-SAME-ORIGINAL-QWEN9B-K8-CACHE-UNIFORM-8S-SQLITE-PREFIX-FRONTIER-V1"
PROTO="BIRD-DEV500-FROZEN-K1-SOURCE-RISK-AND-CONTROLS-V1"

def sha(data):
    if isinstance(data,str):data=data.encode()
    return sha256(data).hexdigest()

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def flags(question,sql):
    """Question/SQL proxy contradictions, NOT true semantic error diagnoses."""
    q=question.casefold()
    s=sql.casefold()
    matched=[]
    has=lambda pat:re.search(pat,s,re.I|re.S) is not None
    asks=lambda pat:re.search(pat,q,re.I|re.S) is not None
    if asks(r"\b(how many|number of|count of)\b") and has(r"\bcount\s*\(\s*\*\s*\)") and has(r"\bjoin\b"):
        matched.append("P_COUNT_STAR_AFTER_JOIN")
    if asks(r"\b(percent(?:age)?|ratio|proportion|fraction|growth rate)\b") and not ("/" in s or has(r"\bavg\s*\(")):
        matched.append("P_RATIO_WITHOUT_DIVISION")
    if asks(r"\b(top\s*\d*|highest|lowest|largest|smallest|youngest|oldest|most|least)\b") and not has(r"\border\s+by\b") and not has(r"\b(max|min)\s*\("):
        matched.append("P_RANK_WITHOUT_ORDER_OR_EXTREMA")
    if asks(r"\b(both|simultaneously|each of|all of)\b") and has(r"\bin\s*\(") and not has(r"\bhaving\b") and not has(r"\bintersect\b"):
        matched.append("P_CONJUNCTION_AS_MEMBERSHIP")
    if asks(r"\b(without|neither|never|except|not having)\b") and not has(r"\b(not|except|is null)\b"):
        matched.append("P_NEGATION_WITHOUT_NEGATIVE_OPERATOR")
    if asks(r"\b(before|after|between|since|during)\b") and asks(r"\b(19|20)\d{2}\b") and not has(r"\b(where|between|having)\b"):
        matched.append("P_DATE_SCOPE_WITHOUT_FILTER")
    return sorted(matched)

def question_tags(question):
    q=question.casefold()
    tags=[]
    rules={
        "Q_COUNT":r"\b(how many|number of|count of)\b",
        "Q_RATIO":r"\b(percent(?:age)?|ratio|proportion|fraction|growth rate)\b",
        "Q_RANK":r"\b(top\s*\d*|highest|lowest|largest|smallest|youngest|oldest|most|least)\b",
        "Q_CONJUNCTION":r"\b(both|simultaneously|each of|all of)\b",
        "Q_NEGATION":r"\b(without|neither|never|except|not having)\b",
        "Q_TEMPORAL":r"\b(before|after|between|since|during)\b",
        "Q_RELATION":r"\b(connected|parent|child|associated|whose|belong|own|joined)\b",
    }
    for name,pat in rules.items():
        if re.search(pat,q):
            tags.append(name)
    return sorted(tags)

def source(args):
    qproject=read(args.question_only)
    manifest=read(args.manifest)
    assert qproject["source_sha256"]==SRC_SHA
    expected={str(i) for i in manifest["dev_case_ids"]}
    holdout={str(i) for i in manifest["holdout_case_ids"]}
    assert manifest["source_sha256"]==SRC_SHA
    assert len(expected)==500 and len(holdout)==300 and not expected&holdout
    qrows={str(x["case_id"]):x for x in qproject["questions"]}
    assert len(qrows)==500 and set(qrows)==expected
    cache_files=sorted(args.cache_dir.glob("k8-dev500-candidate-cache-*.json"))
    assert len(cache_files)==32,("frozen cache shards",len(cache_files))
    records={}
    shard_ids=set()
    for path in cache_files:
        d=read(path)
        assert d["protocol"]==CACHE_PROTO and d["status"]=="CANDIDATE_CACHE_COMPLETE"
        assert d["source_sha256"]==SRC_SHA and d["model_digest"]==MODEL_SHA
        assert d["no_gold_sql_in_cache"] and d["k"]==8
        shard=int(d["shard"])
        assert shard in range(32) and shard not in shard_ids
        shard_ids.add(shard)
        for x in d["cases"]:
            cid=str(x["case_id"])
            assert cid not in records and cid in expected
            samples=sorted(x["samples"],key=lambda z:z["index"])
            assert [z["index"] for z in samples]==list(range(8))
            original=samples[0]
            q=qrows[cid]["question"]
            sql=original["sql"]
            assert x["question_sha256"]==sha(q)
            assert x["db_id"]==qrows[cid]["db_id"]
            records[cid]={
                "case_id":cid,"db":x["db_id"],
                "difficulty":qrows[cid].get("difficulty") or "unlabeled",
                "question_sha256":sha(q),"original_sql_sha256":sha(sql),
                "question_tags":question_tags(q),"source_only_risk_flags":flags(q,sql),
                "original_first_done_reason":original.get("done_reason"),
                "first_sql_is_empty":not bool(sql.strip()),
                "first_sql_starts_select":bool(re.match(r"\s*(select|with)\b",sql,re.I)),
                "first_sql_chars":len(sql),
                "question_excerpt":" ".join(q.split())[:160],
                "original_sql_excerpt":" ".join(sql.split())[:260],
                "publisher_reference_sql_unavailable":True
            }
    assert shard_ids==set(range(32)) and len(records)==500
    rows=[records[cid] for cid in sorted(records,key=int)]
    # Deterministic source-only review cohort, 4 per DB, selected BEFORE score.
    bydb=defaultdict(list)
    for row in rows:bydb[row["db"]].append(row)
    review=[]
    for db,items in sorted(bydb.items()):
        # Some risk-positive and some general items; never consult K1 labels.
        flagged=sorted([x for x in items if x["source_only_risk_flags"]],
                       key=lambda x:sha(db+"|flagged|"+x["case_id"]))
        unflagged=sorted([x for x in items if not x["source_only_risk_flags"]],
                         key=lambda x:sha(db+"|unflagged|"+x["case_id"]))
        chosen=(flagged[:2]+unflagged[:2])
        for row in sorted(items,key=lambda x:sha(db+"|fill|"+x["case_id"])):
            if len(chosen)>=4:break
            if row not in chosen:chosen.append(row)
        assert len(chosen)==4
        review.extend(x["case_id"] for x in chosen)
    assert len(review)==44 and len(set(review))==44 and len(bydb)==11
    return {
      "protocol":PROTO,"phase":"SOURCE_FROZEN","n":500,"dbs":11,
      "model_digest":MODEL_SHA,"source_sha256":SRC_SHA,
      "risk_features_blind_to_reference":True,"first_sample_index":0,
      "review_44_ids_source_preselected":review,
      "rows":rows,"no_publisher_reference_sql_used":True,
      "do_not_interpret_risk_as_root_cause":True,
      "no_new_qwen_calls":True,"holdout300_unread":True
    }

def score(args):
    src=read(args.source_frozen)
    assert src["protocol"]==PROTO and src["phase"]=="SOURCE_FROZEN"
    assert src["n"]==500 and len(src["rows"])==500
    worker_paths=list(args.workers_dir.glob("prefix-frontier-worker-*.json"))
    assert len(worker_paths)==8,("workers",len(worker_paths))
    scores={}
    worker_indices=set()
    for path in worker_paths:
        d=read(path)
        assert d["protocol"]==WORKER_PROTO and d["source_sha256"]==SRC_SHA
        assert d["model_digest"]==MODEL_SHA and d["model_calls"]==0
        assert d["status"]=="SOURCE_LOCKED_K_PREFIX_WORKER_COMPLETE"
        wi=int(d["worker"])
        assert wi in range(8) and wi not in worker_indices
        worker_indices.add(wi)
        for row in d["cases"]:
            cid=str(row["case_id"])
            assert cid not in scores
            scores[cid]=row
    assert len(scores)==500 and worker_indices==set(range(8))
    labels=[]
    subsets=defaultdict(lambda:Counter())
    aggregate=Counter()
    for item in src["rows"]:
        rid=item["case_id"]
        w=scores[rid]
        assert w["db_id"]==item["db"]
        k1=bool(w["k1_set_ex"])
        k8=bool(w["k8_set_ex"])
        goldscorable=bool(w["gold_scorable"])
        aggregate["total"]+=1
        aggregate["k1_correct"]+=int(k1)
        aggregate["k8_correct"]+=int(k8)
        aggregate["reference_scorable"]+=int(goldscorable)
        for label in ["ALL"]+item["question_tags"]+item["source_only_risk_flags"]:
            c=subsets[label]
            c["n"]+=1
            c["first_sql_right"]+=int(k1)
            c["first_sql_wrong"]+=int(not k1)
            c["k8_right"]+=int(k8)
        labels.append({
            "case_id":rid,"db":item["db"],
            "first_sql_reference_set_match":k1,
            "reference_scorable":goldscorable,
            "k8_selected_reference_set_match":k8,
            "first_sql_is_empty":item["first_sql_is_empty"],
            "first_sql_starts_select":item["first_sql_starts_select"],
            "first_generation_hit_384_token_cap":item["original_first_done_reason"]=="length",
            "question_tags":item["question_tags"],
            "source_only_risk_flags":item["source_only_risk_flags"],
        })
    assert aggregate["total"]==500 and aggregate["k1_correct"]==203
    assert aggregate["k8_correct"]==233 and aggregate["reference_scorable"]==499
    bydb={}
    for db in sorted({z["db"] for z in labels}):
        dat=[z for z in labels if z["db"]==db]
        bydb[db]={"n":len(dat),
                  "k1_right":sum(x["first_sql_reference_set_match"] for x in dat),
                  "k1_wrong":sum(not x["first_sql_reference_set_match"] for x in dat),
                  "k1_generation_length_capped":sum(x["first_generation_hit_384_token_cap"] for x in dat)}
    reviewed=set(src["review_44_ids_source_preselected"])
    sample=[{"case_id":z["case_id"],"db":z["db"],"k1_right":z["first_sql_reference_set_match"],
             "question_tags":z["question_tags"],"source_only_risk_flags":z["source_only_risk_flags"]}
            for z in labels if z["case_id"] in reviewed]
    assert len(sample)==44
    rows={k:dict(v) for k,v in sorted(subsets.items())}
    for val in rows.values():
        val["first_wrong_rate"]=round(val["first_sql_wrong"]/val["n"],5)
        val["first_right_rate"]=round(val["first_sql_right"]/val["n"],5)
    return {
      "protocol":PROTO,"phase":"POST_SOURCE_HISTORICAL_PHYSICAL_K1_LABELS",
      "n":500,"k1_right":203,"k1_wrong":297,"k8_right":233,
      "publisher_reference_scorable":499,
      "source_only_risk_strata":rows,"by_database":bydb,
      "source_review_44_with_labels":sample,"first_candidate_labels":labels,
      "k1_wrong_causal_root_annotated":False,
      "diagnostic_flags_are_NOT_causal_errors":True,
      "same_original_32_cache_shards_and_8_physical_workers":True,
      "no_new_qwen_calls":True,"no_qlora_or_sql_train":True,
      "no_new_guarded_bird_ex":True,"holdout300_untouched":True,
      "next_gate":"Manual source-first role/grain/condition review of frozen 44, including correct controls; compare gold only after annotation and require independent prospective matched A/B."
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=("source","score"))
    for name in ("cache_dir","question_only","manifest","source_frozen","workers_dir","out"):
        p.add_argument("--"+name.replace("_","-"),type=Path)
    args=p.parse_args()
    if args.out is None: p.error("--out required")
    if args.out.exists():raise FileExistsError("immutable artifact output exists")
    if args.stage=="source":
        assert args.cache_dir and args.question_only and args.manifest and not args.workers_dir
        result=source(args)
    else:
        assert args.source_frozen and args.workers_dir and not args.cache_dir
        result=score(args)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf8")
    print("K1_SOURCE_FIRST_SURVEY",json.dumps({
        "phase":result["phase"],"n":result["n"],
        "first_correct":result.get("k1_right"),
        "first_wrong":result.get("k1_wrong"),
        "selected_review":len(result.get("review_44_ids_source_preselected",result.get("source_review_44_with_labels",[]))),
        "qwen_calls":0},sort_keys=True),flush=True)

if __name__=="__main__":main()
