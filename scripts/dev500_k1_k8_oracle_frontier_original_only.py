"""Zero-Qwen DEV500 original K1↔K8 candidate-availability/selection frontier.

Uses already-frozen original BIRD DEV500 Qwen9B K8 physical score worker
artifacts and previously frozen first-SQL source-keyword taxonomy. Never
loads official/publisher SQL, opens a physical DB or calls a model.
The measured 8-second SET result matching is historical, not new Guarded EX.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path

SRC_SHA="ffd8018378ddb1a8794753e0a31cfc81862ff7318a5184c22f3dc4ce03feb"
MODEL_SHA="56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
WORKER_PROTO="BIRD-DEV500-SAME-ORIGINAL-QWEN9B-K8-CACHE-UNIFORM-8S-SQLITE-PREFIX-FRONTIER-V1"
PROTOCOL="BIRD-DEV500-QWEN9B-K1-K8-CANDIDATE-FRONTIER-REUSE-SCORES-V1"

def read(path):
    return json.loads(path.read_text(encoding="utf8"))

def run(score_file,workers_dir):
    frozen=read(score_file)
    assert frozen["n"]==500
    assert frozen["k1_right"]==203 and frozen["k1_wrong"]==297
    assert frozen["k8_right"]==233 and frozen["publisher_reference_scorable"]==499
    assert frozen["same_original_32_cache_shards_and_8_physical_workers"]
    labels={x["case_id"]:x for x in frozen["first_candidate_labels"]}
    assert len(labels)==500
    paths=sorted(workers_dir.glob("prefix-frontier-worker-*.json"))
    assert len(paths)==8
    original={}
    shard_ids=set()
    for path in paths:
        d=read(path)
        assert d["protocol"]==WORKER_PROTO
        if d.get("model_digest")!=MODEL_SHA or d.get("source_sha256")!=SRC_SHA:
            raise ValueError(
                "Pinned physical worker mismatch "+path.name
                +": model="+str(d.get("model_digest")==MODEL_SHA)
                +", source="+str(d.get("source_sha256")==SRC_SHA))
        assert d["model_calls"]==0 and d["status"]=="SOURCE_LOCKED_K_PREFIX_WORKER_COMPLETE"
        wi=int(d["worker"])
        assert wi in range(8) and wi not in shard_ids
        shard_ids.add(wi)
        for row in d["cases"]:
            cid=str(row["case_id"])
            assert cid not in original
            original[cid]=row
    assert shard_ids==set(range(8)) and set(original)==set(labels)
    c=Counter()
    bydb=defaultdict(Counter)
    bytag=defaultdict(Counter)
    cohort_ids=defaultdict(list)
    by_case=[]
    for cid in sorted(labels,key=int):
        x=labels[cid]
        w=original[cid]
        assert x["db"]==w["db_id"]
        assert x["first_sql_reference_set_match"]==bool(w["k1_set_ex"])
        assert x["k8_selected_reference_set_match"]==bool(w["k8_set_ex"])
        a=bool(w["k1_set_ex"])
        b=bool(w["k8_set_ex"])
        oracle=bool(w["pass8_oracle_set"])
        if a and not oracle:raise ValueError("K1 right but none of 8 correct")
        if b and not oracle:raise ValueError("K8 right but none of 8 correct")
        if not oracle:kind="K8_ALL_EIGHT_WRONG"
        elif not b:kind="K8_SELECTOR_MISSED_AVAILABLE_CORRECT"
        elif a:kind="K1_RIGHT_K8_RIGHT"
        else:kind="K1_WRONG_K8_RIGHT"
        c[kind]+=1
        c["K1_RIGHT"]+=int(a)
        c["K8_RIGHT"]+=int(b)
        c["K8_ORACLE_HAS_MATCH"]+=int(oracle)
        c["PUBLISHER_SCORABLE"]+=int(bool(w["gold_scorable"]))
        c["K1_TO_K8_RESCUE"]+=int(not a and b)
        c["K1_TO_K8_LOSS"]+=int(a and not b)
        cohort_ids[kind].append(cid)
        db=bydb[x["db"]]
        db["n"]+=1
        db["k1_right"]+=int(a)
        db["k8_right"]+=int(b)
        db["oracle_any"]+=int(oracle)
        db["generator_no_correct"]+=int(not oracle)
        db["selector_miss"]+=int(oracle and not b)
        for tag in x["question_tags"]:
            s=bytag[tag]
            s["n"]+=1
            s["k1_right"]+=int(a)
            s["k8_right"]+=int(b)
            s["oracle_any"]+=int(oracle)
            s["generator_no_correct"]+=int(not oracle)
            s["selector_miss"]+=int(oracle and not b)
        by_case.append({"case_id":cid,"db":x["db"],"outcome":kind,
                        "first_original_right":a,"k8_selected_right":b,
                        "k8_oracle_any_right":oracle,"public_source_tags":x["question_tags"]})
    assert c["K1_RIGHT"]==203 and c["K8_RIGHT"]==233
    assert c["K8_ORACLE_HAS_MATCH"]==285
    assert c["K8_ALL_EIGHT_WRONG"]==215
    assert c["K8_SELECTOR_MISSED_AVAILABLE_CORRECT"]==52
    assert c["K1_RIGHT_K8_RIGHT"]==197
    assert c["K1_WRONG_K8_RIGHT"]==36
    assert c["K1_TO_K8_LOSS"]==6
    assert sum(c[k] for k in ("K8_ALL_EIGHT_WRONG","K8_SELECTOR_MISSED_AVAILABLE_CORRECT",
                               "K1_RIGHT_K8_RIGHT","K1_WRONG_K8_RIGHT"))==500
    return {
        "protocol":PROTOCOL,"n":500,"model_sha":MODEL_SHA,
        "original_bird_source_sha256":SRC_SHA,
        "metrics":"8-second timeout bounded SQLite SET-match (not official end-to-end Guarded EX)",
        "aggregate":dict(sorted(c.items())),
        "by_physical_database":{k:dict(v) for k,v in sorted(bydb.items())},
        "source_question_tags":{k:dict(v) for k,v in sorted(bytag.items())},
        "four_exclusive_cohorts_case_ids":dict(cohort_ids),
        "per_case_historical_label":by_case,
        "not_causal_semantic_labels":True,
        "no_gold_query_loaded":True,"no_new_qwen_or_sql_execution":True,
        "holdout300_untouched":True,"no_training":True,
        "decision":"215/500 have no correct K8 candidate; 52/500 have a correct candidate but K8 selector misses. Improve candidate generation first, independently audit selection.",
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--frozen-k1",required=True,type=Path)
    p.add_argument("--workers-dir",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError("immutable output already exists")
    d=run(a.frozen_k1,a.workers_dir)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(d,sort_keys=True,indent=2)+"\n")
    print("HISTORICAL_K1_K8_SOURCE_ORACLE_FRONTIER",json.dumps({
        "n":d["n"],"aggregate":d["aggregate"],
        "new_model_calls":0},sort_keys=True),flush=True)

if __name__=="__main__":main()
