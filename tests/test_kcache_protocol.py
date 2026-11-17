"""CPU-only tests. Never call live Ollama or use frozen BIRD HOLDOUT labels."""
from __future__ import annotations
import json
from hashlib import sha256
from pathlib import Path
import sqlite3

import pytest

from evaluation import kcache_protocol as k
from evaluation import qwen9b_kcache as g


def test_exact_existing_dev500_and_untouched_final_holdout():
    manifest=k.read_frozen_manifest(Path(k.MANIFEST))
    assert len(manifest["dev_case_ids"])==500
    assert len(manifest["holdout_case_ids"])==300
    assert len(set(manifest["dev_case_ids"]) & set(manifest["holdout_case_ids"]))==0
    with pytest.raises(PermissionError):
        k.source_cases(Path("not-used.json"),manifest,split="holdout")
    assert manifest["population"].startswith("BIRD DEV")
    assert manifest["warning"].startswith("Old BIRD")


def test_deterministic_k_candidate_seeds():
    assert k.candidate_seed("17",0)==k.candidate_seed("17",0)
    assert k.candidate_seed("17",0)!=k.candidate_seed("17",1)
    assert k.candidate_seed("17",1)!=k.candidate_seed("18",1)
    with pytest.raises(ValueError): k.candidate_seed("17",-1)


def small_db(tmp_path):
    db=tmp_path/"state.sqlite"
    connection=sqlite3.connect(db)
    connection.executescript("""
    CREATE TABLE t(a INTEGER, b TEXT);
    INSERT INTO t VALUES(1,'one'),(2,'two'),(2,'two');
    """)
    connection.commit()
    connection.close()
    return db


def test_actual_sqlite_signature_majority_and_invalid_abstention(tmp_path):
    db=small_db(tmp_path)
    a=k.candidate_outcome(db,"SELECT a FROM t")
    b=k.candidate_outcome(db,"SELECT a FROM t ORDER BY a DESC")
    c=k.candidate_outcome(db,"SELECT b FROM t")
    d=k.candidate_outcome(db,"DROP TABLE t")
    assert a["valid"] and b["valid"] and c["valid"] and not d["valid"]
    assert a["set_hash"]==b["set_hash"] and a["bag_hash"]==b["bag_hash"]
    assert a["signature"]==b["signature"]
    assert a["signature"]!=c["signature"]
    selector=k.majority_selector([a,c,b,d])
    assert selector["index"]==0
    assert selector["vote_count"]==2
    assert selector["agreement"]==0.5
    assert selector["unique_valid_result_sets"]==2
    assert k.majority_selector([d,d])["valid"]==0


def test_selector_does_not_consult_reference_gold(tmp_path):
    db=small_db(tmp_path)
    samples=[{"index":i,"sql":sql} for i,sql in enumerate([
        "SELECT a FROM t", "SELECT b FROM t",
        "SELECT a FROM t ORDER BY a DESC", "SELECT a FROM t",
    ])]
    cache={"case_id":"synthetic","db_id":"fake","k":4,"samples":samples}
    # The same chosen SQL index on two incompatible gold outcomes proves
    # candidate voting is not oracle selected.
    r1=k.per_case_offline(cache,db,"SELECT a FROM t")
    r2=k.per_case_offline(cache,db,"SELECT b FROM t")
    assert r1["selected_index"]==r2["selected_index"]==0
    assert r1["passK_oracle_set"] and r2["passK_oracle_set"]
    assert r1["selected_set_ex"] and not r2["selected_set_ex"]


def test_passK_is_upper_bound_not_selector_success(tmp_path):
    db=small_db(tmp_path)
    c={"case_id":"test","db_id":"db","k":3,"samples":[
        {"sql":"SELECT a FROM t"} for _ in range(2)]+[{"sql":"SELECT b FROM t"}]}
    x=k.per_case_offline(c,db,"SELECT b FROM t")
    assert x["passK_oracle_set"] is True
    assert x["selected_set_ex"] is False
    assert x["pass1_empirical_set"]==1/3


def test_paired_ci_determinism_by_question_and_db_cluster():
    rows=[]
    for i in range(80):
        rows.append({"case_id":str(i),"db_id":f"db{i//20}","k":8,
                     "first_set_ex":i%4==0,
                     "selected_set_ex":i%4!=3,
                     "pass1_empirical_set":0.5,
                     "passK_oracle_set":True,
                     "passK_oracle_bag":True,
                     "gold_scorable":True})
    a=k.aggregate_offline(rows)
    b=k.aggregate_offline(rows)
    assert a==b
    assert a["n"]==80 and a["k"]==8
    assert a["selector_gain_vs_first_sample"]==0.5
    assert a["passK_minus_pass1_gap"]==0.5
    assert a["question_paired_95ci"]["n_databases"]==4
    assert a["db_cluster_paired_95ci"]["unit"]=="physical_db_cluster"


def test_reject_missing_case_and_k_and_unscorable_gold():
    r={"case_id":"1","db_id":"db","k":8,"first_set_ex":False,
       "selected_set_ex":True,"pass1_empirical_set":0.2,
       "passK_oracle_set":True,"gold_scorable":True}
    with pytest.raises(ValueError):
        k.aggregate_offline([r],required_ids={"1","2"})
    with pytest.raises(ValueError):
        k.aggregate_offline([r,r])
    with pytest.raises(ValueError):
        k.aggregate_offline([r,{**r,"case_id":"2","k":16}])
    with pytest.raises(ValueError):
        k.aggregate_offline([{**r,"gold_scorable":False}])


def test_candidate_cache_contract_requires_all_original_model_sample_seeds():
    rows=[{"question_id":"17","db_id":"db"}]
    samples=[{"index":i,"seed":k.candidate_seed("17",i),"sql":"SELECT 1"}
             for i in range(8)]
    d={"status":"CANDIDATE_CACHE_COMPLETE","protocol":k.PROTOCOL,
       "source_sha256":k.SOURCE_SHA256,
       "model_digest":k.MODEL_DIGEST,"model":k.MODEL_NAME,"split":"dev",
       "no_gold_sql_in_cache":True,"no_repair":True,"shard":0,"shards":1,"k":8,
       "cases":[{"case_id":"17","db_id":"db","k":8,"samples":samples}]}
    g.validate_cache(d,rows)
    samples[4]["seed"]+=1
    with pytest.raises(ValueError,match="seed"):
        g.validate_cache(d,rows)
    with pytest.raises(ValueError):
        g.source_shard(rows,20,16)


def test_train_cpu_bench_is_original_prompt_source_not_DEV_gold(tmp_path):
    source=tmp_path/"simulated.jsonl"
    source.write_text(json.dumps({"original_user":"Question: X\nSchema context:\nFake(t)",
                                  "source_index":3})+"\n")
    with pytest.raises(ValueError,match="expected previously frozen"):
        g.speed("http://localhost:9",source)


def test_question_evaluation_cannot_relax_final_holdout():
    assert k.MIN_EFFECT==0.05 and k.MIN_ORACLE_GAP==0.08
    assert k.BOOTSTRAP_ROUNDS>=2000
    assert "holdout" not in g.generate.__code__.co_varnames
