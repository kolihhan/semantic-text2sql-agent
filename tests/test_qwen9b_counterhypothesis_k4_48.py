"""All-CPU source-independent protocol guards for original Qwen structural K4."""
from __future__ import annotations
import inspect
import json
import sqlite3

from experiments.qwen9b_counterhypothesis_k4_48 import (
    MODEL,DIGEST,N,N_DB,PER_DB,BASE_SEEDS,COUNTER_SEEDS,GATE,
    choose_plurality,contrast_user,intent_class,shape_signature,
    physical_candidates,summarize,PROTOCOL
)


def test_same_original_qwen_model_and_matched_k4_no_new_training():
    assert MODEL=="qwen3.5:9b-q4_K_M"
    assert DIGEST=="56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
    assert (N,N_DB,PER_DB)==(48,8,6)
    assert BASE_SEEDS==(42,43,44,45) and COUNTER_SEEDS==(44,45)
    assert GATE["min_oracle_at4_net"]==4 and GATE["min_selected_net"]==3
    assert "real" not in PROTOCOL.lower()
    from experiments import qwen9b_counterhypothesis_k4_48 as e
    src=inspect.getsource(e.infer)
    assert src.count("generate(")==7  # function declaration plus 6 physical original calls
    assert src.index("control_choice=choose_plurality")<src.index("cases.append(")
    assert "actual[" not in src and '["SQL"]' not in src


def test_structural_counterhypothesis_changes_task_based_mechanism():
    assert intent_class("What percentage of X are Y?")=="ratio"
    assert intent_class("How many distinct customers?")=="count"
    assert intent_class("Which company has the highest revenue?")=="ranking"
    assert intent_class("Who owns the paper?")=="relation"
    orig="Question: How many unique users?\nSchema context:\nusers(id)"
    c=contrast_user(orig,("SELECT COUNT(*) FROM users",
                          "SELECT COUNT(id) FROM users"),
                    "How many unique users?",0)
    assert "COUNT" in c and "DISTINCT" in c
    assert "UNVERIFIED" in c and "no analysis" in c
    assert "users(id)" in c
    assert "gold" not in c.lower()
    assert shape_signature("SELECT COUNT(*) FROM users")!=shape_signature(
        "SELECT name FROM customers JOIN users ON users.id=customers.id")


def test_blind_plurality_tie_and_invalid_candidates():
    def cand(h=None):
        return {"ok":h is not None,"set_sha256":h,"bag_sha256":h}
    assert choose_plurality([cand("a"),cand("b"),cand("b"),cand("a")])["index"]==0
    assert choose_plurality([cand(),cand("b"),cand("a"),cand("a")])["index"]==2
    assert not choose_plurality([cand(),cand(),cand(),cand()])["ok"]


def test_physical_sqlite_preflight_is_readonly_and_hash_is_structural_metadata(tmp_path):
    db=tmp_path/"real.sqlite"
    conn=sqlite3.connect(db)
    conn.executescript("CREATE TABLE users(id INTEGER);INSERT INTO users VALUES(1),(2),(3);")
    conn.close()
    out=physical_candidates(db,[
        "SELECT COUNT(*) FROM users",
        "SELECT COUNT(id) FROM users",
        "SELECT name FROM unknown",
        "DROP TABLE users"])
    assert [x["ok"] for x in out]==[True,True,False,False]
    assert out[0]["set_sha256"]==out[1]["set_sha256"]
    conn=sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]==3
    conn.close()


def test_full8_db_reporting_and_oracle_screen_never_promotes(tmp_path):
    dbs=[f"db_{i}" for i in range(8)]
    manifest=tmp_path/"freeze.json"
    manifest.write_text(json.dumps({"protocol":PROTOCOL,
                                    "source_sha256":"synthetic-new48",
                                    "dbs":dbs}))
    paths=[]
    for sh,db in enumerate(dbs):
        cases=[]
        for j in range(6):
            correct=(sh==0 and j==0)
            cases.append({"source_index":sh*6+j,"db_id":db,
                "reference_scorable":True,
                "control_selected_set_EX":correct,
                "treatment_selected_set_EX":correct,
                "control_selected_bag_EX":correct,
                "treatment_selected_bag_EX":correct,
                "control_oracle_at4":correct,
                "treatment_oracle_at4":correct,
                "control_executable":4,"treatment_executable":4,
                "control_unique_structures":2,"treatment_unique_structures":3,
                "control_wall_s":5.0,"treatment_wall_s":5.5})
        f=tmp_path/f"scored-{sh}.json"
        f.write_text(json.dumps({"protocol":PROTOCOL,
                 "status":"PUBLISHER_STRICT_PHYSICAL_SET_BAG_EX_SCORED_POST_SEAL",
                 "model_digest":DIGEST,"source_sha256":"synthetic-new48",
                 "shard":sh,"db_id":db,"n":6,"cases":cases}))
        paths.append(f)
    final=tmp_path/"final.json"
    d=summarize(paths,manifest,final)
    assert d["n"]==48 and d["scorable"]==48
    assert d["counts"]["baseline_selected"]==1
    assert d["counts"]["treatment_oracle"]==1
    assert d["mean_structures_treatment"]==3
    assert d["mean_structures_control"]==2
    assert d["selected_net"]==d["oracle_net"]==0
    assert not d["positive_screen_only"] and not d["promotion_allowed"]
    assert d["control_model_calls"]==d["treatment_model_calls"]==192
    assert d["actual_unique_original_model_calls"]==288
    assert json.loads(final.read_text())==d
