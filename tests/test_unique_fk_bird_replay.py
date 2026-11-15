from pathlib import Path
import sqlite3

import pytest

from evaluation.unique_fk_bird_replay import (
    MODEL_DIGEST, PROTOCOL, SOURCE_CRLF_SHA256, aggregate, score_case
)


def db_fixture(tmp_path:Path):
    db=tmp_path/"test.sqlite"
    with sqlite3.connect(db) as c:
        c.executescript(
            "CREATE TABLE account(id INTEGER PRIMARY KEY, label TEXT);"
            "CREATE TABLE transfer(tx INTEGER PRIMARY KEY, acct INTEGER, amount INTEGER,"
            "FOREIGN KEY(acct) REFERENCES account(id));"
            "INSERT INTO account VALUES(1,'Alice'),(2,'Bob');"
            "INSERT INTO transfer VALUES(11,1,25),(12,1,50),(13,2,75);"
        )
    return db


def test_source_only_wrong_join_fix_rescues_actual_rows(tmp_path):
    db=db_fixture(tmp_path)
    original={
        "case_id":"test-cid","database_id":"source-db",
        "guarded":{"status":"ok","final_sql":
            "SELECT a.label FROM account AS a JOIN transfer AS t "
            "ON a.id = t.tx WHERE t.tx = 11"}
    }
    gold=("SELECT a.label FROM account AS a JOIN transfer AS t "
          "ON a.id = t.acct WHERE t.tx = 11")
    scored=score_case(original,db,gold)
    assert scored["triggered"] and scored["verified"]
    assert not scored["original_set"] and scored["treatment_set"]
    assert not scored["original_bag"] and scored["treatment_bag"]


def test_existing_valid_fk_sql_is_invariant(tmp_path):
    db=db_fixture(tmp_path)
    sql=("SELECT a.label FROM account a JOIN transfer t "
         "ON a.id = t.acct WHERE t.tx = 11")
    row={"case_id":"case","database_id":"db",
         "guarded":{"status":"ok","final_sql":sql}}
    x=score_case(row,db,sql)
    assert not x["triggered"]
    assert x["original_set"] and x["treatment_set"]
    assert x["original_bag"] and x["treatment_bag"]


def test_refused_sql_is_never_executed_as_an_unverified_repair(tmp_path):
    db=db_fixture(tmp_path)
    row={"case_id":"case","database_id":"db",
         "guarded":{"status":"verification_failed","final_sql":"DROP TABLE account"}}
    x=score_case(row,db,"SELECT label FROM account")
    assert not x["triggered"]
    assert not x["treatment_set"]


def test_all_four_source_shards_required_and_no_fabricated_gain():
    batches=[]
    for shard in range(4):
        items=[]
        for idx in range(25):
            overall=shard*25+idx
            correct=overall<39
            items.append({
                "case_id":str(overall),
                "database_id":"db_"+str(overall%6),
                "triggered":False,"verified":False,
                "scorable":True,
                "original_set":correct,"treatment_set":correct,
                "original_bag":correct,"treatment_bag":correct,
                "original_label_agreed":True,
            })
        batches.append({
            "protocol":PROTOCOL,"source_sha256":SOURCE_CRLF_SHA256,
            "model_digest":MODEL_DIGEST,"original_status":
                "invalid" if shard==3 else "complete",
            "shard":shard,"cases":items,"new_model_calls":0,
        })
    result=aggregate(batches)
    assert result["stats"]["original_set_ex"]==39
    assert result["stats"]["treatment_set_ex"]==39
    assert result["original_invalid_shards"]==[3]
    assert not result["positive_screen_only"]
    assert not result["promotion_allowed"]
    with pytest.raises(ValueError,match="incomplete"):
        aggregate(batches[:3])
