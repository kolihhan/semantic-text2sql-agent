import sqlite3
from pathlib import Path
import pytest

from evaluation.qwen_grain_direct_spider import (
    PROTOCOL, DIGEST, SHARDS, CASES, DBS, _GRAIN_HINT,
    GrainDirectProvider, aggregate, frozen_source, is_group_question
)


def test_intent_requires_aggregation_and_entity_grain_not_keywords_alone():
    assert is_group_question("Find the average number of songs for each artist")
    assert is_group_question("How many events are there per employee?")
    assert is_group_question("What is the total number of orders by client?")
    assert not is_group_question("What is the average duration?")
    assert not is_group_question("List each student's name")
    assert not is_group_question("Which student was first?")


def test_freeze_is_fully_gold_blind_and_physically_database_diverse(tmp_path):
    rows=[]
    for i in range(10):
        dbid=f"grain_fixture_{i}"
        root=tmp_path/dbid
        root.mkdir()
        with sqlite3.connect(root/f"{dbid}.sqlite") as c:
            c.executescript(
                "CREATE TABLE parent(id INTEGER PRIMARY KEY);"
                "CREATE TABLE child(id INTEGER PRIMARY KEY, parent_id INTEGER);"
            )
        for j in range(2):
            rows.append({
                "db_id":dbid,
                "question":f"What is the average number of children per parent for {i} test {j}?",
                "query":"SELECT 1000" if j else "DROP TABLE parent"
            })
    first=frozen_source(rows,tmp_path)
    assert first["n"]==CASES and first["physical_dbs"]==DBS
    assert len(first["cases"])==CASES
    assert len({r["db_id"] for r in first["cases"]})==DBS
    assert all("query" not in case for case in first["cases"])
    # References are never used in selection, even when every reference changes.
    mutated=[{**r, "query":"SELECT 99999"} for r in rows]
    assert frozen_source(mutated,tmp_path)==first
    assert first["possible_pretraining_overlap"]
    assert not first["source_gold_read"]


def test_small_physical_coverage_refused_without_selection_relaxation(tmp_path):
    dbid="tiny_grain"
    folder=tmp_path/dbid
    folder.mkdir()
    with sqlite3.connect(folder/f"{dbid}.sqlite") as c:
        c.executescript("CREATE TABLE a(x INTEGER); CREATE TABLE b(y INTEGER);")
    rows=[{"db_id":dbid,"question":"How many events per employee?",
           "query":"SELECT 1"}]
    with pytest.raises(ValueError,match="cohort failed"):
        frozen_source(rows,tmp_path)


class CounterProvider:
    def __init__(self):
        self.args=[]
    def complete_text(self,*,system,user):
        self.args.append((system,user))
        return "SELECT 1"


def test_only_first_generation_system_instruction_changes():
    inner=CounterProvider()
    treatment=GrainDirectProvider(inner)
    treatment.complete_text(system="Original SQL-only system",user="same question")
    inner.complete_text(system="Original SQL-only system",user="same question")
    assert len(inner.args)==2
    assert inner.args[0][0].startswith(inner.args[1][0])
    assert inner.args[0][0].endswith(_GRAIN_HINT)
    assert inner.args[0][1]==inner.args[1][1]


def test_paired_aggregate_refuses_missing_shard_and_never_promotes():
    cases=[]
    for index in range(CASES):
        original= index<8
        treated= original or index==8
        dbid=f"fixture_db_{index//2}"
        cases.append({
            "id":str(index),"db_id":dbid,"scorable":True,
            "db_sha256":str(index//2),
            "original":{"set_ex":original,"bag_ex":original,"model_calls":1,"elapsed_s":1.0},
            "grain_direct":{"set_ex":treated,"bag_ex":treated,"model_calls":1,"elapsed_s":1.1}
        })
    shards=[
        {"protocol":PROTOCOL,"digest":DIGEST,
         "cohort_sha256":"pre-frozen","shard":i,
         "cases":cases[i*6:(i+1)*6]}
        for i in range(SHARDS)
    ]
    summary=aggregate(shards)
    assert summary["arms"]["original"]["set_ex"]==8
    assert summary["arms"]["grain_direct"]["set_ex"]==9
    assert summary["rescues"]==["8"]
    assert not summary["screen_pass_only"] # one rescue in one DB is insufficient
    assert not summary["promote"]
    with pytest.raises(ValueError,match="incomplete"):
        aggregate(shards[:-1])
