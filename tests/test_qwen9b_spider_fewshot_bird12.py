from collections import Counter
from evaluation.qwen9b_spider_fewshot_bird12 import (
    PROTOCOL, MODEL_DIGEST, TwoExampleFirstCall, aggregate, ops, retrieve, words,
)


def example(q,sql,db,index):
    return {"source":"Spider TRAIN","db_id":db,"question":q,"sql":sql,
            "fingerprint":f"{index:064x}"}


def test_operator_mapping_from_question_only():
    assert {"mean","group"}<=ops("What is the average salary for each office?")
    assert "count" in ops("How many employees are there?")
    assert "rank" in ops("Which state has the highest spending?")
    assert "employee" in words("list employees salaries")


def test_retrieve_distinct_external_train_dbs_only():
    train=[
        example("How many workers per office?","SELECT office, count(*) FROM workers GROUP BY office","source_a",1),
        example("How many teachers per classroom?","SELECT cls, count(*) FROM teacher GROUP BY cls","source_b",2),
        example("How many workers per office?","SELECT x FROM leaks","target",3),
        example("The names of departments","SELECT name FROM department","source_c",4),
    ]
    chosen=retrieve("How many stores per city?","target",train)
    assert len(chosen)==2
    assert len({item["db_id"] for item in chosen})==2
    assert all(item["db_id"]!="target" for item in chosen)
    assert not retrieve("Print all rows","target",train)


def test_only_first_model_call_is_affected():
    class Base:
        def __init__(self):self.prompts=[]
        def complete_text(self,*,system,user):
            self.prompts.append((system,user))
            return "SELECT 1"
    base=Base()
    wrapped=TwoExampleFirstCall(base,[example("How many workers per office?","SELECT count(*) FROM worker","source_a",1)])
    wrapped.complete_text(system="same",user="Question: Q\nSchema context: staff(x)")
    wrapped.complete_text(system="same",user="Rejected SQL...")
    assert "Two annotated SQL examples" in base.prompts[0][1]
    assert "source_a" not in base.prompts[0][1]
    assert base.prompts[1]==("same","Rejected SQL...")


def make_shards(rescues=0,regressions=0):
    cases=[]
    for i in range(12):
        original=i<4 or (i in range(4,4+regressions))
        trial=(i<4) or (i in range(4+regressions,4+regressions+rescues))
        cases.append({
            "case_id":str(i),"database_id":f"db{i}",
            "reference_scorable":True,
            "arms":{
                "original":{"set_ex":original,"bag_ex":original},
                "two_train_examples":{"set_ex":trial,"bag_ex":trial},
            },
        })
    return [
        {"protocol":PROTOCOL,"model_digest":MODEL_DIGEST,
         "cohort_sha256":"test-frozen","shard":s,
         "cases":cases[s*3:(s+1)*3]}
        for s in range(4)
    ]


def test_gate_requires_no_regressions_and_cross_database_gain():
    reject=aggregate(make_shards(rescues=1,regressions=0))
    assert reject["positive_screen_only"] is False
    assert reject["promoted"] is False
    passing=aggregate(make_shards(rescues=4,regressions=0))
    assert passing["distinct_rescue_dbs"]>=3
    assert passing["positive_screen_only"] is True
    lost=aggregate(make_shards(rescues=4,regressions=1))
    assert lost["positive_screen_only"] is False
