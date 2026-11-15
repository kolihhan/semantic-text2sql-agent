"""Gold-blind, cross-database BIRD train in-context demonstrations."""
import pytest

from evaluation.qwen_icl_bird20 import (
    build_retrieval, select_cases, generate_examples,
    PromptWithExamples, aggregate, PROTOCOL,
)


def test_retrieval_uses_question_only_with_two_different_train_databases():
    train = [
        {"db_id":"one","question":"How many orders were placed in June?","SQL":"SELECT COUNT(*) FROM orders"},
        {"db_id":"two","question":"Count how many orders arrived in June","SQL":"SELECT COUNT(*) FROM shipments"},
        {"db_id":"three","question":"List top sellers by order revenue","SQL":"SELECT name FROM sellers"},
    ]
    index=build_retrieval(train)
    chosen=generate_examples(index,"How many orders were placed in June?",current_db="dev",k=2)
    assert len(chosen)==2
    assert len({c["db_id"] for c in chosen})==2
    assert all("SQL" in c and c["SQL"] for c in chosen)
    assert chosen==generate_examples(index,"How many orders were placed in June?","dev",k=2)


def test_holdout_selection_uses_id_db_question_not_reference_sql():
    dev=[{"question_id":i+1,"db_id":f"db_{i//4}","question":f"Find thing {i}","SQL":"DO NOT USE"} for i in range(48)]
    chosen=select_cases(dev,excluded={"1","2","3"})
    assert len(chosen)==20 and len({r["db_id"] for r in chosen})==10
    assert all("SQL" not in r for r in chosen)
    assert chosen==select_cases(dev,excluded={"1","2","3"})


def test_fewshot_only_affects_first_generation_not_repair():
    class Inner:
        def __init__(self):self.calls=[]
        def complete_text(self,*,system,user):
            self.calls.append((system,user))
            return "SELECT 1"
    x=Inner()
    p=PromptWithExamples(x,[
        {"db_id":"train","question":"How many?","SQL":"SELECT COUNT(*) FROM records"},
        {"db_id":"train2","question":"Which IDs?","SQL":"SELECT id FROM t"},
    ])
    p.complete_text(system="system",user="Question: current")
    p.complete_text(system="repair",user="Rejected SQL")
    assert "SELECT COUNT(*) FROM records" in x.calls[0][1]
    assert "current" in x.calls[0][1]
    assert "SELECT COUNT(*) FROM records" not in x.calls[1][1]


def test_aggregate_refuses_missing_shard_or_score_drift():
    shards=[]
    for shard in range(5):
        rows=[]
        for j in range(4):
            i=shard*4+j
            base=i<10
            icl=base
            if i in (10,11,12):icl=True
            if i==0:icl=False
            rows.append({
                "case_id":str(i),"db_id":f"db_{i//2}","reference_scorable":True,
                "baseline":{"ex":base,"bag":base,"execution_success":True,"latency_s":1},
                "icl":{"ex":icl,"bag":icl,"execution_success":True,"latency_s":2},
                "examples_count":2,
            })
        shards.append({"shard":shard,"protocol":PROTOCOL,"cohort_sha256":"x",
                       "model_digest":"test","cases":rows})
    report=aggregate(shards,expected_digest="test")
    assert report["baseline_ex"]==10 and report["icl_ex"]==12
    assert report["rescues"]==3 and report["regressions"]==1
    assert report["promotion_allowed"] is False
    with pytest.raises(ValueError,match="shards"):
        aggregate(shards[:4],expected_digest="test")
