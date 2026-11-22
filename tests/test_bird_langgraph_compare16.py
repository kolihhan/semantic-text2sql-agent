"""Physical-source identity and cost-accounting checks. Synthetic fixtures only."""
import json
from pathlib import Path

import pytest

from experiments import bird_langgraph_compare16 as comp
from experiments.bird_langgraph_adaptive16 import PROTOCOL as GRAPH_PROTOCOL
from experiments.bird_qwen9b_physical_answer_contract16 import (
    PROTOCOL as K4_PROTOCOL, DIGEST
)


def payload(path: Path, item: dict) -> Path:
    path.write_text(json.dumps(item))
    return path


def fixtures(tmp_path: Path, monkeypatch):
    dbs=["d0","d1","d2","d3"]
    rows=[{"source_index":i,"db_id":dbs[i//4],"question_sha256":str(i)}
          for i in range(16)]
    meta={"source_sha256":"sha-frozen","dbs":dbs}
    monkeypatch.setattr(comp,"checked_source", lambda _s,_m: (rows,meta))
    parts={k:[] for k in ("adaptive_generated","adaptive_scored","k4_generated","k4_scored")}
    for shard in range(4):
        cases={k:[] for k in parts}
        for i in range(shard*4,(shard+1)*4):
            old=(i%2==0)
            new=old or i in (1,5)
            g={"source_index":i,"db_id":dbs[shard],
               "question_sha256":str(i),"original_sql":f"SELECT {i}",
               "control_sql":f"SELECT {i}","treatment_sql":f"SELECT {i+1}",
               "control_calls":0,"treatment_calls":int(i in (1,5,7)),
               "control_wall_s":0.0,"treatment_wall_s":10.0 if i in (1,5,7) else 0.0}
            cases["adaptive_generated"].append(g)
            cases["adaptive_scored"].append({
                "source_index":i,"db_id":dbs[shard],"scorable":True,
                "control_set":old,"treatment_set":new,
                "control_bag":old,"treatment_bag":new,
                "control_calls":g["control_calls"],"treatment_calls":g["treatment_calls"],
                "control_wall_s":g["control_wall_s"],"treatment_wall_s":g["treatment_wall_s"],
                "adaptive_probed":bool(g["treatment_calls"]),
                "adaptive_fallback":False,
            })
            cases["k4_generated"].append({
                "source_index":i,"db_id":dbs[shard],
                "question_sha256":str(i),"control_sql":[f"SELECT {i}","a","b","c"],
            })
            cases["k4_scored"].append({
                "source_index":i,"db_id":dbs[shard],"reference_scorable":True,
                "control_selected_set_EX":old,"treatment_selected_set_EX":new,
                "control_selected_bag_EX":old,"treatment_selected_bag_EX":new,
                "control_wall_s":15.0,"treatment_wall_s":30.0
            })
        for family,case_rows in cases.items():
            code=GRAPH_PROTOCOL if family.startswith("adaptive") else K4_PROTOCOL
            status={
                "adaptive_generated":"BIRD_LANGGRAPH_CANDIDATE_SEALED_PRE_GOLD",
                "adaptive_scored":"POST_SEAL_BIRD_STRICT_PHYSICAL_SCORE",
                "k4_generated":"ORIGINAL_QWEN9B_CANDIDATES_AND_CHOICES_SEALED_PRE_GOLD",
                "k4_scored":"PUBLISHER_STRICT_PHYSICAL_SET_BAG_EX_SCORED_POST_SEAL",
            }[family]
            file=tmp_path/f"{family}-{shard}.json"
            parts[family].append(payload(file,{
                "protocol":code,"status":status,"model_digest":DIGEST,
                "source_sha256":"sha-frozen","shard":shard,"db_id":dbs[shard],
                "n":4,"cases":case_rows
            }))
    return parts


def compare(tmp_path,parts):
    return comp.summarize(
        source=tmp_path/"unused-source",manifest=tmp_path/"unused-meta",
        adaptive_generated=parts["adaptive_generated"],
        adaptive_scored=parts["adaptive_scored"],
        k4_generated=parts["k4_generated"],
        k4_scored=parts["k4_scored"],
        out=tmp_path/"out.json",
    )


def test_reconciles_actual_paired_strict_ex_and_four_arm_cost(tmp_path,monkeypatch):
    f=fixtures(tmp_path,monkeypatch)
    x=compare(tmp_path,f)
    assert x["sample"]["n"]==16
    assert x["correct"]["guarded_set"]==8
    assert x["correct"]["adaptive_set"]==10
    assert x["correct"]["k4_set"]==8
    assert x["correct"]["physical_k4_set"]==10
    assert x["differences"]["adaptive_rescues"]==2
    assert x["differences"]["adaptive_regressions"]==0
    assert x["model_calls"]["guarded_total_logical_calls"]==16
    assert x["model_calls"]["adaptive_total_logical_calls"]==19
    assert x["model_calls"]["k4_total_logical_calls"]==64
    assert x["model_calls"]["physical_k4_total_logical_calls"]==64
    assert x["runtime_seconds"]["adaptive_extra_model_wall_s"]==30.0
    assert x["runtime_seconds"]["k4_total_original_generation_wall_s"]==240.0
    assert x["runtime_seconds"]["physical_k4_over_k4_model_wall_ratio"]==2.0
    assert x["runtime_seconds"]["initial_first_candidate_generation_wall_s"] is None
    assert x["runtime_seconds"]["end_to_end_latency_s"] is None
    assert x["model_calls"]["model_call_cost_currency"] is None
    assert x["production_promotion_permitted"] is False


def test_does_not_compare_different_sql_source(tmp_path,monkeypatch):
    f=fixtures(tmp_path,monkeypatch)
    p=f["adaptive_generated"][2]
    d=json.loads(p.read_text())
    d["cases"][0]["original_sql"]="SELECT other"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match="same original SQL"):
        compare(tmp_path,f)


def test_requires_complete_uncensored_source(tmp_path,monkeypatch):
    f=fixtures(tmp_path,monkeypatch)
    f["adaptive_scored"].pop()
    with pytest.raises(ValueError,match="four DB shards"):
        compare(tmp_path,f)


def test_no_benchmark_label_is_needed_to_compute_comparison(tmp_path,monkeypatch):
    f=fixtures(tmp_path,monkeypatch)
    x=compare(tmp_path,f)
    assert x["no_gold_fed_to_any_model"] is True
    assert x["new_DEV500_or_reserved_HOLDOUT300_used"] is False
    assert x["historic_BIRD_DEV100_33_to_39_is_NOT_same_cohort"] is True
