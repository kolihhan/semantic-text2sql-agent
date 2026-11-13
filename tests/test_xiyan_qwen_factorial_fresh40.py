"""Contract checks for disjoint 2x2 XiYan/Qwen evaluation. No model inference."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import evaluation.xiyan_qwen_factorial_fresh40 as f


def fixtures(tmp_path: Path, monkeypatch):
    prev = []
    for index, size in enumerate(f.COHORT_COUNTS):
        begin = sum(f.COHORT_COUNTS[:index])
        cases = [{"case_id": str(begin + j)} for j in range(size)]
        file = tmp_path / f"historical-{index}.json"
        file.write_text(json.dumps({"selected": cases} if index >= 6 else {"cases": cases}))
        prev.append(file)
    source = tmp_path / "dev.json"
    source.write_text(json.dumps([
        {
            "question_id": str(540 + i),
            "db_id": f"db_{(i // 2) % 12}",
            "question": "How many events happened?" if i % 2 == 0 else
                        f"Which events happened in city {i}?",
            "SQL": "SELECT 1",
            "evidence": "",
        }
        for i in range(200)
    ]))
    from hashlib import sha256
    monkeypatch.setattr(f, "SOURCE_SHA256", sha256(source.read_bytes()).hexdigest())
    return prev, source


def test_frozen_540_exclusion_and_40_new_database_diverse(tmp_path, monkeypatch):
    paths, source = fixtures(tmp_path, monkeypatch)
    seen = f.source_cohort_ids(paths)
    assert len(seen) == 540 and "0" in seen and "539" in seen
    m = f.manifest(source=source, excluded=paths)
    assert m["excluded_identity_count"] == 540
    assert m["strata"] == {"count_intent": 20, "control": 20}
    assert all(n >= 7 for n in m["distinct_databases_by_stratum"].values())
    assert len(m["selected"]) == 40
    assert {x["case_id"] for x in m["selected"]}.isdisjoint(seen)
    assert m["gold_used_for_selection"] is False
    assert m["gold_used_for_generation"] is False
    assert m == f.manifest(source=source, excluded=paths)
    f._verify_manifest(m, source)


def test_excluded_duplicate_and_wrong_model_or_source_fail_closed(tmp_path, monkeypatch):
    paths, source = fixtures(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="eight"):
        f.source_cohort_ids(paths[:-1])
    c = json.loads(paths[-1].read_text())
    c["selected"][0]["case_id"] = "100"
    paths[-1].write_text(json.dumps(c))
    with pytest.raises(ValueError, match="overlap"):
        f.manifest(source=source, excluded=paths)


def test_exact_prompt_factor_and_same_question_schema_evidence():
    question = "How many schools in district A?"
    schema = "Physical SQLite schema:\nschools(id,name,district)"
    evidence = "district code is A"
    sys, user = f.prompt_for(condition="generic",question=question,
                             schema_context=schema,evidence=evidence)
    from semantic_sql.direct import _SYSTEM
    assert sys == _SYSTEM
    assert user == f"Question: {question}\nSchema context:\n{schema}\nExternal evidence:\n{evidence}"
    sys2, user2 = f.prompt_for(condition="xiyan",question=question,
                               schema_context=schema,evidence=evidence)
    assert sys2 == ""
    assert user2.count(question) == 2
    assert schema in user2 and evidence in user2
    assert user2.endswith(chr(96)*3+"sql")
    with pytest.raises(ValueError, match="unknown"):
        f.prompt_for(condition="new",question=question,
                     schema_context=schema,evidence=evidence)


def test_paired_aggregate_gives_four_arms_paired_effects():
    cases=[]
    for i in range(40):
        arms={}
        for name in f.ARM_KEYS:
            ex = (i < 12) if name == "qwen:generic" else (
                (i < 14) if name == "qwen:xiyan" else
                (i < 18) if name == "xiyan:generic" else (i < 20)
            )
            arms[name]={"ex":ex,"preflight":True,"execution_success":True,"sql":"SELECT 1"}
        cases.append({
            "case_id":str(540+i),
            "group":"count_intent" if i < 20 else "control",
            "database_id":"db","arms":arms,
        })
    all_inputs=[{
        "schema":"bird-2x2-xiyan-qwen-shard/v1",
        "shard":index, "manifest_sha256":"frozen",
        "model_seed":f.MODEL_SEED, "model_ids":f.MODEL_IDS,
        "model_digests":{"qwen":f.QWEN_DIGEST,"xiyan":"a"*64},
        "gold_used_for_generation":False,
        "cases":cases[index::8],
    } for index in range(8)]
    r=f.aggregate(all_inputs)
    assert r["n"]==40
    assert [r["overall"]["arms"][key]["ex_correct"] for key in f.ARM_KEYS]==[12,14,18,20]
    assert r["overall"]["model_effect_under_generic"]["net"]==6
    assert r["overall"]["model_effect_under_xiyan_prompt"]["net"]==6
    assert r["overall"]["prompt_effect_qwen"]["net"]==2
    assert r["overall"]["prompt_effect_xiyan"]["net"]==2
    assert r["gold_used_for_generation"] is False
    assert r["no_training"] is True
    all_inputs[0]["model_digests"]["xiyan"]="b"*64
    with pytest.raises(ValueError,match="digest drift"):
        f.aggregate(all_inputs)


def test_real_read_only_ex_scorer(tmp_path):
    db = tmp_path/"sample.sqlite"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE t (id INTEGER)")
        c.executemany("INSERT INTO t VALUES (?)",[(1,),(2,)])
    correct = f._ex(db,"SELECT COUNT(*) FROM t",{(2,)})
    wrong = f._ex(db,"SELECT COUNT(*) FROM t",{(3,)})
    unsafe = f._ex(db,"DELETE FROM t",{(2,)})
    assert correct["ex"] and correct["execution_success"]
    assert not wrong["ex"] and wrong["execution_success"]
    assert not unsafe["ex"] and not unsafe["execution_success"]
    with sqlite3.connect(db) as c:
        assert c.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 2
