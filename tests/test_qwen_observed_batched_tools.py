"""General native Qwen tool-protocol regression tests; no BIRD question IDs."""
from pathlib import Path
import sqlite3

from evaluation.qwen_observed_sql_tools import run_tool_agent,MAX_TOOLS,MAX_TRIAL_QUERIES


def db(tmp_path:Path):
    p=tmp_path/"fixture.sqlite"
    with sqlite3.connect(p) as c:
        c.executescript(
            "CREATE TABLE person(id INTEGER PRIMARY KEY, fullname TEXT);"
            "CREATE TABLE bill(id INTEGER PRIMARY KEY,person_id INT,total INT,"
            "FOREIGN KEY(person_id) REFERENCES person(id));"
            "INSERT INTO person VALUES(1,'A'),(2,'B');"
            "INSERT INTO bill VALUES(11,1,4),(12,2,6);"
        )
    return p


def call(name,args,id=None):
    r={"function":{"name":name,"arguments":args}}
    if id is not None:
        r["id"]=id
    return r


def test_same_model_native_two_call_batch_delivers_real_independent_observations(tmp_path):
    shots=[]
    responses=iter([
        {"message":{"content":"","tool_calls":[
            call("inspect_schema",{"table":"bill"},"1"),
            call("sample_rows",{"table":"person"},"2")
        ]}},
        {"message":{"content":"","tool_calls":[call("submit_sql",{
            "sql":"SELECT COUNT(*) FROM bill"
        })]}},
    ])
    def request(payload):
        shots.append(payload)
        return next(responses)
    result=run_tool_agent("How many bills?","bill(id,person_id,total)",db(tmp_path),
                          None,request_fn=request)
    assert result.sql=="SELECT COUNT(*) FROM bill"
    assert result.outcome=="model_submitted"
    assert result.calls==2
    assert result.inspect_calls==1 and result.sample_calls==1
    assert result.model_calls==2
    assert result.tool_errors==0
    assert shots[1]["messages"][-2]["role"]=="tool"
    assert "foreign_keys" in str(shots[1]["messages"][-2])
    assert shots[1]["messages"][-1]["role"]=="tool"
    assert "A" in str(shots[1]["messages"][-1])
    assert shots[1]["messages"][-1]["tool_call_id"]=="2"


def test_sql_trials_never_exceed_frozen_limit_even_when_batched(tmp_path):
    shots=[]
    responses=iter([
        {"message":{"tool_calls":[
            call("run_sql",{"sql":"SELECT COUNT(*) FROM bill"}),
            call("run_sql",{"sql":"SELECT SUM(total) FROM bill"}),
            call("run_sql",{"sql":"SELECT total FROM bill"}),
        ]}},
        {"message":{"tool_calls":[call("submit_sql",{
            "sql":"SELECT COUNT(*) FROM bill"
        })]}},
    ])
    def req(p):
        shots.append(p)
        return next(responses)
    got=run_tool_agent("How many bills?","bill(id,person_id,total)",
                       db(tmp_path),None,request_fn=req)
    assert got.sql=="SELECT COUNT(*) FROM bill"
    assert got.trial_calls==MAX_TRIAL_QUERIES
    assert got.calls==MAX_TRIAL_QUERIES
    assert got.tool_errors==1 # exhausted third request is transparent
    assert "submit_sql" in [t["function"]["name"] for t in shots[1]["tools"]]
    assert "run_sql" not in [t["function"]["name"] for t in shots[1]["tools"]]
    assert "SQL trial budget exhausted" in shots[1]["messages"][-1]["content"]


def test_full_tool_budget_enforces_submit_only_without_executing_more(tmp_path):
    shots=[]
    replies=iter([
        {"message":{"tool_calls":[
            call("sample_rows",{"table":"bill"}),
            call("inspect_schema",{"table":"bill"}),
            call("sample_rows",{"table":"person"}),
            call("sample_rows",{"table":"bill"}),
        ]}},
        {"message":{"tool_calls":[call("submit_sql",{"sql":"SELECT 1"})]}},
    ])
    def fn(p):
        shots.append(p)
        return next(replies)
    result=run_tool_agent("Question","bill(id)",db(tmp_path),None,request_fn=fn)
    assert result.calls==MAX_TOOLS and result.sql=="SELECT 1"
    assert [r["function"]["name"] for r in shots[1]["tools"]]==["submit_sql"]
    assert result.model_calls==2


def test_mixed_submit_and_probes_never_fabricate_later_tool_knowledge(tmp_path):
    x=run_tool_agent("Question","bill(id)",db(tmp_path),None,
                     request_fn=lambda _:{"message":{"tool_calls":[
                         call("run_sql",{"sql":"SELECT 1"}),
                         call("submit_sql",{"sql":"SELECT 1"}),
                     ]}})
    assert x.sql is None
    assert x.outcome=="mixed_submit_and_tool_calls"
    assert x.calls==0


def test_final_only_last_turn_exposes_no_new_tool_action(tmp_path):
    seen=[]
    def fn(p):
        seen.append(p)
        if len(seen)<6:
            return {"message":{"tool_calls":[call("sample_rows",{"table":"bill"})]}}
        return {"message":{"tool_calls":[call("submit_sql",{"sql":"SELECT 2"})]}}
    r=run_tool_agent("Question","bill(id)",db(tmp_path),None,request_fn=fn)
    assert r.calls==4
    assert r.sql=="SELECT 2"
    assert len(seen)==6
    assert all([x["function"]["name"] for x in seen[-1]["tools"]]==["submit_sql"])
