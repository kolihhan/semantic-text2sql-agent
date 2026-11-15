from pathlib import Path
import sqlite3

from evaluation.qwen_observed_sql_tools import (
    ObservedSqlTools,run_tool_agent,_extract_call,MAX_REPLY_CHARS
)


def source(tmp_path:Path)->Path:
    p=tmp_path/"test.sqlite"
    with sqlite3.connect(p) as c:
        c.executescript(
            "CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT);"
            "CREATE TABLE orders(order_id INTEGER PRIMARY KEY, user_id INTEGER, total REAL,"
            "FOREIGN KEY(user_id) REFERENCES users(id));"
            "INSERT INTO users VALUES(1,'Alice'),(2,'Bob');"
            "INSERT INTO orders VALUES(11,1,12.5),(12,1,10.0),(13,2,5.0);"
        )
    return p


def tool(name,kwargs):
    return {"function":{"name":name,"arguments":kwargs}}


def test_original_source_schema_and_rows_are_observed_not_gold(tmp_path):
    o=ObservedSqlTools(source(tmp_path))
    schema=o.invoke("inspect_schema",{"table":"orders"})
    assert "user_id" in schema and "foreign_keys" in schema
    rows=o.invoke("sample_rows",{"table":"orders"})
    assert "12.5" in rows and "Alice" not in rows
    assert len(rows)<=MAX_REPLY_CHARS
    joined=o.invoke("run_sql",{"sql":
        "SELECT u.name, SUM(o.total) FROM users u JOIN orders o "
        "ON u.id=o.user_id GROUP BY u.id"})
    assert "Alice" in joined and "22.5" in joined
    assert o.used["run_sql"]==1


def test_unsafe_sql_rejected_and_db_is_unchanged(tmp_path):
    path=source(tmp_path)
    o=ObservedSqlTools(path)
    message=o.invoke("run_sql",{"sql":"DELETE FROM orders"})
    assert "Rejected" in message
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]==3


def test_model_original_function_call_observation_then_final(tmp_path):
    prompts=[]
    replies=iter([
        {"message":{"content":"","tool_calls":[tool("inspect_schema",{"table":"orders"})]}},
        {"message":{"content":"","tool_calls":[tool("run_sql",{
            "sql":"SELECT COUNT(*) FROM orders"
        })]}},
        {"message":{"content":"","tool_calls":[tool("submit_sql",{
            "sql":"SELECT COUNT(*) FROM orders"
        })]}},
    ])
    def pretend(payload):
        prompts.append(payload)
        return next(replies)
    output=run_tool_agent(
        "How many orders?", "orders(order_id,user_id,total)",source(tmp_path),
        "A customer has an order.",request_fn=pretend
    )
    assert output.sql=="SELECT COUNT(*) FROM orders"
    assert output.outcome=="model_submitted"
    assert output.calls==2 and output.trial_calls==1
    assert output.inspect_calls==1 and output.tool_errors==0
    assert output.model_calls==3
    assert prompts[0]["model"]=="qwen3.5:9b-q4_K_M"
    assert prompts[0]["options"]["temperature"]==0
    assert "A customer has an order." in prompts[0]["messages"][1]["content"]
    assert "order_id" in str(prompts[1]["messages"][-1])
    assert "3" in str(prompts[2]["messages"][-1])


def test_early_final_no_tools_is_not_fake_interaction(tmp_path):
    output=run_tool_agent("How many orders?","orders(order_id)",
        source(tmp_path),None,request_fn=lambda _:{
        "message":{"content":"SELECT COUNT(*) FROM orders"}
    })
    assert output.calls==0 and output.model_calls==1
    assert output.outcome=="plain_sql_final"


def test_unrelated_or_malformed_tools_fail_closed(tmp_path):
    output=run_tool_agent("How many orders?","orders(order_id)",
        source(tmp_path),None,request_fn=lambda _:{
        "message":{"tool_calls":[tool("DROP_TABLE",{"name":"orders"})],"content":""}
    })
    assert output.sql is None and output.outcome=="unknown_model_tool"
    assert output.tool_errors==1


def test_model_does_not_receive_reference_answer_or_sql(tmp_path):
    observed=[]
    def pretend(payload):
        observed.append(str(payload))
        return {"message":{"content":"SELECT 1"}}
    run_tool_agent("A question without a reference",
        "users(id,name)",source(tmp_path),"user evidence",request_fn=pretend)
    assert "gold_sql" not in observed[0]
    assert "reference_answer" not in observed[0]


def test_sampled_physical_rows_are_bounded(tmp_path):
    path=source(tmp_path)
    with sqlite3.connect(path) as con:
        con.execute("UPDATE users SET name=? WHERE id=1",("x"*5000,))
    o=ObservedSqlTools(path)
    reply=o.invoke("sample_rows",{"table":"users"})
    assert len(reply)<=MAX_REPLY_CHARS
