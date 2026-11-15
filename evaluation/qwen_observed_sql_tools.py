"""Bounded Qwen-only interactive SQLite evidence generation, experimental only.

The model can inspect physical table types, sample a few original rows,
and test (read-only) draft SQL before committing SQL. Original Qwen model
is unchanged; no gold/reference SQL or EX label enters the runtime.
This module is NOT connected to production or default Guarded.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
from urllib.request import Request,urlopen

from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.direct import _strip_fence
from semantic_sql.execution import execute_readonly,SQLExecutionError
from semantic_sql.verifier import verify_sql_preflight

MODEL="qwen3.5:9b-q4_K_M"
DIGEST="56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
MAX_STEPS=6
MAX_TOOLS=4
MAX_TRIAL_QUERIES=2
MAX_REPLY_CHARS=1700

TOOL_DEFS=[
    {"type":"function","function":{
        "name":"inspect_schema","description":"Inspect physical SQL table columns and declared foreign keys from the actual SQLite database.",
        "parameters":{"type":"object","properties":{
            "table":{"type":"string","description":"Exact existing table name"}
        },"required":["table"]}}},
    {"type":"function","function":{
        "name":"sample_rows","description":"See up to three original real rows and column names from one SQL table; read-only.",
        "parameters":{"type":"object","properties":{
            "table":{"type":"string","description":"Exact existing table name"}
        },"required":["table"]}}},
    {"type":"function","function":{
        "name":"run_sql","description":"Execute a provisional read-only SELECT and inspect up to five real result rows; never performs writes.",
        "parameters":{"type":"object","properties":{
            "sql":{"type":"string","description":"A single SQLite SELECT/CTE query to test"}
        },"required":["sql"]}}},
    {"type":"function","function":{
        "name":"submit_sql","description":"Submit the final SQL answer to the user's question; no further tool calls.",
        "parameters":{"type":"object","properties":{
            "sql":{"type":"string","description":"One final read-only SQLite SELECT/CTE"}
        },"required":["sql"]}}},
]

SYSTEM=(
    "You are a Text-to-SQL system. Return one correct SQLite SELECT/CTE that answers "
    "the user question exactly, including the requested entity, filters, unit, "
    "aggregation grain, and return columns. When genuinely uncertain, inspect "
    "actual schema/rows or trial a read-only SQL. Those tools show observed "
    "database facts, not ground truth answers. Do not blindly trust zero rows "
    "or identical-looking values as semantic correctness. Use at most 4 "
    "tools. Call submit_sql with your final SQL; do not write explanations."
)


def _serialize(data:Any)->str:
    return json.dumps(data,ensure_ascii=False,default=str,separators=(",",":"))


@dataclass(frozen=True)
class ToolAgentRun:
    sql: str|None
    outcome: str
    calls: int
    inspect_calls: int
    sample_calls: int
    trial_calls: int
    tool_errors: int
    model_calls: int


class ObservedSqlTools:
    def __init__(self,db:Path):
        self.database=Path(db)
        self.catalog=DatabaseCatalog.from_sqlite(db)
        self.used=Counter()

    def _table(self,table:str)->str|None:
        return next((t for t in self.catalog.tables if t.casefold()==str(table).casefold()),None)

    def invoke(self,name:str,args:dict[str,Any])->str:
        if not isinstance(args,dict):
            raise ValueError("invalid tool arguments")
        self.used[name]+=1
        if name=="inspect_schema":
            table=self._table(args.get("table",""))
            if not table:
                return "Unknown table; inspect only existing table names."
            con=__import__("sqlite3").connect(f"{self.database.resolve().as_uri()}?mode=ro",uri=True)
            try:
                con.execute("PRAGMA query_only=ON")
                quoted='"'+table.replace('"','""')+'"'
                columns=[
                    {"name":str(row[1]),"type":str(row[2]),"pk":bool(row[5])}
                    for row in con.execute(f"PRAGMA table_info({quoted})")
                ]
            finally:
                con.close()
            fk=[(source,sc,target,tc) for source,sc,target,tc in self.catalog.foreign_keys
                if source==table or target==table]
            return _serialize({"table":table,"columns":columns,"physical_foreign_keys":fk})[:MAX_REPLY_CHARS]
        if name=="sample_rows":
            table=self._table(args.get("table",""))
            if not table:
                return "Unknown table; sample existing table only."
            quoted='"'+table.replace('"','""')+'"'
            data=execute_readonly(self.database,f"SELECT * FROM {quoted} LIMIT 3",max_rows=3)
            return _serialize({"table":table,"columns":data.columns,"rows":data.rows})[:MAX_REPLY_CHARS]
        if name=="run_sql":
            sql=str(args.get("sql") or "").strip()
            if not sql or len(sql)>4000:
                return "Rejected SQL: empty or exceeds 4000 chars"
            allowed=verify_sql_preflight(self.database,sql)
            if not allowed.ok:
                return "Rejected SQL: "+",".join(x.code for x in allowed.issues)
            try:
                data=execute_readonly(self.database,sql,max_rows=5)
                return _serialize({"columns":data.columns,"rows":data.rows,
                                   "truncated":data.truncated})[:MAX_REPLY_CHARS]
            except (SQLExecutionError,ValueError) as exc:
                return "Read-only execution error: "+type(exc).__name__
        raise ValueError("unsupported tool function")


def _extract_call(record:dict)->tuple[str,dict]|None:
    if not isinstance(record,dict):
        raise ValueError("bad tool call")
    function=record.get("function")
    if not isinstance(function,dict):
        raise ValueError("tool call lacks function")
    name=function.get("name")
    raw=function.get("arguments")
    if isinstance(raw,str):
        try:
            raw=json.loads(raw)
        except (ValueError,TypeError):
            raise ValueError("invalid JSON arguments")
    if not isinstance(name,str) or not isinstance(raw,dict):
        raise ValueError("tool call arguments must be object")
    return name,raw


def run_tool_agent(question:str,schema:str,db:Path,evidence:str|None,
                   *,url:str="http://127.0.0.1:11434/api/chat",request_fn=None)->ToolAgentRun:
    """Use physical observed tools only; evaluator gold never passed to caller."""
    observer=ObservedSqlTools(db)
    user=f"Question: {question}\nSchema context:\n{schema}"
    if evidence:
        user+=f"\nExternal evidence:\n{evidence}"
    history=[
        {"role":"system","content":SYSTEM},
        {"role":"user","content":user},
    ]
    model_calls=0
    errors=0
    tools=0
    def result(sql,outcome):
        return ToolAgentRun(sql,outcome,tools,
            observer.used["inspect_schema"],observer.used["sample_rows"],
            observer.used["run_sql"],errors,model_calls)
    for step in range(MAX_STEPS):
        payload={
            "model":MODEL,"stream":False,"think":False,"tools":TOOL_DEFS,
            "messages":history,
            "options":{"temperature":0,"seed":42,"num_predict":700},
        }
        if request_fn is not None:
            reply=request_fn(payload)
        else:
            req=Request(url,data=_serialize(payload).encode("utf-8"),
                headers={"Content-Type":"application/json"},method="POST")
            with urlopen(req,timeout=800) as response:
                reply=json.load(response)
        msg=reply.get("message") if isinstance(reply,dict) else None
        if not isinstance(msg,dict):
            return result(None,"malformed_model_reply")
        model_calls+=1
        calls=msg.get("tool_calls") or []
        if not calls:
            sql=_strip_fence(str(msg.get("content") or "")).strip()
            return result(sql or None,"plain_sql_final" if sql else "empty_model_final")
        if len(calls)!=1:
            errors+=1
            return result(None,"multiple_tool_calls_unsupported")
        try:
            name,args=_extract_call(calls[0])
        except ValueError:
            errors+=1
            return result(None,"invalid_model_tool_args")
        if name=="submit_sql":
            sql=str(args.get("sql") or "").strip()
            return result(sql or None,"model_submitted")
        if tools>=MAX_TOOLS:
            return result(None,"tool_budget_exhausted")
        if name not in {"inspect_schema","sample_rows","run_sql"}:
            errors+=1
            return result(None,"unknown_model_tool")
        if name=="run_sql" and observer.used["run_sql"]>=MAX_TRIAL_QUERIES:
            errors+=1
            return result(None,"trial_sql_budget_exhausted")
        try:
            feedback=observer.invoke(name,args)
        except (ValueError,SQLExecutionError) as exc:
            errors+=1
            feedback="Tool refused: "+type(exc).__name__
        tools+=1
        history.append({"role":"assistant","tool_calls":calls})
        history.append({"role":"tool","name":name,"content":feedback})
    return result(None,"model_step_budget_exhausted")
