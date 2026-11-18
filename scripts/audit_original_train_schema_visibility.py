"""Audit whether fixed-model BIRD TRAIN lexical schema prompts omit required tables.

Cohorts: original pinned BIRD TRAIN242 (16 DBs) vs original physical
Qwen3.5-9B TRAIN-VAL64 (4 database-disjoint DBs). No model calls, training,
DEV500, or HOLDOUT300. Reference SQL is a source-only diagnostic, never a
prompt or a supervised candidate selector.
"""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
import argparse
import json
import re
from pathlib import Path

TRAIN_SHA = "b6851fb0f2b5027833aff45a0b5e01326359b245be1c812edc66986d0df1d402"
VAL_SHA = "cc112305bc9de33a03cea704f2edecfdfb45e62cc953ae498cea27fbed93b506"
MODEL_DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
MIN_SOURCE_RATE = 0.05
PROTOCOL = "ORIGINAL-TRAIN242-VAL64-SCHEMA-PROMPT-TABLE-VISIBILITY-V1"
_SCHEMA_LINE = re.compile(r"^([^()\n]+)\(([^()]*)\)\s*$")
_FROM_JOIN = re.compile(
    r'(?i)\b(?:FROM|JOIN)\s+(?:\x60([^\x60]+)\x60|"([^"]+)"|\[([^\]]+)\]|([A-Za-z_]\w*))'
)
_CTE_NAME = re.compile(r"(?i)(?:\bWITH\s+(?:RECURSIVE\s+)?|,\s*)([A-Za-z_]\w*)\s+AS\s*\(")


def visible_tables(prompt: str) -> set[str]:
    if "\nSchema context:\n" not in prompt:
        raise ValueError("missing fixed model schema prompt boundary")
    body=prompt.split("\nSchema context:\n",1)[1]
    seen={m.group(1).strip().casefold() for line in body.splitlines()
          if (m:=_SCHEMA_LINE.fullmatch(line.strip()))}
    if not seen:
        raise ValueError("missing visible SQLite tables in original prompt")
    return seen


def referenced_tables(sql: str) -> set[str]:
    # Mask quoted string literals and comments, not quoted identifiers.
    sql=re.sub(r"/\*.*?\*/"," ",sql,flags=re.S)
    sql=re.sub(r"--[^\r\n]*"," ",sql)
    sql=re.sub(r"'(?:''|[^'])*'","''",sql)
    ctes={x.casefold() for x in _CTE_NAME.findall(sql)}
    result=set()
    for m in _FROM_JOIN.finditer(sql):
        name=next((x for x in m.groups() if x is not None),None)
        if name and name.casefold() not in ctes:
            result.add(name.casefold())
    return result


def source_visibility(prompt: str, sql: str) -> dict:
    visible=visible_tables(prompt)
    refs=referenced_tables(sql)
    return {"parsed":bool(refs),"visible_count":len(visible),
            "reference_count":len(refs),
            "missing":sorted(refs-visible) if refs else []}


def original_rows(path: Path,digest: str,n: int,db_count: int) -> list[dict]:
    blob=path.read_bytes()
    if sha256(blob).hexdigest()!=digest:
        raise ValueError("fixed BIRD TRAIN original source digest drift")
    rows=[json.loads(line) for line in blob.decode().splitlines() if line.strip()]
    if len(rows)!=n or len({x["source_index"] for x in rows})!=n:
        raise ValueError("incomplete or duplicated original TRAIN source IDs")
    if len({x["source_db_id"] for x in rows})!=db_count:
        raise ValueError("original physical DB count drift")
    for r in rows:
        msg=r["messages"]
        if ([m["role"] for m in msg]!=["system","user","assistant"] or
            sha256(msg[1]["content"].encode()).hexdigest()!=r["input_sha256"]):
            raise ValueError("source question/prompt/reference alignment drift")
    return rows


def reference_census(rows: list[dict]) -> dict:
    omitted=Counter()
    missing_ids=[]
    abstained=0
    visible_total=0
    for r in rows:
        v=source_visibility(r["messages"][1]["content"],r["messages"][2]["content"])
        visible_total+=v["visible_count"]
        if not v["parsed"]:
            abstained+=1
        if v["missing"]:
            omitted[r["source_db_id"]]+=1
            missing_ids.append(r["source_index"])
    return {"n":len(rows),"parser_abstentions":abstained,
            "reference_omission_count":len(missing_ids),
            "reference_omission_rate":len(missing_ids)/len(rows),
            "by_physical_db":dict(sorted(omitted.items())),
            "dbs_with_omissions":len(omitted),
            "mean_prompt_visible_tables":visible_total/len(rows),
            "missing_source_indices":sorted(missing_ids)}


def real_qwen_validation(rows: list[dict],original: dict,train_dbs: set[str]) -> dict:
    if {r["source_db_id"] for r in rows}&train_dbs:
        raise ValueError("TRAIN and independent VAL physical DB overlap")
    if (original.get("status")!="complete" or original.get("n")!=64 or
        original.get("source_validation_sha256")!=VAL_SHA or
        original.get("model_digest")!=MODEL_DIGEST or
        original.get("direct_ex")!=27 or original.get("guarded_ex")!=31 or
        len(original.get("cases",[]))!=64):
        raise ValueError("original real Qwen3.5-9B model identity/baseline drift")
    index={r["source_index"]:r for r in rows}
    seen=set()
    candidate_omits=Counter()
    candidate_omits_but_executable=Counter()
    ref_omit_and_wrong=Counter()
    model_parse_abstained=Counter()
    for c in original["cases"]:
        idx=c["source_index"]
        if idx not in index or idx in seen:
            raise ValueError("duplicate/unmatched model source ID")
        seen.add(idx)
        source=index[idx]
        prompt=source["messages"][1]["content"]
        if c["db_id"]!=source["source_db_id"] or c["prompt_sha256"]!=sha256(prompt.encode()).hexdigest():
            raise ValueError("original model/source physical question mismatch")
        ref=source_visibility(prompt,source["messages"][2]["content"])
        for key in ("direct","guarded"):
            out=source_visibility(prompt,c[key+"_sql"])
            if not out["parsed"]:
                model_parse_abstained[key]+=1
            if out["missing"]:
                candidate_omits[key]+=1
                if c[key]["execution_ok"]:
                    candidate_omits_but_executable[key]+=1
            if ref["missing"] and not c[key]["ex"]:
                ref_omit_and_wrong[key]+=1
    if len(seen)!=64:
        raise ValueError("original Qwen missing independent physical source tasks")
    return {
        "n":64,"physical_dbs":4,"original_direct_ex_correct":27,
        "original_guarded_ex_correct":31,
        "model_sql_named_tables_not_visible_in_prompt":dict(candidate_omits),
        "such_model_sql_still_physically_executable":dict(candidate_omits_but_executable),
        "reference_omitted_table_and_model_wrong":dict(ref_omit_and_wrong),
        "model_sql_table_parser_abstentions":dict(model_parse_abstained),
    }


def audit(train: Path,val: Path,model:Path) -> dict:
    tr=original_rows(train,TRAIN_SHA,242,16)
    va=original_rows(val,VAL_SHA,64,4)
    t=reference_census(tr)
    v=reference_census(va)
    m=real_qwen_validation(va,json.loads(model.read_text()),{r["source_db_id"] for r in tr})
    gate=(t["reference_omission_rate"]>=MIN_SOURCE_RATE and
          v["reference_omission_rate"]>=MIN_SOURCE_RATE and
          not t["parser_abstentions"] and not v["parser_abstentions"])
    return {
        "status":"COMPLETE_ORIGINAL_BIRD_TRAIN_TO_INDEPENDENT_VAL_SCHEMA_VISIBILITY_DIAGNOSTIC",
        "protocol":PROTOCOL,"train242_sha256":TRAIN_SHA,
        "val64_sha256":VAL_SHA,"model_digest":MODEL_DIGEST,
        "original_train_source":t,"independent_original_validation_source":v,
        "independent_real_Qwen3_5_9B":m,
        "schema_recall_gate_passed":gate,
        "frozen_minimum_missing_source_rate":MIN_SOURCE_RATE,
        "new_model_calls":0,"new_training_steps":0,
        "DEV500_used":False,"HOLDOUT300_used":False,
        "claim_boundary":"Lexical FROM/JOIN visibility of publisher-reference tables is a source-level information audit, not verified ground truth or a new EX score. CTE and quoting handled; any parser abstentions are explicit. Model stays Qwen3.5-9B. No source-exposed cases used to tune selector."
    }


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--validation",type=Path,required=True)
    p.add_argument("--model-summary",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():
        raise FileExistsError("immutable source-only audit must not overwrite output")
    result=audit(a.train,a.validation,a.model_summary)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("SCHEMA_VISIBILITY_CAUSAL_REACHABILITY_GATE",json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
