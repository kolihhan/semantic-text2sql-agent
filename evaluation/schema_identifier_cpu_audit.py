"""CPU-only original Text2SQL schema-linking audit; zero model calls.

Exactly one proposed change: identifier CamelCase/acronym boundary splitting
BEFORE casefolding. Existing select_schema_context remains UNMODIFIED.

Replay original frozen 48 public BIRD TRAIN task IDs (already exposed).
First seal *label-blind* old/fixed schema selections. Separately inspect the
publisher reference SQL's physical table coverage. No SQL generation and no
accuracy inference from schema recall; fail closed when labels cannot parse.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

from evaluation.schema_context import (
    _tokens as old_tokens, _STOPWORDS, _description_rows, select_schema_context,
)
from semantic_sql.catalog import DatabaseCatalog

PROTOCOL="QWEN9B-BIRD-TRAIN48-CPU-IDENTIFIER-BOUNDARY-SCHEMA-COVERAGE-V1"
EXPECTED_SOURCE_SHA="006773fbd40894ba202c9d31aa54e0af95bcbd7f50e82d65871b36b7e2b89fc0"
SOURCE_N=48
KNOWN_DBS=(
    "retail_complains","shooting","mental_health_survey",
    "book_publishing_company","cookbook","address","shipping",
    "computer_student",
)
SCHEMA_GAIN_GATE={"min_new_gold_table_coverage_cases":3,
                  "max_lost_gold_table_coverage_cases":0,
                  "max_mean_context_bytes_ratio":1.20,
                  "require_complete_scoring":48}

def fixed_identifier_tokens(text: str) -> set[str]:
    # Both acronym->TitleCase and lower/digit->Upper boundaries must be
    # separated before case folding. The old implementation casefolds first.
    expanded=re.sub(r"([A-Z])([A-Z][a-z])",r"\1 \2",text)
    expanded=re.sub(r"([a-z0-9])([A-Z])",r"\1 \2",expanded)
    tokens=re.findall(r"[a-z0-9]+",expanded.casefold())
    return {w.rstrip("s") for w in tokens
            if w not in _STOPWORDS and len(w.rstrip("s"))>1}

def select_with_tokenizer(catalog,root:Path,db_id:str,question:str,tok):
    query_tokens=tok(question)
    descriptions=_description_rows(root,db_id)
    by_column={(table,col):desc for table,col,desc in descriptions}
    scores={}
    matched={}
    for table,info in catalog.tables.items():
        table_tokens=tok(table)
        column_tokens=set().union(*(tok(col) for col in info.columns)) if info.columns else set()
        description_tokens=set().union(*(
            tok(desc) for (dtable,col),desc in by_column.items()
            if dtable==table)) if info.columns else set()
        hits=sorted(query_tokens & (table_tokens|column_tokens|description_tokens))
        if hits:
            scores[table]=(len(query_tokens & table_tokens)*3+
                           len(query_tokens & column_tokens)*2+
                           len(query_tokens & description_tokens))
            matched[table]=hits
    selected={table for table,score in scores.items() if score>0}
    # Deliberately mirrors original one-hop expansion, including edge order.
    # Do NOT hide a second structural FK-graph intervention in this experiment.
    if selected:
        for source,_source_col,target,_target_col in catalog.foreign_keys:
            if source in selected or target in selected:
                selected.update(t for t in (source,target) if t in catalog.tables)
    uncertain=not query_tokens or not selected or max(scores.values(),default=0)<2
    if uncertain:
        selected=set(catalog.tables)
    lines=[]
    for table in sorted(selected):
        lines.append(f"{table}({', '.join(catalog.tables[table].columns)})")
    for a,ca,b,cb in catalog.foreign_keys:
        if a in selected and b in selected:
            lines.append(f"FK {a}.{ca} -> {b}.{cb}")
    descs=[(t,c,d) for t,c,d in descriptions
           if t in selected and c in catalog.tables[t].columns]
    if descs:
        lines.append("Column descriptions:")
        lines.extend(f"- {t}.{c}: {d}" for t,c,d in descs)
    context="\n".join(("Physical SQLite schema:",*lines))
    diagnostics={
        "selector":"deterministic_lexical_v1",
        "fallback_full_schema":uncertain,
        "query_tokens":sorted(query_tokens),
        "selected_tables":sorted(selected),
        "table_scores":{t:scores.get(t,0) for t in sorted(selected)},
        "matched_tokens":{t:matched.get(t,[]) for t in sorted(selected)},
        "schema_context_sha256":sha256(context.encode("utf-8")).hexdigest(),
    }
    return context,diagnostics

def compare_schema(catalog,root,db_id,question):
    # Exact baseline replay is a safety invariant: only CamelCase tokenization
    # is permitted to change; no graph rule, schema, or retrieval policy.
    original=select_schema_context(catalog,root,db_id,question)
    replay=select_with_tokenizer(catalog,root,db_id,question,old_tokens)
    if original!=replay:
        raise AssertionError("original schema selector has drifted from exact replay")
    fixed=select_with_tokenizer(catalog,root,db_id,question,fixed_identifier_tokens)
    return original,fixed

def source_cases(path:Path):
    raw=path.read_bytes()
    if sha256(raw).hexdigest()!=EXPECTED_SOURCE_SHA:
        raise ValueError("original BIRD TRAIN48 pre-model source bytes changed")
    rows=[json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
    if (len(rows)!=SOURCE_N or len({r["source_index"] for r in rows})!=SOURCE_N or
        set(r["db_id"] for r in rows)!=set(KNOWN_DBS) or
        any("SQL" in r or "gold" in r or "reference_sql" in r for r in rows)):
        raise ValueError("wrong or non-blind frozen source")
    return rows

def blind_audit(source:Path,dbroot:Path)->dict:
    cases=source_cases(source)
    dbs={}
    rows=[]
    for item in cases:
        db_id=item["db_id"]
        if db_id not in dbs:
            dbpath=dbroot/db_id/(db_id+".sqlite")
            if not dbpath.is_file():
                raise FileNotFoundError("missing verified original physical SQLite "+db_id)
            dbs[db_id]=DatabaseCatalog.from_sqlite(dbpath)
        catalog=dbs[db_id]
        old,new=compare_schema(catalog,dbroot,db_id,item["question"])
        old_tables=set(old[1]["selected_tables"])
        new_tables=set(new[1]["selected_tables"])
        changed=old[0]!=new[0]
        # Context bytes/selected TABLES recorded with no publisher gold.
        rows.append({
            "source_index":item["source_index"],"db_id":db_id,
            "question_sha256":item["question_sha256"],
            "old_table_set":sorted(old_tables),"new_table_set":sorted(new_tables),
            "added_tables":sorted(new_tables-old_tables),
            "removed_tables":sorted(old_tables-new_tables),
            "old_full_schema_fallback":old[1]["fallback_full_schema"],
            "new_full_schema_fallback":new[1]["fallback_full_schema"],
            "old_context_bytes":len(old[0].encode("utf-8")),
            "new_context_bytes":len(new[0].encode("utf-8")),
            "old_context_sha":old[1]["schema_context_sha256"],
            "new_context_sha":new[1]["schema_context_sha256"],
            "changed_context":changed
        })
    return {"protocol":PROTOCOL,
            "status":"GOLD_FREE_SCHEMA_SELECTIONS_FROZEN",
            "source_sha":EXPECTED_SOURCE_SHA,"n":SOURCE_N,
            "no_publisher_SQL_loaded":True,
            "same_original_schema_selector_replayed":True,
            "only_identifier_token_boundaries_changed":True,
            "no_model_calls":True,
            "cases":rows}

def score_audit(sealed:dict,gold_source:Path)->dict:
    # The reference is loaded ONLY AFTER blind_audit result is serialized.
    if (sealed.get("protocol")!=PROTOCOL or
        sealed.get("status")!="GOLD_FREE_SCHEMA_SELECTIONS_FROZEN" or
        sealed.get("source_sha")!=EXPECTED_SOURCE_SHA or
        sealed.get("n")!=SOURCE_N or
        sealed.get("no_publisher_SQL_loaded") is not True or
        len(sealed.get("cases",[]))!=SOURCE_N):
        raise ValueError("unsealed or tampered source")
    from scripts.prepare_birdtrain_qwen9b_sft_source import TRAIN_SHA, load_checked
    from sqlglot import parse_one, exp
    train=load_checked(gold_source,TRAIN_SHA)
    counts=Counter()
    table_rows=[]
    for row in sealed["cases"]:
        gold=train[row["source_index"]]
        if gold.get("db_id")!=row["db_id"]:
            raise ValueError("database identity mismatch")
        try:
            ast=parse_one(str(gold["SQL"]),read="sqlite",error_level="RAISE")
            if ast is None:raise ValueError("empty SQL")
            names={x.name.casefold() for x in ast.find_all(exp.Table)}
            if not names:
                raise ValueError("no physical table in publisher SQL")
            physical=set(t.casefold() for t in row["old_table_set"]+row["new_table_set"])
            if not names<=physical:
                # unknown table in gold or missing in BOTH selected contexts
                # can be real; score only known physical entities below.
                pass
            old={t.casefold() for t in row["old_table_set"]}
            new={t.casefold() for t in row["new_table_set"]}
            old_ok=names<=old
            new_ok=names<=new
            counts["scored"]+=1
            counts["old_all_gold_tables_covered"]+=int(old_ok)
            counts["new_all_gold_tables_covered"]+=int(new_ok)
            counts["rescues"]+=int(not old_ok and new_ok)
            counts["regressions"]+=int(old_ok and not new_ok)
            counts["old_table_mentions_covered"]+=len(names&old)
            counts["new_table_mentions_covered"]+=len(names&new)
            counts["gold_table_mentions"]+=len(names)
            table_rows.append({"source_index":row["source_index"],"db_id":row["db_id"],
                               "gold_table_count":len(names),
                               "old_all_covered":old_ok,"new_all_covered":new_ok,
                               "missing_old_tables":sorted(names-old),
                               "missing_new_tables":sorted(names-new)})
        except Exception as exc:
            # This is a CPU diagnostic; preserve parse errors, never impute misses.
            counts["parse_failed"]+=1
            table_rows.append({"source_index":row["source_index"],"db_id":row["db_id"],
                               "parser_error_type":type(exc).__name__})
        counts["schema_changed"]+=int(row["changed_context"])
        counts["added_tables"]+=len(row["added_tables"])
        counts["removed_tables"]+=len(row["removed_tables"])
        counts["old_context_bytes"]+=row["old_context_bytes"]
        counts["new_context_bytes"]+=row["new_context_bytes"]
    if counts["scored"]+counts["parse_failed"]!=SOURCE_N:
        raise ValueError("partial source scored")
    ratio=counts["new_context_bytes"]/max(1,counts["old_context_bytes"])
    positive=(counts["scored"]==SCHEMA_GAIN_GATE["require_complete_scoring"]
              and counts["rescues"]>=SCHEMA_GAIN_GATE["min_new_gold_table_coverage_cases"]
              and counts["regressions"]<=SCHEMA_GAIN_GATE["max_lost_gold_table_coverage_cases"]
              and ratio<=SCHEMA_GAIN_GATE["max_mean_context_bytes_ratio"])
    return {"protocol":PROTOCOL,
            "status":"COMPLETE_RETROSPECTIVE_ORIGINAL_TRAIN48_SCHEMA_COVERAGE",
            "source_sha":EXPECTED_SOURCE_SHA,"n":SOURCE_N,
            "counts":dict(counts),"mean_context_bytes_ratio":round(ratio,4),
            "predeclared_gate":SCHEMA_GAIN_GATE,
            "positive_screen_only":bool(positive),"promotion_allowed":False,
            "no_Qwen_calls":True,"no_training":True,
            "no_new_EX_score":True,"no_DEV500_or_HOLDOUT300":True,
            "limitation":"Previously exposed BIRD TRAIN48; physical reference table coverage is necessary but not sufficient for SQL semantic correctness. Neither SQL execution match nor independent generalization was tested.",
            "case_details":table_rows}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("blind","score"))
    p.add_argument("--source",type=Path)
    p.add_argument("--dbroot",type=Path)
    p.add_argument("--sealed",type=Path)
    p.add_argument("--train",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError("immutable audit output already exists")
    if a.mode=="blind":
        result=blind_audit(a.source,a.dbroot)
    else:
        result=score_audit(json.loads(a.sealed.read_text()),a.train)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False)+"\n")
    print("TEXT2SQL_SCHEMA_IDENTIFIER_CPU_AUDIT",json.dumps(
        {k:v for k,v in result.items() if k not in ("cases","case_details")},sort_keys=True),flush=True)
if __name__=="__main__":
    main()
