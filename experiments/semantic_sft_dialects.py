"""BIRD TRAIN-only SQL-pattern SFT across SQLite and conservatively portable MySQL.

A question is supervised ONCE, in one deterministic dialect, never duplicated
for both syntaxes. The source dataset must already be database-disjoint via
semantic_sft_data.prepare_semantic_sft_rows. The converter accepts only
read-only, portable SQL and checks SQLGlot round-trip canonical structure.
Unsupported date/time and custom functions remain SQLite, not hallucinated
MySQL positives. MySQL dialect parseability is NOT execution equivalence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from experiments.semantic_sft_data import make_semantic_prompt, sql_is_readonly


def _single_ast(sql: str, dialect: str):
    import sqlglot
    from sqlglot import exp
    try:
        nodes=sqlglot.parse(sql,read=dialect)
    except (ValueError,TypeError,sqlglot.errors.SqlglotError):
        return None
    if len(nodes)!=1 or nodes[0] is None:
        return None
    ast=nodes[0]
    if not isinstance(ast,(exp.Select,exp.Union,exp.Intersect,exp.Except)):
        return None
    if any(isinstance(node,(exp.Insert,exp.Delete,exp.Update,exp.Create,
                            exp.Drop,exp.Alter,exp.Command)) for node in ast.walk()):
        return None
    return ast


def transfer_portable_sql(sql: str) -> str | None:
    """Accept only a conservative subset with an identical SQLite round trip."""
    import sqlglot
    from sqlglot import exp
    ast=_single_ast(sql,"sqlite")
    if ast is None:
        return None
    portable_func=(exp.Count,exp.Sum,exp.Avg,exp.Min,exp.Max,exp.Coalesce)
    for node in ast.walk():
        if isinstance(node,exp.Func) and not isinstance(node,portable_func):
            return None
    try:
        mysql=ast.sql(dialect="mysql",pretty=False)
        mysql_ast=_single_ast(mysql,"mysql")
        if mysql_ast is None:
            return None
        recovered=mysql_ast.sql(dialect="sqlite",pretty=False)
        back=_single_ast(recovered,"sqlite")
        if back is None:
            return None
        # Parser-based equivalence of supported *syntax*, not data semantics.
        if back.sql(dialect="sqlite",pretty=False) != ast.sql(dialect="sqlite",pretty=False):
            return None
    except (ValueError,TypeError,sqlglot.errors.SqlglotError):
        return None
    return mysql


def _transfer_mysql_schema(sqlite_ddl: str) -> str | None:
    import sqlglot
    from sqlglot import exp
    try:
        parsed=sqlglot.parse(sqlite_ddl,read="sqlite")
        if not parsed or any(x is None or not isinstance(x,exp.Create) for x in parsed):
            return None
        statements=[node.sql(dialect="mysql",pretty=False) for node in parsed]
        if not all(sqlglot.parse_one(stmt,read="mysql") for stmt in statements):
            return None
        return ";\n".join(statements)+";"
    except (ValueError,TypeError,sqlglot.errors.SqlglotError):
        return None


def _convert_row(row: dict, dialect: str, mysql_schema: str | None,
                 mysql_target: str | None) -> dict:
    if dialect not in {"sqlite","mysql"}:
        raise ValueError("invalid SQL dialect")
    converted=dict(row)
    schema=row["schema_ddl"] if dialect=="sqlite" else mysql_schema
    target=row["target_sql"] if dialect=="sqlite" else mysql_target
    if not isinstance(schema,str) or not isinstance(target,str):
        raise ValueError("missing grounded schema or SQL target")
    prompt=make_semantic_prompt(row["question"],row.get("evidence") or "",schema)
    if dialect=="mysql":
        prompt=prompt.replace("expert SQLite query writer","expert MySQL 8 query writer")
    converted.update({"schema_ddl":schema,"target_sql":target,
                      "prompt":prompt,"dialect":dialect})
    return converted


def build_dialect_training_rows(
    train: list[dict], validation: list[dict], *,
    seed: str = "bird-qlora-mysql-source-shift-v1",
) -> tuple[list[dict],list[dict],dict]:
    if not train or not validation:
        raise ValueError("both database-disjoint splits required")
    train_dbs={r["db_id"] for r in train}
    val_dbs={r["db_id"] for r in validation}
    if train_dbs & val_dbs:
        raise ValueError("train/validation database overlap")
    out=[]
    counts=[]
    for part,source in (("train",train),("validation",validation)):
        rows=[]
        seen=set()
        metrics=Counter()
        for row in source:
            db_id=row.get("db_id")
            question=row.get("question")
            sql=row.get("target_sql")
            ddl=row.get("schema_ddl")
            if not all(isinstance(x,str) and x.strip() for x in (db_id,question,sql,ddl)):
                raise ValueError("incomplete training source")
            key=(db_id," ".join(question.split()).casefold())
            if key in seen:
                raise ValueError("duplicate supervised question in database")
            seen.add(key)
            if not sql_is_readonly(sql):
                raise ValueError("non-read-only training SQL")
            # One dialect per question. The original golden SQL remains intact.
            bucket=int(sha256(f"{seed}:{db_id}:{question}".encode()).hexdigest(),16)%2
            mysql_sql=transfer_portable_sql(sql) if bucket else None
            mysql_schema=_transfer_mysql_schema(ddl) if mysql_sql else None
            dialect="mysql" if mysql_sql and mysql_schema else "sqlite"
            rows.append(_convert_row(row,dialect,mysql_schema,mysql_sql))
            metrics[dialect]+=1
            if bucket and dialect=="sqlite":
                metrics["mysql_transfer_rejected"]+=1
        if not rows:
            raise ValueError("no source rows after dialect annotation")
        counts.append((part,metrics))
        out.append(rows)
    tr,val=out
    metrics=dict(counts)
    report={
        "schema":"bird-db-disjoint-sqlite-mysql-single-variant-pattern-sft-v1",
        "source":"BIRD filtered TRAIN; no DEV reference labels",
        "questions_train":len(tr),"questions_validation":len(val),
        "train_dbs":len(train_dbs),"validation_dbs":len(val_dbs),
        "train_dialect_counts":dict(sorted(metrics["train"].items())),
        "validation_dialect_counts":dict(sorted(metrics["validation"].items())),
        "mysql_syntax_transfers":metrics["train"].get("mysql",0),
        "train_duplicate_question_exposure":0,
        "train_val_database_overlap":0,
        "new_model_trained":False,
        "mysql_execution_verified":False,
        "generalization_gate":"separate database heldout SQL correctness + MySQL execution on real MySQL required",
        "limitations":"SQLGlot syntax roundtrip is NOT proof of equivalent DB results or MySQL production support.",
    }
    return tr,val,report


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--validation",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--heldout",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    a=p.parse_args()
    destinations=(a.output,a.heldout,a.report)
    if len({x.resolve() for x in destinations})!=3 or any(x.exists() for x in destinations):
        raise FileExistsError("refuse to overwrite frozen artifacts")
    train=[json.loads(x) for x in a.train.read_text().splitlines() if x.strip()]
    val=[json.loads(x) for x in a.validation.read_text().splitlines() if x.strip()]
    tr,vl,report=build_dialect_training_rows(train,val)
    for path,rows in ((a.output,tr),(a.heldout,vl)):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text("".join(json.dumps(r,sort_keys=True,ensure_ascii=False)+"\n"
                                for r in rows),encoding="utf-8")
    report["train_sha256"]=sha256(a.output.read_bytes()).hexdigest()
    report["validation_sha256"]=sha256(a.heldout.read_bytes()).hexdigest()
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("BIRD_MYSQL_PORTABLE_DATA_GATE",json.dumps(report,sort_keys=True))


if __name__=="__main__":
    main()
