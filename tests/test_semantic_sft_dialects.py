"""Cross-dialect semantic SFT: same BIRD TRAIN answer pattern, no duplicate training question."""
from collections import Counter
import pytest

from experiments.semantic_sft_dialects import build_dialect_training_rows, transfer_portable_sql


def _row(db, n, sql):
    schema='CREATE TABLE "orders" ("id" INTEGER, "amount" REAL, "status" TEXT);'
    return {"db_id":db,"question":f"Which total for task {n}?",
            "evidence":"","schema_ddl":schema,"target_sql":sql,
            "prompt":f"SQLite task {n} with database schema."}


def test_portable_sql_keeps_join_group_filter_aggregation():
    sql=("""
        SELECT o.status, COUNT(DISTINCT o.id) AS n
        FROM orders AS o
        WHERE o.amount >= 10
        GROUP BY o.status
        HAVING COUNT(*) > 1
        ORDER BY n DESC LIMIT 5
    """)
    converted=transfer_portable_sql(sql)
    assert converted is not None
    assert "COUNT(DISTINCT" in converted.upper()
    assert "GROUP BY" in converted.upper()
    assert "HAVING" in converted.upper()
    assert "LIMIT" in converted.upper()


def test_refuse_unverified_dialect_sensitive_date_functions():
    assert transfer_portable_sql("SELECT strftime('%Y', id) FROM orders") is None
    assert transfer_portable_sql("DELETE FROM orders") is None
    assert transfer_portable_sql("SELECT * FROM orders; DROP TABLE orders") is None


def test_train_validation_database_disjoint_and_one_answer_per_question():
    train=[_row("a",i,"SELECT SUM(amount) FROM orders") for i in range(30)] + [
        _row("b",i+100,"SELECT COUNT(*) FROM orders WHERE status = 'paid'")
        for i in range(30)]
    val=[_row("heldout",i+200,"SELECT COUNT(DISTINCT id) FROM orders")
         for i in range(10)]
    tr,v,report=build_dialect_training_rows(train,val)
    assert len(tr)==len(train)
    assert len(v)==len(val)
    assert {r["db_id"] for r in tr}.isdisjoint({r["db_id"] for r in v})
    assert len({(r["db_id"],r["question"]) for r in tr})==len(tr)
    assert set(r["dialect"] for r in tr)=={"sqlite","mysql"}
    assert set(r["dialect"] for r in v)=={"sqlite","mysql"}
    assert report["questions_train"]==60
    assert report["train_dbs"]==2 and report["validation_dbs"]==1
    assert report["mysql_syntax_transfers"] > 0
    assert report["new_model_trained"] is False
    assert report["mysql_execution_verified"] is False
    assert all("expert MySQL 8 query writer" in r["prompt"] for r in tr if r["dialect"]=="mysql")


def test_mysql_schema_uses_backticks_and_preserves_columns():
    train=[_row("a",i,"SELECT amount FROM orders") for i in range(20)]
    val=[_row("b",i,"SELECT amount FROM orders") for i in range(20)]
    tr,_,_=build_dialect_training_rows(train,val)
    for row in tr:
        if row["dialect"]=="mysql":
            assert "`orders`" in row["schema_ddl"]
            assert "`amount`" in row["schema_ddl"]


def test_same_question_across_training_databases_is_not_hidden_leakage():
    train=[_row("a",i,"SELECT amount FROM orders") for i in range(10)]
    with pytest.raises(ValueError,match="overlap"):
        build_dialect_training_rows(train,train)
