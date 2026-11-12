import sqlite3
from pathlib import Path

import pytest

from experiments.xiyan_mschema_frozen100 import build_mschema
from experiments.merge_xiyan_mschema import aggregate
from semantic_sql.catalog import DatabaseCatalog


def test_mschema_has_physical_columns_types_pk_fk_and_bounded_values(tmp_path):
    db = tmp_path / "mini.sqlite"
    with sqlite3.connect(db) as connection:
        connection.executescript("""
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER REFERENCES customers(id),
            total NUMERIC
        );
        INSERT INTO customers VALUES(1, 'Alice'), (2, 'Bob');
        INSERT INTO orders VALUES(1, 2, 19.5);
        """)
    catalog = DatabaseCatalog.from_sqlite(db)
    text = build_mschema(
        catalog=catalog, database_id="mini", selected_tables=["customers", "orders"],
        descriptions={("customers", "name"): "Customer's display name"}, sample_values=2,
    )
    assert "Database: mini" in text
    assert "Table: \"customers\"" in text
    assert '("id":INTEGER, Primary Key)' in text
    assert "Alice" in text and "Bob" in text
    assert "Customer's display name" in text
    assert '"orders"."customer_id" = "customers"."id"' in text
    with pytest.raises(ValueError):
        build_mschema(catalog=catalog, database_id="mini", selected_tables=["missing"], descriptions={})


def test_mschema_aggregate_strict_and_beats_frozen_baseline():
    shards = []
    for shard in range(4):
        rows = []
        for index in range(shard, 100, 4):
            rows.append({
                "case_id": str(index),
                "baseline_ex": index < 58,
                "mschema_ex": index < 63,
                "baseline_execution_success": True,
                "mschema_execution_success": True,
                "baseline_truncated": False,
                "mschema_truncated": False,
                "generation_latency_s": 2.5,
            })
        shards.append({"schema": "xiyan-mschema-frozen100-v1", "model": "xiyan",
                       "shard_index": shard, "num_shards": 4, "cases": rows})
    result = aggregate(shards)
    assert result["baseline_lexical_ex"] == 58
    assert result["mschema_ex"] == 63
    assert result["wrong_to_correct"] == 5
    assert result["correct_to_wrong"] == 0
    assert result["gate_met"]
    with pytest.raises(ValueError, match="four shards"):
        aggregate(shards[:3])
    shards[3]["cases"][0]["baseline_ex"] = False
    with pytest.raises(ValueError, match="baseline drifted"):
        aggregate(shards)


def test_mschema_workflow_reuses_exact_frozen_baseline():
    flow = Path(".github/workflows/xiyan-mschema-frozen100.yml").read_text()
    assert "gh run download 37090833784" in flow
    assert "xiyan-scv-shard-" in flow
    assert "python -m experiments.xiyan_mschema_frozen100" in flow
    assert "python -m experiments.merge_xiyan_mschema" in flow
    assert "--min-correct 61" in flow
    assert "XIYAN_MSCHEMA_TRIGGER" in flow
    assert "hf.co/wanhin/XiYanSQL-QwenCoder-7B-2504-gguf:Q4_K_M" in flow


def test_mschema_resolves_implicit_fk_to_single_primary_key(tmp_path):
    """BIRD schemas may declare REFERENCES parent without a target column."""
    db = tmp_path / "implicit.sqlite"
    with sqlite3.connect(db) as connection:
        connection.executescript("""
        CREATE TABLE parent (id INTEGER PRIMARY KEY, label TEXT);
        CREATE TABLE child (parent_id INTEGER REFERENCES parent, name TEXT);
        INSERT INTO parent VALUES (1, 'alpha');
        INSERT INTO child VALUES (1, 'beta');
        """)
    catalog = DatabaseCatalog.from_sqlite(db)
    assert any(dst_col is None for _, _, _, dst_col in catalog.foreign_keys)
    result = build_mschema(
        catalog=catalog, database_id="implicit",
        selected_tables=["parent", "child"], descriptions={},
    )
    assert '"child"."parent_id" = "parent"."id"' in result


def test_mschema_drops_ambiguous_implicit_composite_fk_without_crash(tmp_path):
    db = tmp_path / "composite.sqlite"
    with sqlite3.connect(db) as connection:
        connection.executescript("""
        CREATE TABLE parent (a INTEGER, b INTEGER, PRIMARY KEY(a, b));
        CREATE TABLE child (a INTEGER, b INTEGER, FOREIGN KEY(a, b) REFERENCES parent);
        """)
    catalog = DatabaseCatalog.from_sqlite(db)
    result = build_mschema(
        catalog=catalog, database_id="composite",
        selected_tables=["parent", "child"], descriptions={},
    )
    assert "Table:" in result
    assert "Foreign keys:" not in result


def test_mschema_aggregate_counts_truncated_case_as_wrong_not_as_missing():
    """Matches frozen score_query(): truncation is counted as incorrect."""
    shards = []
    for shard in range(4):
        rows = []
        for index in range(shard, 100, 4):
            rows.append({
                "case_id": str(index),
                "baseline_ex": index < 58,
                "mschema_ex": index < 61,
                "baseline_execution_success": True,
                "mschema_execution_success": True,
                "baseline_truncated": index == 99,
                "mschema_truncated": index == 98,
                "generation_latency_s": 1.0,
            })
        shards.append({"schema": "xiyan-mschema-frozen100-v1", "model": "xiyan",
                       "shard_index": shard, "num_shards": 4, "cases": rows})
    result = aggregate(shards)
    assert result["baseline_lexical_ex"] == 58
    assert result["mschema_ex"] == 61
    assert result["baseline_truncated"] == 1
    assert result["mschema_truncated"] == 1
    # Unsafe: a truncated result must never count toward correctness.
    target = next(row for shard in shards for row in shard["cases"] if row["case_id"] == "98")
    target["mschema_ex"] = True
    with pytest.raises(ValueError, match="truncated"):
        aggregate(shards)
