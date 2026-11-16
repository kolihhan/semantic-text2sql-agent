from pathlib import Path
import sqlite3

from evaluation.qwen9b_full_schema_bird12 import schemas, aggregate, MODEL_DIGEST, PROTOCOL
from semantic_sql.catalog import DatabaseCatalog


def test_only_selected_tables_change_no_model_swap(tmp_path: Path):
    db = tmp_path / "data.sqlite"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE staff(staff_id INTEGER, full_name TEXT)")
        con.execute("CREATE TABLE payments(pay_id INTEGER, salary INTEGER)")
        con.execute("CREATE TABLE courses(course_id INTEGER, title TEXT)")
        con.execute("CREATE TABLE offices(office_id INTEGER, city TEXT)")
    catalog = DatabaseCatalog.from_sqlite(db)
    (small, metadata), (full, all_meta) = schemas(
        catalog, tmp_path, "fixture", "List staff full names")
    assert "staff(" in small and "payments(" not in small
    assert "staff(" in full and "payments(" in full
    assert set(metadata["selected_tables"]) < set(all_meta["selected_tables"])
    assert "Physical SQLite schema:" in small and "Physical SQLite schema:" in full


def synthetic_shards(rescues):
    cases=[]
    for i in range(12):
        good_before = i in (9, 10, 11)
        good_after = good_before or i < rescues
        cases.append({
            "case_id": str(i),
            "database_id": f"db{i//3}",
            "reference_scorable": True,
            "arms": {
                "original_lexical": {"strict_ex": good_before, "strict_bag": good_before},
                "expanded_physical": {"strict_ex": good_after, "strict_bag": good_after},
            },
        })
    return [{"protocol": PROTOCOL, "shard": k,
             "cohort_sha256": "frozen", "model_digest": MODEL_DIGEST,
             "cases": cases[3*k:3*k+3]} for k in range(4)]


def test_positive_requires_diverse_rescues():
    negative = aggregate(synthetic_shards(2))
    assert negative["positive_screen_only"] is False
    positive = aggregate(synthetic_shards(7))
    assert positive["net_rescues"] == 7
    assert positive["distinct_rescue_dbs"] == 3
    assert positive["positive_screen_only"] is True
    assert positive["promoted"] is False
