"""One-variable, no-model-call regression tests for original schema linking."""
import sqlite3
from evaluation.schema_context import _tokens as old_tokens, select_schema_context
from evaluation.schema_identifier_cpu_audit import (
    PROTOCOL,SCHEMA_GAIN_GATE,compare_schema,fixed_identifier_tokens,
    select_with_tokenizer,
)
from semantic_sql.catalog import DatabaseCatalog


def test_casefold_before_split_is_an_actual_existing_bug():
    # Original tokenizer performs casefold BEFORE CamelCase split.
    assert "birth" not in old_tokens("workerBirthDate")
    assert "date" not in old_tokens("workerBirthDate")
    assert {"worker","birth","date"} <= fixed_identifier_tokens("workerBirthDate")
    assert {"cbsa","name"} <= fixed_identifier_tokens("CBSAName")
    assert {"user","id"} <= fixed_identifier_tokens("UserID")
    assert fixed_identifier_tokens("city_name")==old_tokens("city_name")
    assert fixed_identifier_tokens("orders")==old_tokens("orders")


def test_identical_default_selector_plus_only_token_boundary_changed(tmp_path):
    dbroot=tmp_path
    dbdir=dbroot/"people"
    dbdir.mkdir()
    path=dbdir/"people.sqlite"
    con=sqlite3.connect(path)
    con.executescript("""
    CREATE TABLE EmployeeArchive(workerBirthDate TEXT, WorkerStartDate TEXT, employeeId INTEGER);
    CREATE TABLE InventoryPart(partNumber TEXT, warehouseId INTEGER);
    CREATE TABLE ServiceRequests(requestDate TEXT, requestType TEXT);
    """)
    con.close()
    cat=DatabaseCatalog.from_sqlite(path)
    q="What worker birth date is recorded?"
    old,corrected=compare_schema(cat,dbroot,"people",q)
    original=select_schema_context(cat,dbroot,"people",q)
    assert old==original
    assert old[1]["fallback_full_schema"]
    assert not corrected[1]["fallback_full_schema"]
    assert "EmployeeArchive" in corrected[1]["selected_tables"]
    assert "InventoryPart" not in corrected[1]["selected_tables"]
    assert old[1]["selected_tables"]==sorted(cat.tables)
    assert corrected[0]!=old[0]
    assert len(corrected[0])<len(old[0])


def test_underscore_case_and_no_extra_graph_rule(tmp_path):
    dbroot=tmp_path
    dbdir=dbroot/"demo"
    dbdir.mkdir()
    path=dbdir/"demo.sqlite"
    con=sqlite3.connect(path)
    con.executescript("""
    CREATE TABLE user_accounts(user_id INTEGER, user_name TEXT);
    CREATE TABLE billing_events(event_id INTEGER, user_id INTEGER,
        FOREIGN KEY(user_id) REFERENCES user_accounts(user_id));
    """)
    con.close()
    cat=DatabaseCatalog.from_sqlite(path)
    for q in ("How many billing events occurred?", "List the user accounts",
              "Give the user name from billing events"):
        old,new=compare_schema(cat,dbroot,"demo",q)
        assert old==new
    assert SCHEMA_GAIN_GATE["max_lost_gold_table_coverage_cases"]==0
    assert PROTOCOL.startswith("QWEN9B-BIRD-TRAIN48-CPU-")
