"""Contract for train-only, schema-grounded semantic SQL QLoRA data."""
from __future__ import annotations

import json
import pytest

from experiments.semantic_sft_data import (
    render_bird_schema, sql_is_readonly, prepare_semantic_sft_rows,
    tokenize_semantic_pair, make_semantic_prompt,
)

SCHEMAS = [
    {
        "db_id": "people_db",
        "table_names_original": ["people", "visits"],
        "column_names_original": [[-1, "*"], [0, "id"], [0, "full_name"], [1, "id"], [1, "person_id"]],
        "column_types": ["text", "number", "text", "number", "number"],
        "primary_keys": [1, 3],
        "foreign_keys": [[4, 1]],
    },
    {
        "db_id": "food_db",
        "table_names_original": ["food", "orders"],
        "column_names_original": [[-1, "*"], [0, "id"], [0, "category"], [1, "id"], [1, "food_id"]],
        "column_types": ["text", "number", "text", "number", "number"],
        "primary_keys": [1, 3],
        "foreign_keys": [[4, 1]],
    },
]


def test_schema_uses_true_table_columns_primary_and_foreign_keys():
    ddl = render_bird_schema(SCHEMAS[0])
    assert 'CREATE TABLE "people"' in ddl
    assert '"full_name" TEXT' in ddl
    assert 'CREATE TABLE "visits"' in ddl
    assert 'FOREIGN KEY ("person_id") REFERENCES "people" ("id")' in ddl
    assert 'PRIMARY KEY ("id")' in ddl
    assert '"*"' not in ddl
    assert ddl == render_bird_schema(SCHEMAS[0])


@pytest.mark.parametrize("sql", [
    "SELECT full_name FROM people",
    "WITH x AS (SELECT id FROM people) SELECT * FROM x",
    "SELECT full_name FROM people UNION SELECT category FROM food",
])
def test_accepts_readonly_sql(sql):
    assert sql_is_readonly(sql)


@pytest.mark.parametrize("sql", [
    "DELETE FROM people", "UPDATE people SET id = 1", "DROP TABLE people",
    "SELECT 1; DELETE FROM people", "SELECT 1; SELECT 2", "PRAGMA table_info(people)",
    "WITH x AS (DELETE FROM people RETURNING id) SELECT * FROM x",
])
def test_rejects_unsafe_sql(sql):
    assert not sql_is_readonly(sql)


def test_group_disjoint_validation_and_no_gold_in_prompt():
    rows = [
        {"db_id": "people_db", "question": "Who visited?", "evidence": "visitor refers to person_id",
         "SQL": "SELECT full_name FROM people"},
        {"db_id": "people_db", "question": "How many visits?", "evidence": "",
         "SQL": "SELECT COUNT(*) FROM visits"},
        {"db_id": "food_db", "question": "Which food?", "evidence": "",
         "SQL": "SELECT category FROM food"},
        {"db_id": "food_db", "question": "Where food?", "evidence": "",
         "SQL": "SELECT id FROM food"},
    ]
    result = prepare_semantic_sft_rows(
        rows, SCHEMAS, seed="v1", validation_db_fraction=0.5, min_train=1, min_val=1
    )
    train, validation, report = result
    assert len(train) == len(validation) == 2
    assert set(x["db_id"] for x in train).isdisjoint({x["db_id"] for x in validation})
    for item in train + validation:
        assert item["target_sql"] not in item["prompt"]
        assert item["target_sql"].startswith("SELECT")
        assert item["schema_ddl"].startswith("CREATE TABLE")
        assert item["prompt"].endswith("SQL:\n")
    assert report["source"] == "BIRD filtered TRAIN ONLY"
    assert report["train"] == 2 and report["validation"] == 2


def test_prevent_dev_leakage_and_skip_invalid_gold():
    rows = [
        {"db_id": "people_db", "question": "Who visited?", "evidence": "",
         "SQL": "SELECT full_name FROM people"},
        {"db_id": "people_db", "question": "Delete rows", "evidence": "",
         "SQL": "DELETE FROM people"},
        {"db_id": "food_db", "question": "Which food?", "evidence": "",
         "SQL": "SELECT category FROM food"},
    ]
    with pytest.raises(ValueError, match="overlap"):
        prepare_semantic_sft_rows(rows, SCHEMAS, dev_db_ids={"people_db"},
                                  min_train=1, min_val=1)
    rows = rows[:2] + [rows[-1]]
    train, val, report = prepare_semantic_sft_rows(
        rows, SCHEMAS, validation_db_fraction=0.5, min_train=1, min_val=1
    )
    assert report["rejected"]["non_readonly_or_unparseable_sql"] == 1


class FakeTokenizer:
    eos_token = "</s>"
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) % 100 + 1 for ch in text]


def test_completion_only_masking_and_fail_closed_long_prompt():
    item = {"prompt": "Schema and question\nSQL:\n", "target_sql": "SELECT 1"}
    tokens = tokenize_semantic_pair(FakeTokenizer(), item, max_length=64)
    first_labels = tokens["labels"]
    prefix_len = len(item["prompt"])
    assert first_labels[:prefix_len] == [-100] * prefix_len
    assert all(label > 0 for label in first_labels[prefix_len:])
    with pytest.raises(ValueError, match="too_long"):
        tokenize_semantic_pair(FakeTokenizer(), item, max_length=10)
