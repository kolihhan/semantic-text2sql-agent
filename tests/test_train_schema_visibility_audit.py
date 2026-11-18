"""Source-only parser safeguards for fixed original Qwen BIRD prompt visibility."""
from hashlib import sha256
import pytest

from scripts.audit_original_train_schema_visibility import (
    visible_tables, referenced_tables, source_visibility,
    reference_census, real_qwen_validation
)


def sample_prompt():
    return ("Question: Which cities?\nSchema context:\n"
            "Physical SQLite schema:\n"
            "Sales Orders(OrderNumber, CustomerID)\n"
            "social-media(bioguide, twitter)\n"
            "City(ID, Name)\n"
            "FK Sales Orders.CustomerID -> City.ID\n"
            "External evidence:\nregion = 'FROM fake_table JOIN anything'\n")


def test_schema_names_include_hyphen_spaces_and_are_case_insensitive():
    assert visible_tables(sample_prompt()) == {"sales orders","social-media","city"}
    sql=("SELECT t.OrderNumber FROM \x60Sales Orders\x60 t "
         "JOIN [City] c ON c.ID=t.CustomerID "
         "JOIN \x60social-media\x60 social ON 1=1")
    assert referenced_tables(sql)=={"sales orders","city","social-media"}
    assert not source_visibility(sample_prompt(),sql)["missing"]


def test_from_in_literals_and_comments_do_not_create_false_missing_tables():
    sql=("SELECT City.Name FROM City "
         "WHERE Name='FROM unwanted JOIN shadow' "
         "-- FROM leaked\n"
         "/* JOIN absent */")
    assert referenced_tables(sql)=={"city"}
    assert not source_visibility(sample_prompt(),sql)["missing"]


def test_cte_does_not_count_as_missing_physical_table():
    sql=("WITH cte AS (SELECT Name FROM City) "
         "SELECT Name FROM cte")
    assert referenced_tables(sql)=={"city"}
    assert not source_visibility(sample_prompt(),sql)["missing"]


def test_true_reference_table_absent_prompt_is_detected():
    result=source_visibility(sample_prompt(),"SELECT a.id FROM Actors a JOIN City c ON 1=1")
    assert result["parsed"] and result["missing"]==["actors"]


def test_source_census_is_not_implicit_accuracy_label():
    r={"source_index":1,"source_db_id":"s","messages":[
        {"role":"system","content":""},
        {"role":"user","content":sample_prompt()},
        {"role":"assistant","content":"SELECT name FROM Actors"}]}
    stats=reference_census([r])
    assert stats["reference_omission_count"]==1
    assert stats["reference_omission_rate"]==1.
    assert stats["by_physical_db"]=={"s":1}


def test_independent_model_validation_rejects_same_db():
    with pytest.raises(ValueError,match="overlap"):
        real_qwen_validation(
            [{"source_db_id":"existing"}], {}, {"existing"})


def test_independent_model_validation_rejects_unverified_model_digest():
    with pytest.raises(ValueError,match="identity"):
        real_qwen_validation(
            [{"source_db_id":"new"}], {"n":64,"model_digest":"switched-model"}, {"existing"})
