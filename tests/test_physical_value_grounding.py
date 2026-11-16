from __future__ import annotations

import sqlite3

from evaluation.physical_value_grounding import (
    candidate_literals,physical_value_columns,evidence_string,
)


def test_generic_entity_value_linking_queries_only_read_only_source_sqlite(tmp_path):
    db=tmp_path/"example.sqlite"
    with sqlite3.connect(db) as con:
        con.executescript("""
            CREATE TABLE country(country TEXT, code INTEGER);
            CREATE TABLE address(district TEXT, street TEXT);
            CREATE TABLE customer(first_name TEXT, last_name TEXT);
            INSERT INTO country VALUES('India',1);
            INSERT INTO address VALUES('Punjab','Street');
            INSERT INTO customer VALUES('Mary','Smith');
        """)
    found=physical_value_columns(db,"List customer names from India")
    assert {"value":"India","table":"country","column":"country"} in found
    assert not any(x["column"]=="district" and x["value"]=="India" for x in found)
    assert found and "country.country" in evidence_string(found)
    assert "\n- country.country" in evidence_string(found)
    assert "\\n" not in evidence_string(found)
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT COUNT(*) FROM country").fetchone()[0]==1


def test_no_question_or_reference_answer_leak_and_bound_candidates():
    q="What customers from Mary Smith in 'India'?"
    literals=candidate_literals(q)
    assert "India" in literals and "Mary" in literals and "Smith" in literals
    assert "What" not in literals
    assert candidate_literals("how many customers?")==[]


def test_case_insensitive_matches_and_identifier_quotes(tmp_path):
    path=tmp_path/"quoted.sqlite"
    with sqlite3.connect(path) as con:
        con.execute('CREATE TABLE "weird "" table"("State Name" TEXT)')
        con.execute('INSERT INTO "weird "" table" VALUES (?)',("Brunei",))
    evidence=physical_value_columns(path,"Find Brunei")
    assert evidence==[{"value":"Brunei","table":'weird " table',"column":"State Name"}]
    assert physical_value_columns(path,"Find a missing Lithuania")==[]
