"""Relational grounding invariant tests: generic SQLite, not BIRD/Spider gold."""
import sqlite3

import pytest

from experiments.relational_grounding_v2 import (
    relational_metadata,grounded_value_hits,enrich_mschema,
)


@pytest.fixture
def db(tmp_path):
    path=tmp_path/"generic.sqlite"
    with sqlite3.connect(path) as con:
        con.executescript("""
          PRAGMA foreign_keys=ON;
          CREATE TABLE clients(
              id INTEGER PRIMARY KEY,
              name TEXT NOT NULL UNIQUE
          );
          CREATE TABLE orders(
              id INTEGER PRIMARY KEY,
              client_id INTEGER NOT NULL REFERENCES clients(id),
              label TEXT
          );
          CREATE TABLE profiles(
              id INTEGER PRIMARY KEY,
              client_id INTEGER NOT NULL UNIQUE REFERENCES clients(id),
              nickname TEXT
          );
          CREATE TABLE country_region(
              country TEXT NOT NULL,
              region TEXT NOT NULL,
              PRIMARY KEY(country,region)
          );
          CREATE TABLE places(
              id INTEGER PRIMARY KEY,
              country TEXT,region TEXT,
              FOREIGN KEY(country,region)
                  REFERENCES country_region(country,region)
          );
          CREATE UNIQUE INDEX partial_tag ON orders(label) WHERE label != '';
          INSERT INTO clients(id,name) VALUES (1,'Alice'),(2,'Bob');
          INSERT INTO orders(id,client_id,label) VALUES
            (10,1,'bought'),(11,1,'bought'),(12,2,'gift');
          INSERT INTO profiles(id,client_id,nickname) VALUES (1,1,'A'),(2,2,'B');
          INSERT INTO country_region(country,region) VALUES ('US','CA');
          INSERT INTO places(id,country,region) VALUES (1,'US','CA');
        """)
    return path


def test_fk_declarations_do_not_invent_one_to_one(db):
    m=relational_metadata(db,["clients","orders","profiles"])
    fks={(fk["source"],fk["target"]):fk for fk in m["foreign_keys"]}
    orders=fks[("orders","clients")]
    assert orders["target_unique_declared"] is True
    assert orders["source_unique_declared"] is False
    profile=fks[("profiles","clients")]
    assert profile["target_unique_declared"] is True
    assert profile["source_unique_declared"] is True
    assert not any(k["columns"]==("label",) for k in m["tables"]["orders"]["keys"])


def test_composite_join_preserves_both_columns_and_direction(db):
    m=relational_metadata(db,["country_region","places"])
    assert len(m["foreign_keys"])==1
    fk=m["foreign_keys"][0]
    assert fk["source_columns"]==("country","region")
    assert fk["target_columns"]==("country","region")
    assert fk["target_unique_declared"] and not fk["source_unique_declared"]


def test_unknown_relationship_when_destination_not_selected(db):
    m=relational_metadata(db,["orders"])
    assert m["foreign_keys"]==[]
    with pytest.raises(ValueError,match="not present"):
        relational_metadata(db,["orders","made_up"])
    with pytest.raises(ValueError,match="distinct"):
        relational_metadata(db,["orders","orders"])


def test_quoted_value_hit_never_fabricates_question_semantics(db):
    m=relational_metadata(db,["clients","orders"])
    hits=grounded_value_hits(db,m,question='Find customers named "Alice" and order "gift"')
    assert hits["quotes"]==["Alice","gift"]
    assert len(hits["hits"])==2
    assert all(h["claim"].endswith("unknown") for h in hits["hits"])
    assert hits["probes"]<=24
    assert {(h["table"],h["column"]) for h in hits["hits"]}=={
        ("clients","name"),("orders","label")
    }


def test_sql_injection_in_quote_stays_data_not_sql(db):
    m=relational_metadata(db,["clients"])
    q='Find "Alice\\x27); DROP TABLE clients; --"'
    hits=grounded_value_hits(db,m,question=q,max_probes=2)
    assert not hits["hits"]
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT COUNT(*) FROM clients").fetchone()[0]==2


def test_max_probe_and_quote_budget_fail_closed(db):
    m=relational_metadata(db,["clients","orders"])
    for bad in (0,65):
        with pytest.raises(ValueError,match="bounded"):
            grounded_value_hits(db,m,question='"Alice"',max_probes=bad)
    result=grounded_value_hits(db,m,question='"Alice" "Bob" "gift"',max_probes=1)
    assert result["probes"]==1


def test_annotated_mschema_is_monotonic_and_reproducible(db):
    baseline="Database: db\nTable: clients\nTable: orders\n..."
    v1,meta1=enrich_mschema(
        base_mschema=baseline,database=db,selected_tables=["clients","orders","profiles"],
        question='How many orders are there for "Alice"?')
    v2,meta2=enrich_mschema(
        base_mschema=baseline,database=db,selected_tables=["clients","orders","profiles"],
        question='How many orders are there for "Alice"?')
    assert v1==v2 and meta1==meta2
    assert v1.startswith(baseline+"\n")
    assert "declared many-to-1" in v1
    assert "declared 1-to-1" in v1
    assert "exact case-insensitive match" in v1
    assert meta1["literal_hits"]==1
    assert meta1["preserves_original_mschema"]
    with pytest.raises(ValueError,match="nonempty"):
        enrich_mschema(base_mschema="",database=db,
                       selected_tables=["clients"],question="Count")


def test_not_treat_literal_samples_as_cardinality_proof(db):
    baseline="Table: orders; Table: clients"
    v,m=enrich_mschema(
        base_mschema=baseline,database=db,selected_tables=["clients","orders"],
        question="How many orders per client?")
    assert "many-to-1" in v
    assert "actual FK integrity not checked" in v
    assert m["literal_hits"]==0 and m["value_probes"]==0
