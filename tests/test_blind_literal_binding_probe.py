"""Synthetic negative controls, independent of any historical gold SQL."""
import sqlite3

from scripts.blind_physical_semantic_signals import _schema
from scripts.blind_literal_binding_probe import classify_operand, physical_probe


def make_db():
    conn=sqlite3.connect(":memory:")
    conn.executescript("""
      CREATE TABLE schools(id INTEGER PRIMARY KEY, county TEXT, doc TEXT);
      INSERT INTO schools VALUES (1,'Los Angeles County','Locally Funded');
      CREATE TABLE majors(id TEXT PRIMARY KEY, college TEXT, key TEXT);
      INSERT INTO majors VALUES ('M1','Computer Science','rec1N0upiVLy5esTO');
    """)
    conn.execute("PRAGMA query_only=ON")
    return conn


def probe(table,col,value,question=""):
    conn=make_db()
    tables,cols,_fks=_schema(conn)
    return classify_operand(conn,tables,cols,table,col,value,question,[table])


def test_exact_value_never_proposes_repair():
    x=probe("schools","county","Los Angeles County")
    assert x["status"]=="ALREADY_EXACT"


def test_normalized_real_value_has_a_witness_only():
    x=probe("schools","doc","locally funded","Schools locally funded")
    assert x["status"]=="NORMALIZED_SAME_COLUMN"
    assert x["possible_compatible_values"]==1
    assert "candidate_value_sha256" in x


def test_substring_not_declared_safe_correction():
    x=probe("schools","county","Los Angeles")
    assert x["status"]=="SUBSTRING_AMBIGUOUS"
    assert x["reason"]=="PARTIAL_PHYSICAL_MATCH_NOT_SYNONYM_PROOF"


def test_cross_column_exact_presence_is_not_automatic_remap():
    x=probe("majors","college","rec1N0upiVLy5esTO","major rec1N0upiVLy5esTO")
    assert x["status"]=="POSSIBLE_COLUMN_MISBIND"
    assert x["alternative_column_count"]==1
    assert x["alternate_binding_columns"]==["majors.key"]


def test_missing_value_abstains_without_unsafe_guess():
    x=probe("schools","county","Unknown Island")
    assert x["status"]=="ABSTAIN"
    assert x["reason"]=="NO_PHYSICAL_ALTERNATIVE_FOUND"


def test_parameterized_physical_probe_escapes_adversarial_value():
    conn=make_db()
    result,status=physical_probe(conn,"schools","county",'"county" = ?',"' OR 1=1 --")
    assert status=="OK"
    assert result==[]
