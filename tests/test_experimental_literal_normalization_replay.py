"""No-gold controls for deterministic value normalization counterfactual."""
import sqlite3
from scripts.experimental_literal_normalization_replay import build_candidate


def database(tmp_path):
    file=tmp_path/"test.sqlite"
    conn=sqlite3.connect(file)
    conn.executescript("""
      CREATE TABLE sets(id INT, type TEXT);
      INSERT INTO sets VALUES (1,'Commander'),(2,'Modern');
      CREATE TABLE cards(id INT, bond_type TEXT);
      INSERT INTO cards VALUES (10,'='),(11,'-');
    """)
    conn.commit()
    return conn,file


def test_case_corrected_only_with_unique_real_witness(tmp_path):
    conn,db=database(tmp_path)
    sql="SELECT id FROM sets WHERE type = 'commander'"
    result=build_candidate(sql,conn,db)
    assert result["status"]=="FROZEN_NORMALIZATION_CANDIDATE"
    assert result["sql"]=="SELECT id FROM sets WHERE type = 'Commander'"
    assert len(result["replacements"])==1
    conn.close()


def test_no_change_when_original_already_exact(tmp_path):
    conn,db=database(tmp_path)
    sql="SELECT id FROM sets WHERE type='Commander'"
    assert build_candidate(sql,conn,db)["status"]=="NO_UNIQUE_NORMALIZED_WITNESS"
    conn.close()


def test_no_unsafe_prefix_expansion(tmp_path):
    conn,db=database(tmp_path)
    sql="SELECT id FROM sets WHERE type='Command'"
    assert build_candidate(sql,conn,db)["status"]=="NO_UNIQUE_NORMALIZED_WITNESS"
    conn.close()


def test_no_cross_column_or_fuzzy_correction(tmp_path):
    conn,db=database(tmp_path)
    sql="SELECT id FROM sets WHERE type='Alien'"
    assert build_candidate(sql,conn,db)["status"]=="NO_UNIQUE_NORMALIZED_WITNESS"
    conn.close()


def test_space_normalization_for_symbolic_sql_literal(tmp_path):
    conn,db=database(tmp_path)
    sql="SELECT id FROM cards WHERE bond_type = '= '"
    d=build_candidate(sql,conn,db)
    assert d["status"]=="FROZEN_NORMALIZATION_CANDIDATE"
    assert d["sql"]=="SELECT id FROM cards WHERE bond_type = '='"
    conn.close()


def test_duplicate_predicates_both_corrected_without_other_changes(tmp_path):
    conn,db=database(tmp_path)
    sql="SELECT id FROM sets WHERE type='commander' AND type='commander'"
    d=build_candidate(sql,conn,db)
    assert d["status"]=="FROZEN_NORMALIZATION_CANDIDATE"
    assert d["sql"]=="SELECT id FROM sets WHERE type='Commander' AND type='Commander'"
    assert len(d["replacements"])==2
    conn.close()


def test_ambiguous_physical_witness_abstains(tmp_path):
    file=tmp_path/"ambig.sqlite"
    conn=sqlite3.connect(file)
    conn.executescript("CREATE TABLE x(v TEXT); INSERT INTO x VALUES ('Commander'),('COMMANDER');")
    conn.commit()
    assert build_candidate("SELECT v FROM x WHERE v='commander'",conn,file)["status"]=="NO_UNIQUE_NORMALIZED_WITNESS"
    conn.close()
