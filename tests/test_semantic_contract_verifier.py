from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from semantic_sql.inference import run_guarded
from semantic_sql.semantic_contracts import (
    detect_semantic_concerns, outer_select, projection_columns
)


def codes(q: str, s: str) -> tuple[set[str], set[str]]:
    found=detect_semantic_concerns(q,s)
    return ({i.code for i in found if i.blocking},
            {i.code for i in found if not i.blocking})


def test_outer_projection_does_not_stop_at_subquery_from_or_quoted_string():
    s=("SELECT COUNT(*) * 100.0 / (SELECT COUNT(*) FROM molecule) "
       "AS pct FROM atom JOIN molecule ON atom.molecule_id = molecule.molecule_id")
    p=outer_select(s)
    assert p is not None and "AS pct" in p
    assert "FROM atom" not in p
    assert len(projection_columns(p))==1
    assert outer_select("WITH q AS (SELECT 1) SELECT 1 FROM q") is None
    assert outer_select("SELECT 'not FROM here', SUM(x) FROM t")=="'not FROM here', SUM(x)"


def test_no_gold_or_domain_specific_molecule_code_in_detector():
    import inspect
    import semantic_sql.semantic_contracts as mod
    source=inspect.getsource(mod.detect_semantic_concerns)
    assert "gold" not in source.casefold().replace("no gold", "")
    assert "molecule" not in source
    assert "provider" not in source
    assert "execute_" not in source


def test_explicit_semantic_contradictions_are_blocking():
    q="What was the growth rate of total loans between 1996 and 1997?"
    direct=("SELECT SUM(CASE WHEN yr=1997 THEN amount END) AS numerator, "
            "SUM(CASE WHEN yr=1996 THEN amount END) AS denominator FROM loans")
    b,a=codes(q,direct)
    assert b=={"growth_rate_not_computed"} and not a
    corrected=("SELECT (SUM(CASE WHEN yr=1997 THEN amount END) * 1.0 / "
               "NULLIF(SUM(CASE WHEN yr=1996 THEN amount END),0)) - 1 AS growth_rate "
               "FROM loans")
    assert codes(q,corrected)==(set(),set())

    age_sql=("SELECT CASE WHEN strftime('%Y', client.birth_date) - "
             "strftime('%Y', '1900-01-01') < 30 THEN 'Young' ELSE 'Old' END "
             "FROM client")
    assert "age_wrong_1900_reference" in codes(
        "Break down people by age group",age_sql)[0]
    season_sql="SELECT COUNT(*) FROM Match WHERE Match.season = 2015"
    assert "two_year_season_collapsed" in codes(
        "How many games in the 2015–2016 season?",season_sql)[0]


def test_target_grain_is_advisory_and_does_not_force_a_wrong_repair():
    q="How many carcinogenic molecules consisted of Nitrogen?"
    s=("SELECT COUNT(*) FROM atom a JOIN molecule m "
       "ON a.molecule_id=m.molecule_id WHERE a.element='N'")
    b,a=codes(q,s)
    assert not b
    assert "possible_join_fanout_grain" in a

    ratio_q="Calculate the percentage of molecules containing nitrogen."
    ratio_sql=("SELECT COUNT(*) * 100.0 / "
               "(SELECT COUNT(*) FROM molecule) AS percent "
               "FROM atom JOIN molecule ON atom.molecule_id=molecule.molecule_id")
    b,a=codes(ratio_q,ratio_sql)
    assert not b and "possible_join_fanout_grain" in a

    q="Identify atoms connected to atoms from another molecule"
    b,a=codes(q,"SELECT atom_id, element FROM atom WHERE molecule_id='TR181'")
    assert not b and "connectivity_relation_maybe_missing" in a


class SequenceProvider:
    def __init__(self,*responses):
        self.responses=list(responses)
        self.calls=0

    def complete_text(self,*,system,user):
        self.calls+=1
        if not self.responses:
            raise AssertionError("unexpected model call")
        return self.responses.pop(0)


@pytest.fixture
def loans(tmp_path:Path):
    db=tmp_path/"loans.sqlite"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE loans(yr INTEGER,amount REAL)")
        con.executemany("INSERT INTO loans VALUES (?,?)",[(1996,100.0),(1997,150.0)])
    return db


def test_opt_in_routes_detected_rate_contradiction_to_existing_bounded_repair(loans):
    q="What is the growth rate of the total loans between 1996 and 1997?"
    initial=("SELECT SUM(CASE WHEN yr=1997 THEN amount ELSE 0 END) AS numerator, "
             "SUM(CASE WHEN yr=1996 THEN amount ELSE 0 END) AS denominator FROM loans")
    corrected=("SELECT (SUM(CASE WHEN yr=1997 THEN amount ELSE 0 END) - "
               "SUM(CASE WHEN yr=1996 THEN amount ELSE 0 END)) * 1.0 / "
               "NULLIF(SUM(CASE WHEN yr=1996 THEN amount ELSE 0 END),0) "
               "AS growth_rate FROM loans")

    control=SequenceProvider(initial)
    baseline=run_guarded(database=loans,provider=control,question=q,
                         schema_context="loans(yr, amount)",
                         max_repairs=1,max_rows=10)
    assert baseline.status=="ok" and baseline.candidate.sql==initial
    assert baseline.rows==((150.0,100.0),)
    assert control.calls==1
    assert all("semantic_" not in stage.summary for stage in baseline.stages)

    treatment=SequenceProvider(initial,corrected)
    revised=run_guarded(database=loans,provider=treatment,question=q,
                        schema_context="loans(yr, amount)",
                        max_repairs=1,max_rows=10,
                        semantic_contract_checks=True)
    assert revised.status=="ok" and revised.candidate.sql==corrected
    assert revised.rows==((0.5,),)
    assert treatment.calls==2
    assert [s.name for s in revised.stages]==[
        "sql","verify","repair","verify","execute"]
    assert "semantic_growth_rate_not_computed" in revised.stages[1].summary


def test_advisory_does_not_trigger_repair_and_exhausted_block_refuses(loans):
    advisory=SequenceProvider("SELECT COUNT(*) FROM loans l JOIN loans m ON l.yr=m.yr")
    result=run_guarded(database=loans,provider=advisory,
                       question="How many loans are counted?",
                       schema_context="loans(yr,amount)",max_repairs=1,
                       semantic_contract_checks=True)
    assert result.status=="ok" and advisory.calls==1

    bad=SequenceProvider("SELECT 150 AS numerator, 100 AS denominator FROM loans")
    denied=run_guarded(database=loans,provider=bad,
                       question="What is the growth rate?",schema_context="loans(yr,amount)",
                       max_repairs=0,semantic_contract_checks=True)
    assert denied.status=="verification_failed"
    assert bad.calls==1
    assert denied.verification and any(i.code.startswith("semantic_") for i in denied.verification.issues)
