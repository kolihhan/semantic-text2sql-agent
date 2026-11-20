"""ID-agnostic source-only parent-count gate; no publisher gold or Qwen."""
from scripts.dev500_parent_count_generic_source_gate import detect, singular


def schema():
    return {
        "Patient":{"primary_key":["ID"],"declared_fk":[]},
        "Laboratory":{"primary_key":["ID","Date"],
                      "declared_fk":[{"from":"ID","to_table":"Patient","to_col":"ID"}]},
        "members":{"primary_key":["member_id"],"declared_fk":[]},
        "major":{"primary_key":["major_id"],"declared_fk":[]},
        "Match":{"primary_key":["id"],"declared_fk":[{"from":"league_id","to_table":"League","to_col":"id"}]},
        "League":{"primary_key":["id"],"declared_fk":[]},
    }


def test_qualified_fk_parent_count_hypothesis_is_generic():
    q="How many patients with abnormal lab readings have no description?"
    sql=("SELECT COUNT(*) FROM Laboratory L JOIN Patient P ON L.ID = P.ID "
         "WHERE L.CRP = '+' AND P.Description IS NULL")
    a=detect(q,sql,schema())
    assert a is not None
    assert a["parent"]=="Patient" and a["child"]=="Laboratory"
    assert "COUNT(DISTINCT P.ID)" in a["proposed_sql"]


def test_falsifiers_no_unsafe_rewrite():
    s=schema()
    assert singular("patients")=="patient"
    assert detect("How many matches were played?",
                  "SELECT COUNT(*) FROM Match JOIN League ON Match.league_id = League.id",s) is None
    assert detect("How many members are in a major?",
                  "SELECT COUNT(*) FROM members JOIN major ON members.major_id=major.major_id",s) is None
    assert detect("What percentage of patients were admitted?",
                  "SELECT 100.0 * COUNT(*) / 12 FROM Laboratory JOIN Patient ON Laboratory.ID=Patient.ID",s) is None
    assert detect("How many patients have been admitted?",
                  "SELECT COUNT(*) FROM Laboratory JOIN Patient ON Laboratory.ID=Patient.ID GROUP BY Patient.ID",s) is None


def test_missing_or_unverified_pk_fk_abstains():
    q="How many patients had a test?"
    sql="SELECT COUNT(*) FROM Laboratory L JOIN Patient P ON L.ID=P.ID"
    assert detect(q,sql,{"Patient":{"primary_key":["ID"]},"Laboratory":{"declared_fk":[]}}) is None
    assert detect(q,sql,{"Patient":{"primary_key":["ID","other"]},
                         "Laboratory":{"declared_fk":[{"from":"ID","to_table":"Patient","to_col":"ID"}]}}) is None
    assert detect(q,sql,{"Patient":{"primary_key":["ID"]},
                         "Laboratory":{"declared_fk":[{"from":"ID","to_table":"Patient","to_col":"ID"}]} }) is not None
