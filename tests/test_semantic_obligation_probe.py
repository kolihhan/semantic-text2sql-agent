"""Gold-free positive and negative controls for executable-but-wrong SQL signals."""
import sqlite3
from scripts.semantic_obligation_probe import detect_obligations


def flags(q, s):
    return set(detect_obligations(q,s)["flags"])


def test_explicit_identifier_bound_to_wrong_attribute_without_mutation():
    bad=flags(
        "Find record with ID 16.",
        "SELECT name FROM items WHERE artist LIKE '16%'",
    )
    assert "EXPLICIT_ID_BOUND_TO_NON_ID_ATTRIBUTE" in bad
    good=flags("Find record with ID 16.",
        "SELECT name FROM items WHERE item_id = 16")
    assert "EXPLICIT_ID_BOUND_TO_NON_ID_ATTRIBUTE" not in good


def test_named_bond_entity_uses_identity_not_bond_type():
    f=flags(
        "What is the carcinogenic label for bond TR001_2_4?",
        "SELECT m.label FROM molecule m JOIN bond b ON m.molecule_id = b.molecule_id WHERE b.bond_type = 'TR001_2_4'",
    )
    assert "NAMED_ENTITY_TOKEN_FILTERED_BY_NONKEY" in f
    safe=flags(
        "What is the label for bond TR001_2_4?",
        "SELECT m.label FROM molecule m JOIN bond b ON m.molecule_id=b.molecule_id WHERE b.bond_id = 'TR001_2_4'",
    )
    assert "NAMED_ENTITY_TOKEN_FILTERED_BY_NONKEY" not in safe


def test_can_be_executable_and_nonempty_but_semantically_wrong():
    db=sqlite3.connect(":memory:")
    db.executescript("""
        CREATE TABLE bond (bond_id TEXT, bond_type TEXT, label TEXT);
        INSERT INTO bond VALUES ('TARGET_5', 'S', 'correct'),('OTHER_8','TARGET_5','wrong');
    """)
    wrong="SELECT label FROM bond WHERE bond_type='TARGET_5'"
    correct="SELECT label FROM bond WHERE bond_id='TARGET_5'"
    assert db.execute(wrong).fetchall()==[("wrong",)]
    assert db.execute(correct).fetchall()==[("correct",)]
    f=flags("What is the label for bond TARGET_5?",wrong)
    assert "NAMED_ENTITY_TOKEN_FILTERED_BY_NONKEY" in f
    db.close()


def test_growth_rate_requires_computed_rate_not_just_numerator_denominator():
    q="What was the growth rate between 1996 and 1997?"
    bad=flags(q,"SELECT numerator, denominator FROM loans")
    good=flags(q,"SELECT (new_amount - old_amount) * 100.0 / NULLIF(old_amount,0) FROM loans")
    assert "GROWTH_RATE_WITHOUT_DIVISION" in bad
    assert "GROWTH_RATE_WITHOUT_DIVISION" not in good


def test_parent_count_on_joined_child_row_grain():
    q="How many carcinogenic molecules that consisted of Nitrogen?"
    bad=flags(q,"SELECT COUNT(*) FROM atom a JOIN molecule m ON a.molecule_id = m.molecule_id WHERE a.element='N'")
    good=flags(q,"SELECT COUNT(DISTINCT m.molecule_id) FROM atom a JOIN molecule m ON a.molecule_id=m.molecule_id WHERE a.element='N'")
    assert "PARENT_COUNT_ON_JOINED_CHILD_ROWS" in bad
    assert "PARENT_COUNT_ON_JOINED_CHILD_ROWS" not in good


def test_missing_entity_value_in_sql_is_abstention_not_hallucinated_conflict():
    q="What is the label for bond TR001_2_4?"
    ret=detect_obligations(q,"SELECT bond_id FROM bond")
    assert ret["flags"]==[]


def test_not_sql_is_abstained():
    ret=detect_obligations("What?", "Error: SQL is unavailable")
    assert ret["flags"]==[]
    assert ret["abstentions"]==["NOT_A_READONLY_QUERY"]


def test_query_with_literal_id_and_correct_column_is_not_flagged():
    q="Which record has identifier 16?"
    ret=detect_obligations(q,"SELECT name FROM items WHERE id = '16'")
    assert "EXPLICIT_ID_BOUND_TO_NON_ID_ATTRIBUTE" not in ret["flags"]


def test_count_target_noun_phrase_does_not_confuse_contextual_join():
    # These are synthetic controls, NOT publisher correctness or GAIA gold.
    controls=[
        (
            "How many matches in the 2015 season were held in the Premier League?",
            "SELECT COUNT(*) FROM Match m JOIN League l ON m.league_id = l.id WHERE l.name='Premier League'",
        ),
        (
            "How many events did the member with the phone number 1234 attend?",
            "SELECT COUNT(*) FROM attendance a JOIN member m ON a.link_to_member = m.member_id WHERE m.phone='1234'",
        ),
        (
            "Among the students who attended the event, how many of them want a medium T-shirt?",
            "SELECT COUNT(*) FROM attendance a JOIN event e ON a.event_id = e.id JOIN member m ON a.member_id=m.id",
        ),
    ]
    for question,sql in controls:
        assert "PARENT_COUNT_ON_JOINED_CHILD_ROWS" not in flags(question,sql)


def test_count_target_noun_phrase_remains_sensitive_to_molecules():
    q="How many carcinogenic molecules that consisted of Nitrogen?"
    sql="SELECT COUNT(*) FROM atom a JOIN molecule m ON a.molecule_id=m.molecule_id WHERE a.element='N'"
    assert "PARENT_COUNT_ON_JOINED_CHILD_ROWS" in flags(q,sql)
