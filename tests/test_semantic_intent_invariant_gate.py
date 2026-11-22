"""Synthetic, gold-free adversarial controls for semantic candidate gate."""
import pytest
from scripts.semantic_intent_invariant_gate import (
    candidate_violations, numeric_id_binding_conflict, freeze_decisions,
)

def test_id_value_on_name_column_is_flagged():
    assert numeric_id_binding_conflict(
        'Which artist has id "16"? Provide the tag genre.',
        "SELECT t.tag FROM tags t JOIN torrents r ON t.id=r.id WHERE r.artist LIKE '16%'"
    )
    assert not numeric_id_binding_conflict(
        'Which artist has id "16"?', "SELECT artist FROM torrents WHERE id=16"
    )
    assert not numeric_id_binding_conflict(
        'Find 16 artists by name', "SELECT artist FROM torrents WHERE artist='16'"
    )

def test_wrong_entity_substitution_is_blocked():
    reasons=candidate_violations(
        'What is the language of "Moon Valley"?',
        "SELECT language FROM film WHERE title='Moon Valley'",
        "SELECT language FROM film WHERE title='Winter Night'",
    )
    assert "QUESTION_NAMED_ENTITY_REPLACED" in reasons
    assert "ORIGINAL_NAMED_LITERAL_REPLACED" in reasons

def test_safe_case_only_spelling_remains_eligible_not_claimed_correct():
    assert candidate_violations(
        "Who is rapper '2Pac'?",
        "SELECT name FROM artist WHERE name='2Pac'",
        "SELECT name FROM artist WHERE name='2pac'",
    )==[]

def test_malformed_semantic_broadening_is_blocked_for_exact_id():
    assert "EXACT_FILTER_BROADENED_TO_LIKE" in candidate_violations(
        "Find record with ID 16",
        "SELECT name FROM records WHERE name='16'",
        "SELECT name FROM records WHERE name LIKE '16%'",
    )

def test_preserve_aggregation():
    assert "AGGREGATION_REMOVED" in candidate_violations(
        "How many items?",
        "SELECT COUNT(*) FROM items",
        "SELECT name FROM items",
    )

def test_do_not_assume_case_only_changes_are_semantically_incorrect():
    # Without publisher SQL/independent annotations, uppercase could be right.
    assert candidate_violations(
        "Find customer Lillie Kim",
        "SELECT * FROM customer WHERE first_name='Lillie' AND last_name='Kim'",
        "SELECT * FROM customer WHERE first_name='LILLIE' AND last_name='KIM'",
    ) == []

def test_gold_outcome_field_prohibited_before_gate():
    rows=[{"source_index":i,"db_id":"db","question":"ID 16","sql":"SELECT * FROM a",
           "prompt_sha256":"x"} for i in range(64)]
    predictions=[{"source_index":i,"db_id":"db","status":"ORIGINAL_NONEMPTY"} for i in range(64)]
    with pytest.raises(ValueError,match="gold/outcome"):
        freeze_decisions({"cases":[dict(row, strict_ex=True) for row in rows]},
                         {"cases":predictions})
    result=freeze_decisions({"cases":rows},{"cases":predictions})
    assert result["no_gold_accessed"] is True
    assert all(r["action"]=="UNCHANGED" for r in result["decisions"])
