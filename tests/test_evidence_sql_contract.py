"""Gold-blind semantic evidence contract, no publisher/reference SQL required."""
from evaluation.evidence_sql_contract import (
    explicit_evidence_values,evidence_signal,evidence_division_contract,
    select_two_original_qwen_candidates,candidate_literals
)


def original_user(q: str,e: str) -> str:
    return "Question: "+q+"\nSchema context:\nPhysical SQLite schema:\npeople(name, mode)\nExternal evidence:\n"+e


def test_evidence_non_numeric_exact_quoted_assignment():
    prompt=original_user("Which items arrive fastest?",
                         "fastest shipping method refers to Ship Mode = 'First Class';")
    assert explicit_evidence_values(prompt)==("first class",)
    assert not evidence_signal(prompt,"SELECT * FROM people WHERE mode = 'First Class'").violation
    assert evidence_signal(prompt,"SELECT * FROM people WHERE mode = 'Standard Class'").violation


def test_escapes_quotes_in_sql_without_leaking_a_model_answer():
    prompt=original_user("Who is O'Sullivan?", "Customer = 'O''Sullivan';")
    assert explicit_evidence_values(prompt)==("o'sullivan",)
    assert candidate_literals("SELECT * FROM people WHERE name='O''Sullivan'") == frozenset({"o'sullivan"})
    assert not evidence_signal(prompt,"SELECT * FROM people WHERE name='O''Sullivan'").violation


def test_ignore_numeric_null_and_short_sql_literal_contracts():
    prompt=original_user("By ID?", "ID = '12'; Region = 'N'; Null = 'null';")
    assert explicit_evidence_values(prompt)==()
    assert not evidence_signal(prompt,"SELECT 1").violation


def test_sql_comment_fake_value_not_satisfy():
    prompt=original_user("Which?", "Category = 'Furniture';")
    assert evidence_signal(prompt,"SELECT * FROM people -- 'Furniture'").violation
    assert evidence_signal(prompt,"SELECT * FROM people /* 'Furniture' */").violation


def test_explicit_divide_evidence_detects_wrong_subtraction():
    prompt=original_user("How many times more?",
                         "times number of films = Divide(Count(actor_A), Count(actor_B))")
    assert evidence_division_contract(prompt,"SELECT (SELECT COUNT(*) FROM a)-(SELECT COUNT(*) FROM b)")==-1
    assert evidence_division_contract(prompt,"SELECT CAST(a AS REAL)/b FROM t")==1
    assert evidence_division_contract(original_user("Mean?","average = DIVIDE(SUM(month),COUNT(month))"),
                                      "SELECT AVG(month) FROM t")==1


def test_no_evidence_abstain():
    prompt="Question: How many items?\nSchema context:\na(id, category)"
    assert explicit_evidence_values(prompt)==()
    assert evidence_division_contract(prompt,"SELECT 1")==0
    assert not evidence_signal(prompt,"SELECT 1").covered


def test_switch_only_when_original_both_executable_and_one_satisfies():
    prompt=original_user("How many orders?", "Ship Mode = 'First Class'")
    bad="SELECT COUNT(*) FROM orders WHERE ShipMode='Standard Class'"
    good="SELECT COUNT(*) FROM orders WHERE ShipMode='First Class'"
    yes=select_two_original_qwen_candidates(prompt,bad,good,True,True)
    assert yes["switch_to_guarded"] is True
    assert not select_two_original_qwen_candidates(prompt,bad,good,True,False)["switch_to_guarded"]
    assert not select_two_original_qwen_candidates(prompt,good,bad,True,True)["switch_to_guarded"]


def test_column_values_in_schema_not_evidence_cannot_trigger():
    prompt="Question: What is the shipping mode?\nSchema context:\nfoo(First Class)\n"
    assert not evidence_signal(prompt,"SELECT foo FROM orders").covered
