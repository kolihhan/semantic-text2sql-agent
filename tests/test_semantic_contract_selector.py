"""Frozen quantifier/mean rule: pure wording+candidate SQL, no gold."""
from evaluation.semantic_contract_selector import (
    compatibility, contract_for_question, question_only, select_by_contract,
)


def o(label,valid=True):
    return {"valid":valid,"signature":label if valid else None}


def test_cardinality_must_be_outer_projected_aggregate():
    q="How many customers have an address in Abu Dhabi city?"
    assert contract_for_question(q)=="cardinality"
    assert compatibility(q,"SELECT COUNT(*) FROM customer")==1
    assert compatibility(q,"SELECT SUM(CASE WHEN female THEN 1 ELSE 0 END) FROM actor")==1
    assert compatibility(q,"SELECT customer_id FROM customer")==-1
    assert compatibility(q,"SELECT name FROM people WHERE id IN (SELECT COUNT(*) FROM t)")==-1


def test_mean_not_subquery_only_or_literal_text():
    q="What is the average price of the products?"
    assert contract_for_question(q)=="mean"
    assert compatibility(q,"SELECT AVG(price) FROM product")==1
    assert compatibility(q,"SELECT SUM(cost) * 1.0 / COUNT(*) FROM product")==1
    assert compatibility(q,"SELECT avg_house_value FROM zip_data")==1
    assert compatibility(q,"SELECT product_id FROM product")==-1
    assert compatibility(q,"SELECT name FROM p WHERE EXISTS (SELECT AVG(cost) FROM t)")==-1
    assert compatibility(q,"SELECT 'AVG(price)' FROM p")==-1


def test_question_scope_never_includes_schema_tokens():
    user="Question: Which customers are from Berlin?\nSchema context:\npeople(avg_age, count, city)"
    assert question_only(user)=="Which customers are from Berlin?"
    assert compatibility(question_only(user),"SELECT avg_age FROM people")==0


def test_multi_output_question_abstains_instead_of_blindly_forcing_count():
    q="How many customers have an address? List those customer names."
    assert contract_for_question(q)=="abstain"
    assert compatibility(q,"SELECT name FROM customer")==0
    assert contract_for_question("What is the highest average city house value?")=="abstain"


def test_goldblind_validity_gated_contract_override():
    q="How many items are there?"
    sql=["SELECT id FROM a","SELECT id FROM a","SELECT COUNT(*) FROM a"]
    out=[o("wrong"),o("wrong"),o("correct")]
    result=select_by_contract(q,sql,out)
    assert result["original_index"]==0 and result["index"]==2
    assert result["contract"]=="cardinality" and result["changed"]
    # DO NOT need publisher SQL or correctness labels to make the choice.


def test_never_choose_nonexecuting_candidate_and_abstain_on_no_safe_alternative():
    q="How many items are there?"
    result=select_by_contract(q,["SELECT id FROM a","SELECT COUNT(*) FROM a"],
                              [o("wrong"),o("broken",False)])
    assert not result["changed"] and result["index"]==0


def test_no_contract_no_change_and_already_compliant_no_change():
    result=select_by_contract("Which items have IDs?",["SELECT id FROM a","SELECT COUNT(*) FROM a"],
                              [o("same"),o("other")])
    assert not result["changed"]
    result=select_by_contract("How many items?",["SELECT COUNT(*) FROM a","SELECT id FROM a"],
                              [o("count"),o("id")])
    assert result["index"]==0 and not result["changed"]


def test_no_source_sql_validation_data_required_or_allowed():
    try:
        select_by_contract("How many items?",["select count(*)"],[])
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched source candidates must fail")
