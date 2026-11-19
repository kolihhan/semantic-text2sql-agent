import json
from experiments.template_binding_pilot import mask_template, bind, operations, intent


def test_structural_rebinding_substitutes_identifiers_and_lit_literal():
    source = "Question: How many users from country Italy?\nSchema context:\nPhysical SQLite schema:\nusers(id, first_name, city)\n"
    sql = "SELECT COUNT(users.id) FROM users WHERE users.city = 'Rome'"
    skeleton,slots = mask_template(sql, source)
    assert "users" not in skeleton and "'Rome'" not in skeleton
    assert {"table","column","value_string"} <= {x["kind"] for x in slots.values()}
    mapping = {key: ("customers" if v["kind"]=="table" else "customer_id" if
                      v["source"]=="id" else "city" if v["kind"]=="column" else "Dublin")
               for key,v in slots.items()}
    target = "Question: How many customers from Dublin?\nSchema context:\nPhysical SQLite schema:\ncustomers(customer_id, city)\n"
    out = bind(skeleton,slots,json.dumps(mapping),target)
    assert '"customers"' in out
    assert "Dublin" in out
    assert "__" not in out


def test_invalid_target_identifier_fails_closed():
    source = "Physical SQLite schema:\npeople(id, city)"
    skeleton,slots = mask_template("SELECT id FROM people WHERE city='Rome'",source)
    answers = {p: "missing_table" if m["kind"]=="table" else
               "id" if m["kind"]=="column" else "Rome" for p,m in slots.items()}
    target = "Physical SQLite schema:\nactual(id, city)"
    assert bind(skeleton,slots,json.dumps(answers),target) == ""


def test_query_operation_intents_are_gold_independent():
    assert "count" in intent("How many employees were hired?")
    assert "/" in intent("What is the percentage of active customers?")
    assert "group by" in operations("SELECT x, count(*) FROM t GROUP BY x")
