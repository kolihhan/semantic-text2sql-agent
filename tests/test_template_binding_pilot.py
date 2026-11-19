import json
from experiments.template_binding_pilot import mask_template, bind, operations, intent, template_shape_compatible


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


def test_source_only_stoploss_rejects_four_actual_pilot_mismatches():
    pairs = [
        ("What are the addresses of the inactive customers?",
         "SELECT COUNT(*) FROM customer WHERE cust_type = 'manufacturer'"),
        ("An American rapper released an album in 1991, how many years have passed until the next?",
         "SELECT T1.playerID FROM players AS T1 JOIN awards_players AS T2 ON T1.playerID=T2.playerID GROUP BY T1.playerID ORDER BY COUNT(T2.award) DESC LIMIT 1"),
        ("What is the percentage of furniture orders shipped through first class?",
         "SELECT T1.Name FROM country T1 JOIN religion T2 ON T1.Code=T2.country ORDER BY T1.Population/T1.Area ASC LIMIT 1"),
        ("What is the life expectancy of the people living in Calama city?",
         "SELECT COUNT(DISTINCT T2.patient) FROM conditions T1 JOIN patients T2 ON T1.patient=T2.patient"),
    ]
    assert all(not template_shape_compatible(q,sql) for q,sql in pairs)


def test_table_column_name_ambiguity_after_qualifier():
    schema="Physical SQLite schema:\ncountry(country, area)\nreligion(country, name)"
    skeleton,meta=mask_template(
        "SELECT T2.country FROM religion AS T2 JOIN country AS T1 ON T2.country=T1.country",schema)
    assert "T2.__C" in skeleton
    assert all("__T" not in token for token,v in meta.items() if v["kind"]=="column")
