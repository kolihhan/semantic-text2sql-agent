from scripts.audit_qwen9b_role96_goldblind_selector_cpu import (
    choose, projection_width, evidence_literals, suspicious_invalid,
    PROTOCOL, GATE
)


def test_hard_literal_loss_and_no_reference_labels_as_selector_inputs():
    b="SELECT COUNT(*) FROM events WHERE issue = 'Billing disputes' AND division = 'Mountain'"
    t="SELECT COUNT(*) FROM events WHERE issue = 'Billing disputes'"
    ans=choose("How many billing disputes are in the Mountain division?",
               "billing dispute refers to issue = 'Billing disputes'",b,t)
    assert ans["selected"]=="direct"
    assert "reference" not in choose.__code__.co_varnames
    assert len(evidence_literals("foo = 'Billing disputes'; c = 'MA'"))==2


def test_singular_projection_and_no_premature_template_promotion():
    b="SELECT name, amount FROM items"
    t="SELECT name FROM items"
    assert choose("What is the name of the item?", "", b,t)["selected"]=="constrained"
    assert projection_width("SELECT COUNT(DISTINCT x), SUM(price) FROM t")==2
    assert projection_width("SELECT sum(price) FROM t")==1
    assert suspicious_invalid("SELECT t.None FROM t")
    assert not suspicious_invalid("SELECT col FROM t")
    assert GATE["require_scorable"]==96
    assert PROTOCOL.startswith("ORIGINAL-QWEN9B-TRAIN96")
