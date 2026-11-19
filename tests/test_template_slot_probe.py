import copy
import hashlib

import pytest

from evaluation.template_slot_probe import PROTOCOL, seal_choices, score_sealed


def candidate(i, sig):
    return {"id": i, "valid": sig is not None, "signature": sig}


def sample(template="new"):
    return {
        "protocol": PROTOCOL, "manifest_sha256": "frozen-source", "model_digest": "fixed-qwen9b",
        "cases": [{"case_hash": hashlib.sha256(b"case-1").hexdigest(), "db_id": "db1",
                   "baseline": [candidate("c1", "a"), candidate("c2", "b"),
                                candidate("c3", "c"), candidate("c4", "a")],
                   "template": None if template is None else candidate("template", template)}],
    }


def refs(source, gold):
    return {"manifest_sha256": source["manifest_sha256"],
            "cases": [{"case_hash": source["cases"][0]["case_hash"],
                       "scorable": gold is not None, "set_hash": gold}]}


def test_tie_first_template_correct_and_position_control_not_correct():
    source = sample("new")
    sealed = seal_choices(source)
    assert {a: sealed["cases"][0]["choices"][a]["chosen_id"] for a in ("A", "B", "C")} == {
        "A": "c1", "B": "template", "C": "c4"}
    score = score_sealed(sealed, refs(source, "new"))
    assert score["final_correct"] == {"A": 0, "B": 1, "C": 0}
    assert score["oracle_at_least_one_correct"] == {"A": 0, "B": 1}
    assert (score["B_vs_A_rescues"], score["B_vs_A_regressions"]) == (1, 0)


def test_majority_wrong_can_hide_correct_template():
    source = sample("correct")
    for x in source["cases"][0]["baseline"]:
        x["signature"] = "wrong"
    sealed = seal_choices(source)
    assert sealed["cases"][0]["choices"]["B"]["chosen_id"] == "c1"
    score = score_sealed(sealed, refs(source, "correct"))
    assert score["oracle_at_least_one_correct"]["B"] == 1
    assert score["final_correct"]["B"] == 0


def test_no_template_fallback_is_exact_original():
    source = sample(None)
    row = seal_choices(source)["cases"][0]
    assert row["choices"]["A"] == row["choices"]["B"] == row["choices"]["C"]
    assert row["arm_candidates"]["A"] == row["arm_candidates"]["B"]


def test_ref_leakage_and_post_seal_tampering_blocked():
    source = sample()
    source["cases"][0]["baseline"][0]["gold_correct"] = True
    with pytest.raises(ValueError):
        seal_choices(source)
    source = sample()
    sealed = seal_choices(source)
    sealed["cases"][0]["choices"]["B"]["signature"] = "hacked"
    with pytest.raises(ValueError, match="changed"):
        score_sealed(sealed, refs(source, "new"))


def test_incomplete_reference_rejected_and_unscorable_explicit():
    source = sample()
    sealed = seal_choices(source)
    with pytest.raises(ValueError):
        score_sealed(sealed, {"manifest_sha256": "other", "cases": []})
    score = score_sealed(sealed, refs(source, None))
    assert score["n_scorable"] == 0 and score["n_unscorable"] == 1


def test_tied_earliest_valid_and_all_invalid():
    source = sample()
    for x in source["cases"][0]["baseline"]:
        x["valid"] = False
        x["signature"] = None
    source["cases"][0]["template"] = candidate("template", None)
    row = seal_choices(source)["cases"][0]
    assert all(row["choices"][arm]["signature"] is None for arm in ("A", "B", "C"))
    assert row["choices"]["B"]["chosen_id"] == "template"
