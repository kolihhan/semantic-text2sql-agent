"""Factual post-hoc A0/A1/A2/A3 diagnostics, no benchmark-conditioned selectors."""
from copy import deepcopy

import pytest

from experiments.analyze_semantic_abc_ablation import analyze, exact_mcnemar_p


def _artifact():
    rows=[]
    for i in range(100):
        a0=i<50
        a2=(i<50 or i==50)
        a3=(i<50 or i in {50,51,52}) and i != 3
        rows.append({
            "case_id":str(i),
            "database_id": "db-a" if i < 60 else "db-b",
            "ex":{"a0":a0,"a1":a0,"a2":a2,"a3":a3},
            "a2_reason":"compiler_repair_accepted" if i==50 else "unchanged",
            "a3_reason":"semantic_output_repair_accepted" if i in {3,51,52} else "unchanged",
            "extra_model_calls":int(i in {3,50,51,52}),
        })
    return {
        "schema":"semantic-abc-aggregate-v1",
        "sample_size":100,
        "cohort_sha256":"unseen-hash",
        "scores":{"a0":50,"a1":50,"a2":51,"a3":52},
        "cases": rows,
    }


def test_full_paired_exploratory_transitions_and_ablation_attribution():
    r=analyze(_artifact())
    assert r["scores"]=={"a0":50,"a1":50,"a2":51,"a3":52}
    assert r["paired_transitions"]["a0_to_a2"]["net_correct"]==1
    assert r["paired_transitions"]["a2_to_a3"]["wrong_to_correct"]==2
    assert r["paired_transitions"]["a2_to_a3"]["correct_to_wrong"]==1
    assert r["paired_transitions"]["a2_to_a3"]["net_correct"]==1
    assert r["accepted_semantic_outcomes"]=={
        "count":3, "wrong_to_correct":2,"correct_to_wrong":1,"both_wrong":0,"both_correct":0
    }
    assert r["accepted_compiler_repair_count"]==1
    assert r["extra_model_calls"]==4
    assert r["database_count"]==2
    assert "overlap" in " ".join(r["important_limitations"])


def test_mcnemar_is_symmetric_for_paired_wins_losses():
    assert exact_mcnemar_p(0,0)==1.0
    assert exact_mcnemar_p(1,0)==1.0
    assert exact_mcnemar_p(3,0)==0.25
    assert exact_mcnemar_p(2,1)==1.0
    assert exact_mcnemar_p(10,1)==exact_mcnemar_p(1,10)
    with pytest.raises(ValueError):
        exact_mcnemar_p(-1,1)


def test_corrupted_paired_data_fails_closed():
    o=_artifact()
    variants=[]
    v=deepcopy(o);v["scores"]["a3"]=60;variants.append(v)
    v=deepcopy(o);v["cases"][1]["ex"]["a1"]=False;variants.append(v)
    v=deepcopy(o);v["cases"][1]["ex"]["a3"]="true";variants.append(v)
    v=deepcopy(o);v["cases"][1]["case_id"]="0";variants.append(v)
    v=deepcopy(o);v["cases"][1]["a3_reason"]="unknown";variants.append(v)
    v=deepcopy(o);v["cases"].pop();variants.append(v)
    for value in variants:
        with pytest.raises(ValueError):
            analyze(value)
