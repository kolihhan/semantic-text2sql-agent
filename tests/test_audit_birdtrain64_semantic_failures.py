from __future__ import annotations

from collections import Counter
import pytest

from scripts.audit_birdtrain64_semantic_failures import TAXON,LABEL_WARNINGS,audit


def test_posthoc_taxonomy_has_23_unique_executable_wrong_cases():
    indices=[v for vals in TAXON.values() for v in vals]
    assert len(indices)==23 and len(set(indices))==23
    assert Counter({k:len(v) for k,v in TAXON.items()})==Counter({
        "suspected_reference_question_mismatch":2,
        "projection_or_result_shape":8,
        "table_join_filter_or_identifier_grounding":8,
        "aggregation_ratio_or_topk_semantics":5,
    })


def test_reference_question_mismatch_warnings_are_not_auto_gold_corrections():
    assert set(LABEL_WARNINGS)=={6969,5867}
    assert all(isinstance(s,str) and len(s)>40 for s in LABEL_WARNINGS.values())


def test_audit_fails_closed_if_original_source_sha_missing():
    with pytest.raises(ValueError,match="SHA mismatch"):
        audit({},b"tampered")
