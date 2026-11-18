"""Tests for source-locked BIRD TRAIN-only semantic contract feasibility."""
from hashlib import sha256

import pytest

from scripts.probe_semantic_contract_train import (
    train_reference_safety, independent_real_model_validation,
)


def _row(i:int,db:str,question:str,sql:str)->dict:
    user=f"Question: {question}\nSchema context:\nfoo(id, price)"
    return {
        "source_index":i,"source_db_id":db,
        "input_sha256":sha256(user.encode()).hexdigest(),
        "messages":[{"role":"system","content":"source-only"},
                    {"role":"user","content":user},
                    {"role":"assistant","content":sql}]
    }


def test_train_reference_counts_and_source_safety():
    rows=[_row(i,"train"+str(i%16),"How many entries are there?",
               "SELECT COUNT(*) FROM foo") for i in range(242)]
    stats=train_reference_safety(rows)
    assert stats["n"]==242 and stats["dbs"]==16
    assert stats["category_cases"]=={"cardinality":242}
    assert not stats["publisher_reference_flags"]


def test_same_train_val_database_fails_closed():
    val=[_row(i,"val"+str(i%4),"How many entries?", "SELECT COUNT(*) FROM foo")
         for i in range(64)]
    with pytest.raises(ValueError,match="overlap"):
        independent_real_model_validation(val,{},{"val0"})


def test_exact_original_real_model_source_and_labels_are_required():
    val=[_row(i,"val"+str(i%4),"How many entries?", "SELECT COUNT(*) FROM foo")
         for i in range(64)]
    with pytest.raises(ValueError,match="source drift"):
        independent_real_model_validation(val,{"n":63}, {"train0"})


def test_malformed_original_training_references_rejected():
    rows=[_row(i,"train0","How many?", "SELECT COUNT(*)") for i in range(242)]
    with pytest.raises(ValueError,match="16 databases"):
        train_reference_safety(rows)
