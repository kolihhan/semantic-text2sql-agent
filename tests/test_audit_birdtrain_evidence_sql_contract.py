"""Strict SHA/model/source and independent-DB gates for evidence contracts."""
from hashlib import sha256
import pytest

from scripts.audit_birdtrain_evidence_sql_contract import (
    train_reference_screen, independent_original_model64
)


def _row(i:int,db:str,prompt:str,ref:str)->dict:
    return {
        "source_index":i,"source_db_id":db,"input_sha256":sha256(prompt.encode()).hexdigest(),
        "messages":[{"role":"system","content":"immutable"},
                    {"role":"user","content":prompt},
                    {"role":"assistant","content":ref}]
    }


def _prompt(evidence:str)->str:
    return "Question: How many items?\nSchema context:\nitems(id,value)\nExternal evidence:\n"+evidence


def test_train242_reference_evidence_risk_is_not_accuracy():
    rows=[_row(i,f"train{i%16}",_prompt("Item kind = 'First Class'"),
               "SELECT COUNT(*) FROM items WHERE kind='First Class'") for i in range(242)]
    result=train_reference_screen(rows)
    assert result["n"]==242 and result["physical_dbs"]==16
    assert result["coverage"]["reference_contract_covered"]==242
    assert result["reference_violations"]==0


def test_train242_detects_missing_publisher_evidence_value():
    rows=[_row(i,f"train{i%16}",_prompt("Item kind = 'First Class'"),
               "SELECT COUNT(*) FROM items WHERE kind='First Class'") for i in range(242)]
    rows[0]["messages"][2]["content"]="SELECT COUNT(*) FROM items WHERE kind='Standard Class'"
    result=train_reference_screen(rows)
    assert result["reference_violations"]==1
    assert result["reference_violation_source_indices"]==[0]


def test_validate_physical_db_disjointness_before_model_score():
    rows=[_row(i,f"val{i%4}",_prompt("Item kind = 'First Class'"),
               "SELECT 1") for i in range(64)]
    with pytest.raises(ValueError,match="overlap"):
        independent_original_model64(rows,{},{"val0"})


def test_validate_exact_frozen_qwen_model_digest():
    rows=[_row(i,f"val{i%4}",_prompt("Item kind = 'First Class'"),
               "SELECT 1") for i in range(64)]
    with pytest.raises(ValueError,match="identity"):
        independent_original_model64(rows,{"n":64,"model_digest":"switched"},set())
