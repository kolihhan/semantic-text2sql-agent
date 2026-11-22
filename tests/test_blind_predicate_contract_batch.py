import hashlib
import sqlite3
import pytest
from scripts.blind_predicate_contract_batch import diagnose, sha256

def test_blind_batch_digest_and_sealed_output(tmp_path):
    db = tmp_path / "fixture.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE people(city TEXT)")
        conn.execute("INSERT INTO people VALUES ('Paris')")
    row = {"case_id": "a", "question": 'Find "Paris"',
           "selected_sql": "SELECT city FROM people WHERE city='Paris'",
           "database": str(db), "database_sha256": sha256(db)}
    result = diagnose([row], tmp_path)
    assert result["count"] == 1
    assert result["database_count"] == 1
    assert result["status_counts"] == {"OBSERVED": 1}
    assert result["reason_counts"] == {"EXACT_VALUE_PRESENT": 1}
    abstain = {**row, "case_id": "b", "selected_sql": "SELECT city FROM people"}
    mixed = diagnose([row, abstain], tmp_path)
    assert mixed["count"] == 2 and mixed["database_count"] == 1
    assert mixed["status_counts"] == {"ABSTAIN": 1, "OBSERVED": 1}
    assert mixed["reason_counts"] == {"COMPOUND_OR_MISSING_PREDICATE": 1, "EXACT_VALUE_PRESENT": 1}
    assert result["cases"][0]["status"] == "OBSERVED"
    assert "Paris" not in str(result)
    assert str(db) not in str(result)
    with pytest.raises(ValueError, match="Duplicate"):
        diagnose([row, row], tmp_path)
    with pytest.raises(ValueError, match="Unexpected"):
        diagnose([{**row, "gold_sql": "SELECT 1"}], tmp_path)
    with pytest.raises(ValueError, match="digest"):
        diagnose([{**row, "database_sha256": "0" * 64}], tmp_path)
    with pytest.raises(ValueError, match="outside"):
        diagnose([row], tmp_path / "not-allowed")
    protected = tmp_path / "simpson_episodes.db"
    protected.write_bytes(db.read_bytes())
    with pytest.raises(ValueError, match="Protected"):
        diagnose([{**row, "database": str(protected), "database_sha256": sha256(protected)}], tmp_path)
