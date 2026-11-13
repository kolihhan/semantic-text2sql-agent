from collections import Counter

import pytest

from evaluation.bird_generator_transfer import select_cases


def test_transfer_selection_excludes_used_databases_and_questions_and_ignores_gold():
    dev = [{"db_id": "used-db", "question": "already used"}]
    train = [
        {"db_id": db, "question": f"{db} question {i}", "SQL": "secret"}
        for db in ["used-db", "oversized", *(f"unused{i}" for i in range(20))]
        for i in range(3)
    ]
    train[-1]["question"] = "  Already   Used "
    mirror = {
        row["db_id"]: {"size": 100, "sha": "a" * 64} for row in train
    }
    mirror["oversized"]["size"] = 256 * 1024**2
    selected = select_cases(train, dev, mirror)
    assert len(selected) == 40
    assert set(Counter(row["database_id"] for row in selected).values()) == {2}
    assert all(row["database_id"].startswith("unused") for row in selected)
    assert all("SQL" not in row and "gold" not in row for row in selected)
    assert all("already" not in row["question"].casefold() for row in selected)
    assert select_cases([{**row, "SQL": "changed"} for row in train], dev, mirror) == selected
    with pytest.raises(ValueError, match="20 eligible"):
        select_cases(train, dev, {"used-db": mirror["used-db"]})


def test_database_preflight_checks_lfs_and_normal_git_blobs():
    from hashlib import sha1, sha256

    from evaluation.bird_generator_transfer import check_database_bytes

    data = b"db"
    metadata = {"size": len(data)}
    assert check_database_bytes(data, {**metadata, "lfs": {"oid": sha256(data).hexdigest()}})
    assert check_database_bytes(data, {**metadata, "oid": sha1(b"blob 2\0db").hexdigest()})
    assert not check_database_bytes(b"xx", {**metadata, "oid": sha1(b"blob 2\0db").hexdigest()})
