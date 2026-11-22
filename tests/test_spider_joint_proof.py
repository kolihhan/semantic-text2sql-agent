"""Physical Spider proof runner: tests prevent answer leakage and score inflation."""
import json
import sqlite3
from pathlib import Path
import zipfile

from evaluation.prove_spider_joint import freeze_source, strict_execution_compare


def _fixture_archive(path: Path):
    rows=[]
    for i in range(14):
        rows.append({"db_id":f"db_{i}", "question":f"How many orders with a customer in group {i}?", "query":"SELECT 1"})
    with zipfile.ZipFile(path,"w") as z:
        z.writestr("spider_data/dev.json",json.dumps(rows))


def test_freeze_selects_12_distinct_db_without_gold(tmp_path):
    archive=tmp_path/"spider_data.zip"
    _fixture_archive(archive)
    f=freeze_source(archive)
    assert f["sample_size"]==12
    assert len({c["db_id"] for c in f["cases"]})==12
    assert all("query" not in c and "SQL" not in c for c in f["cases"])
    assert all("question" in c and "source_index" in c for c in f["cases"])
    assert f["gold_exposed_to_inference"] is False


def test_physical_set_bag_scoring_and_duplicate_multiplicity(tmp_path):
    db=tmp_path/"test.sqlite"
    con=sqlite3.connect(db)
    con.executescript("CREATE TABLE t(x INT); INSERT INTO t VALUES (1),(1),(2);")
    con.close()
    full="SELECT x FROM t"
    only_distinct="SELECT DISTINCT x FROM t"
    assert strict_execution_compare(db, full, full)==(True,True)
    assert strict_execution_compare(db, only_distinct, full)==(True,False)
    assert strict_execution_compare(db, "SELECT x FROM t WHERE 1=0", full)==(False,False)


def test_invalid_candidate_counts_as_wrong_not_removed(tmp_path):
    db=tmp_path/"test.sqlite"
    sqlite3.connect(db).execute("CREATE TABLE t(x INT)").connection.close()
    assert strict_execution_compare(db,"SELECT FROM t","SELECT x FROM t")==(False,False)
