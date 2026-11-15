from pathlib import Path
import sqlite3
import pytest

from semantic_sql.catalog import DatabaseCatalog
from evaluation.spider_published_fk import published_foreign_keys, enrichment


def sample(tmp_path):
    db = tmp_path / "source.sqlite"
    with sqlite3.connect(db) as c:
        c.executescript(
            "CREATE TABLE Parent(pid INTEGER PRIMARY KEY, title TEXT);"
            "CREATE TABLE Child(cid INTEGER, owner INTEGER, note TEXT);"
            "CREATE TABLE Decoy(did INTEGER, title TEXT);"
        )
    published = {
        "table_names_original": ["Parent","Child","Decoy"],
        "column_names_original": [[-1,"*"],[0,"pid"],[1,"owner"],[2,"did"]],
        "foreign_keys": [[2,1]],
    }
    return db,published


def test_valid_source_annotation_adds_only_physically_existing_fk(tmp_path):
    db,entry=sample(tmp_path)
    cat=DatabaseCatalog.from_sqlite(db)
    assert not cat.foreign_keys
    assert published_foreign_keys(cat,entry) == (("Child","owner","Parent","pid"),)
    old,new,diag=enrichment(cat,entry,tmp_path,"source","Which child has the parent?")
    assert new != old
    assert "FK Child.owner -> Parent.pid" in new
    assert diag["visible_new_fks"]==1
    assert diag["missing_physical_fk"]==1


def test_invalid_claim_not_sent_as_fk(tmp_path):
    db,entry=sample(tmp_path)
    entry["column_names_original"][2][1]="imagined_column"
    cat=DatabaseCatalog.from_sqlite(db)
    assert published_foreign_keys(cat,entry)==()
    before,after,diag=enrichment(cat,entry,tmp_path,"source","Which child has the parent?")
    assert before==after
    assert diag["missing_physical_fk"]==0


def test_out_of_range_and_invalid_table_indices_rejected(tmp_path):
    db,entry=sample(tmp_path)
    entry["foreign_keys"]=[[99,1]]
    with pytest.raises(ValueError,match="out of bounds"):
        published_foreign_keys(DatabaseCatalog.from_sqlite(db),entry)


def test_does_not_change_correctly_declared_existing_fk(tmp_path):
    db,entry=sample(tmp_path)
    cat=DatabaseCatalog.from_sqlite(db)
    cat.foreign_keys=(("Child","owner","Parent","pid"),)
    old,new,diag=enrichment(cat,entry,tmp_path,"source","Which child has the parent?")
    assert old==new
    assert diag["missing_physical_fk"]==0
