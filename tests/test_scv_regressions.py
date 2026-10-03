from __future__ import annotations

import sqlite3
from pathlib import Path

from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.contract_verifier import verify_semantic_contract
from semantic_sql.semantic_contract import AggregationSpec, RankingSpec, SemanticContract
from semantic_sql.sql_semantics import parse_sql_semantics


def _catalog(tmp_path: Path) -> DatabaseCatalog:
    db = tmp_path / "regressions.sqlite"
    with sqlite3.connect(db) as con:
        con.executescript(
            """
            PRAGMA foreign_keys=ON;
            CREATE TABLE molecule (
                molecule_id TEXT PRIMARY KEY,
                carcinogenic INTEGER
            );
            CREATE TABLE atom (
                atom_id INTEGER PRIMARY KEY,
                molecule_id TEXT,
                element TEXT,
                FOREIGN KEY (molecule_id) REFERENCES molecule(molecule_id)
            );
            CREATE TABLE constructors (
                constructorId INTEGER PRIMARY KEY,
                name TEXT
            );
            CREATE TABLE races (
                raceId INTEGER PRIMARY KEY,
                name TEXT
            );
            CREATE TABLE results (
                resultId INTEGER PRIMARY KEY,
                raceId INTEGER,
                constructorId INTEGER,
                points REAL,
                FOREIGN KEY (raceId) REFERENCES races(raceId),
                FOREIGN KEY (constructorId) REFERENCES constructors(constructorId)
            );
            """
        )
    return DatabaseCatalog.from_sqlite(db)


def _semantics(sql: str):
    parsed = parse_sql_semantics(sql)
    assert parsed.status == "ok" and parsed.semantics is not None
    return parsed.semantics


def _codes(result) -> set[str]:
    return {v.code for v in result.violations}


def test_entity_grain_count_requires_distinct(tmp_path: Path) -> None:
    contract = SemanticContract(
        confidence="high",
        aggregation=AggregationSpec(
            function="COUNT",
            target="atom.molecule_id",
            entity_grain="molecule",
            distinct=True,
        ),
    )
    result = verify_semantic_contract(
        contract,
        _semantics(
            "SELECT COUNT(a.molecule_id) FROM atom a "
            "JOIN molecule m ON a.molecule_id = m.molecule_id "
            "WHERE a.element = 'N' AND m.carcinogenic = 1"
        ),
        _catalog(tmp_path),
    )
    assert "DISTINCT_REQUIRED" in _codes(result)


def test_projection_failure_is_structurally_visible(tmp_path: Path) -> None:
    contract = SemanticContract(confidence="high", projection=("atom.atom_id",))
    result = verify_semantic_contract(
        contract,
        _semantics("SELECT element FROM atom WHERE molecule_id = 'TR186'"),
        _catalog(tmp_path),
    )
    assert "PROJECTION_MISMATCH" in _codes(result)


def test_wrong_join_key_is_rejected_but_fk_join_passes(tmp_path: Path) -> None:
    contract = SemanticContract(
        confidence="high",
        projection=("constructors.name",),
        join_entities=("constructors", "results"),
    )
    catalog = _catalog(tmp_path)
    wrong = verify_semantic_contract(
        contract,
        _semantics(
            "SELECT c.name FROM constructors c "
            "JOIN results r ON c.constructorId = r.raceId"
        ),
        catalog,
    )
    correct = verify_semantic_contract(
        contract,
        _semantics(
            "SELECT c.name FROM constructors c "
            "JOIN results r ON c.constructorId = r.constructorId"
        ),
        catalog,
    )
    assert "JOIN_EDGE_INVALID" in _codes(wrong)
    assert correct.status == "pass"


def test_row_level_top1_is_not_aggregate_top1(tmp_path: Path) -> None:
    contract = SemanticContract(
        confidence="high",
        projection=("constructors.name",),
        aggregation=AggregationSpec(function="SUM", target="results.points"),
        group_by=("constructors.name",),
        ranking=RankingSpec(
            metric="SUM(results.points)",
            direction="DESC",
            top_k=1,
            ties="single",
        ),
        join_entities=("constructors", "results"),
    )
    wrong = verify_semantic_contract(
        contract,
        _semantics(
            "SELECT c.name FROM constructors c "
            "JOIN results r ON c.constructorId = r.constructorId "
            "ORDER BY r.points DESC LIMIT 1"
        ),
        _catalog(tmp_path),
    )
    codes = _codes(wrong)
    assert "AGGREGATION_MISSING" in codes
    assert "GROUP_BY_MISSING" in codes
    assert "ORDER_METRIC_MISMATCH" in codes


def test_explicit_all_ties_rejects_limit_one(tmp_path: Path) -> None:
    contract = SemanticContract(
        confidence="high",
        ranking=RankingSpec(
            metric=None,
            direction=None,
            top_k=1,
            ties="all",
        ),
    )
    result = verify_semantic_contract(
        contract,
        _semantics("SELECT name FROM constructors ORDER BY constructorId DESC LIMIT 1"),
        _catalog(tmp_path),
    )
    assert "TOPK_MISMATCH" in _codes(result)


def test_unspecified_ties_do_not_create_extra_violation(tmp_path: Path) -> None:
    contract = SemanticContract(
        confidence="high",
        ranking=RankingSpec(
            metric=None,
            direction=None,
            top_k=1,
            ties="unspecified",
        ),
    )
    result = verify_semantic_contract(
        contract,
        _semantics("SELECT name FROM constructors ORDER BY constructorId DESC LIMIT 1"),
        _catalog(tmp_path),
    )
    assert "TOPK_MISMATCH" not in _codes(result)


def test_correct_structural_queries_remain_unmodified_candidates(tmp_path: Path) -> None:
    contract = SemanticContract(
        confidence="high",
        projection=("constructors.name",),
        aggregation=AggregationSpec(function="SUM", target="results.points"),
        group_by=("constructors.name",),
        ranking=RankingSpec(
            metric="SUM(results.points)",
            direction="DESC",
            top_k=1,
            ties="single",
        ),
        join_entities=("constructors", "results"),
    )
    result = verify_semantic_contract(
        contract,
        _semantics(
            "SELECT c.name, SUM(r.points) FROM constructors c "
            "JOIN results r ON c.constructorId = r.constructorId "
            "GROUP BY c.name ORDER BY SUM(r.points) DESC LIMIT 1"
        ),
        _catalog(tmp_path),
    )
    assert result.status == "pass"
    assert result.violations == ()
