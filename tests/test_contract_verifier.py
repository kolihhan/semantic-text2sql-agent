from __future__ import annotations

import sqlite3
from pathlib import Path

from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.contract_verifier import verify_semantic_contract
from semantic_sql.schema_graph import SchemaGraph
from semantic_sql.semantic_contract import (
    AggregationSpec,
    FilterSpec,
    RankingSpec,
    SemanticContract,
)
from semantic_sql.sql_semantics import parse_sql_semantics


def _catalog(tmp_path: Path) -> DatabaseCatalog:
    db = tmp_path / "scv.sqlite"
    with sqlite3.connect(db) as con:
        con.executescript(
            """
            PRAGMA foreign_keys=ON;
            CREATE TABLE constructors (
                constructorId INTEGER PRIMARY KEY,
                name TEXT
            );
            CREATE TABLE races (
                raceId INTEGER PRIMARY KEY,
                name TEXT,
                year INTEGER
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
    result = parse_sql_semantics(sql)
    assert result.status == "ok" and result.semantics is not None
    return result.semantics


def _codes(result) -> set[str]:
    return {v.code for v in result.violations}


def test_schema_graph_accepts_declared_fk_edge(tmp_path: Path) -> None:
    graph = SchemaGraph.from_catalog(_catalog(tmp_path))
    assert graph.supports_fk_equality("results", "constructorId", "constructors", "constructorId")


def test_schema_graph_accepts_reverse_equality_form(tmp_path: Path) -> None:
    graph = SchemaGraph.from_catalog(_catalog(tmp_path))
    assert graph.supports_fk_equality("constructors", "constructorId", "results", "constructorId")


def test_schema_graph_rejects_unrelated_identifier_join(tmp_path: Path) -> None:
    graph = SchemaGraph.from_catalog(_catalog(tmp_path))
    assert not graph.supports_fk_equality("races", "raceId", "results", "constructorId")


def test_projection_mismatch_requires_resolved_projection(tmp_path: Path) -> None:
    contract = SemanticContract(confidence="high", projection=("constructors.name",))
    result = verify_semantic_contract(
        contract,
        _semantics("SELECT c.constructorId FROM constructors c"),
        _catalog(tmp_path),
    )
    assert "PROJECTION_MISMATCH" in _codes(result)
    violation = result.violations[0]
    assert violation.expected and violation.actual and violation.evidence


def test_unresolved_projection_abstains_from_projection_check(tmp_path: Path) -> None:
    contract = SemanticContract(confidence="high", projection=("constructor",))
    result = verify_semantic_contract(
        contract,
        _semantics("SELECT c.constructorId FROM constructors c"),
        _catalog(tmp_path),
    )
    assert "PROJECTION_MISMATCH" not in _codes(result)


def test_aggregation_missing_detected(tmp_path: Path) -> None:
    contract = SemanticContract(
        confidence="high",
        aggregation=AggregationSpec(function="SUM", target="results.points"),
    )
    result = verify_semantic_contract(
        contract,
        _semantics("SELECT r.points FROM results r"),
        _catalog(tmp_path),
    )
    assert "AGGREGATION_MISSING" in _codes(result)


def test_aggregation_function_mismatch_detected(tmp_path: Path) -> None:
    contract = SemanticContract(
        confidence="high",
        aggregation=AggregationSpec(function="SUM", target="results.points"),
    )
    result = verify_semantic_contract(
        contract,
        _semantics("SELECT MAX(r.points) FROM results r"),
        _catalog(tmp_path),
    )
    assert "AGGREGATION_FUNCTION_MISMATCH" in _codes(result)


def test_distinct_required_for_explicit_entity_grain_count(tmp_path: Path) -> None:
    contract = SemanticContract(
        confidence="high",
        aggregation=AggregationSpec(
            function="COUNT",
            target="results.constructorId",
            entity_grain="constructor",
            distinct=True,
        ),
    )
    result = verify_semantic_contract(
        contract,
        _semantics("SELECT COUNT(r.constructorId) FROM results r"),
        _catalog(tmp_path),
    )
    assert "DISTINCT_REQUIRED" in _codes(result)


def test_group_by_missing_detected(tmp_path: Path) -> None:
    contract = SemanticContract(
        confidence="high",
        aggregation=AggregationSpec(function="SUM", target="results.points"),
        group_by=("constructors.name",),
    )
    result = verify_semantic_contract(
        contract,
        _semantics(
            "SELECT SUM(r.points) FROM results r "
            "JOIN constructors c ON r.constructorId = c.constructorId"
        ),
        _catalog(tmp_path),
    )
    assert "GROUP_BY_MISSING" in _codes(result)


def test_group_by_mismatch_detected(tmp_path: Path) -> None:
    contract = SemanticContract(
        confidence="high",
        group_by=("constructors.name",),
    )
    result = verify_semantic_contract(
        contract,
        _semantics(
            "SELECT c.constructorId, SUM(r.points) FROM results r "
            "JOIN constructors c ON r.constructorId = c.constructorId "
            "GROUP BY c.constructorId"
        ),
        _catalog(tmp_path),
    )
    assert "GROUP_BY_MISMATCH" in _codes(result)


def test_order_metric_direction_and_topk_mismatches_detected(tmp_path: Path) -> None:
    contract = SemanticContract(
        confidence="high",
        ranking=RankingSpec(
            metric="SUM(results.points)",
            direction="DESC",
            top_k=1,
            ties="single",
        ),
    )
    result = verify_semantic_contract(
        contract,
        _semantics("SELECT points FROM results ORDER BY points ASC LIMIT 3"),
        _catalog(tmp_path),
    )
    codes = _codes(result)
    assert {"ORDER_METRIC_MISMATCH", "ORDER_DIRECTION_MISMATCH", "TOPK_MISMATCH"} <= codes


def test_invalid_explicit_join_edge_detected(tmp_path: Path) -> None:
    contract = SemanticContract(
        confidence="high",
        join_entities=("races", "results"),
    )
    result = verify_semantic_contract(
        contract,
        _semantics(
            "SELECT races.name FROM races "
            "JOIN results ON races.raceId = results.constructorId"
        ),
        _catalog(tmp_path),
    )
    assert "JOIN_EDGE_INVALID" in _codes(result)


def test_valid_fk_join_with_aliases_passes(tmp_path: Path) -> None:
    contract = SemanticContract(
        confidence="high",
        projection=("constructors.name",),
        join_entities=("constructors", "results"),
    )
    result = verify_semantic_contract(
        contract,
        _semantics(
            "SELECT c.name FROM results r "
            "JOIN constructors c ON r.constructorId = c.constructorId"
        ),
        _catalog(tmp_path),
    )
    assert result.status == "pass"
    assert result.violations == ()


def test_missing_high_confidence_literal_detected(tmp_path: Path) -> None:
    contract = SemanticContract(
        confidence="high",
        filters=(FilterSpec(field="races.name", op="=", value="Monaco Grand Prix"),),
    )
    result = verify_semantic_contract(
        contract,
        _semantics("SELECT raceId FROM races WHERE name = 'British Grand Prix'"),
        _catalog(tmp_path),
    )
    assert "FILTER_LITERAL_MISSING" in _codes(result)


def test_correct_aggregate_group_rank_passes(tmp_path: Path) -> None:
    contract = SemanticContract(
        confidence="high",
        projection=("constructors.name",),
        aggregation=AggregationSpec(function="SUM", target="results.points"),
        group_by=("constructors.name",),
        ranking=RankingSpec(
            metric="SUM(results.points)", direction="DESC", top_k=1, ties="single"
        ),
        join_entities=("results", "constructors"),
    )
    result = verify_semantic_contract(
        contract,
        _semantics(
            "SELECT c.name, SUM(r.points) FROM results r "
            "JOIN constructors c ON r.constructorId = c.constructorId "
            "GROUP BY c.name ORDER BY SUM(r.points) DESC LIMIT 1"
        ),
        _catalog(tmp_path),
    )
    assert result.status == "pass"
    assert result.violations == ()


def test_missing_optional_contract_fields_do_not_create_violations(tmp_path: Path) -> None:
    contract = SemanticContract(confidence="high")
    result = verify_semantic_contract(
        contract,
        _semantics("SELECT constructorId FROM constructors"),
        _catalog(tmp_path),
    )
    assert result.status == "pass"
    assert result.violations == ()
