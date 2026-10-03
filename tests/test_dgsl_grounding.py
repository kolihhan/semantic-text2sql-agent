from __future__ import annotations

from pathlib import Path
import sqlite3

from evaluation.grounding_metrics import (
    extract_gold_schema_usage,
    score_grounding_pack,
)
from semantic_sql.catalog import DatabaseCatalog
from semantic_sql.grounding import (
    IndexedValue,
    QuestionDecomposition,
    build_grounding_pack,
    build_value_index,
    match_indexed_values,
)


def _make_formula_db(path: Path) -> DatabaseCatalog:
    con = sqlite3.connect(path)
    try:
        con.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE circuits (
                circuitId INTEGER PRIMARY KEY,
                name TEXT,
                lat REAL,
                lng REAL
            );
            CREATE TABLE races (
                raceId INTEGER PRIMARY KEY,
                circuitId INTEGER,
                year INTEGER,
                name TEXT,
                FOREIGN KEY (circuitId) REFERENCES circuits(circuitId)
            );
            CREATE TABLE results (
                resultId INTEGER PRIMARY KEY,
                raceId INTEGER,
                constructorId INTEGER,
                fastestLapTime TEXT,
                points REAL,
                FOREIGN KEY (raceId) REFERENCES races(raceId)
            );
            INSERT INTO circuits VALUES (1, 'Monaco', 43.7347, 7.42056);
            INSERT INTO races VALUES (10, 1, 2010, 'Monaco Grand Prix');
            INSERT INTO results VALUES (100, 10, 7, '1:29.488', 25.0);
            """
        )
        con.commit()
    finally:
        con.close()
    return DatabaseCatalog.from_sqlite(path)


def test_dgsl_value_hits_and_fk_shortest_bridge(tmp_path: Path) -> None:
    catalog = _make_formula_db(tmp_path / "formula.sqlite")
    value_index = build_value_index(catalog, max_values_per_column=50)
    decomposition = QuestionDecomposition(
        entity="constructor lap time",
        filters=("Monaco Grand Prix", "1:29.488"),
        raw_question="Which constructor recorded 1:29.488 at the Monaco Grand Prix?",
    )

    pack = build_grounding_pack(
        catalog,
        decomposition=decomposition,
        question=decomposition.raw_question,
        descriptions={
            ("results", "fastestLapTime"): "fastest lap time recorded by the result",
            ("races", "name"): "grand prix race name",
        },
        value_index=value_index,
    )

    assert "circuits" in pack.selected_tables
    assert "races" in pack.selected_tables
    assert "results" in pack.selected_tables
    assert any(hit.value == "1:29.488" for hit in pack.value_hits)
    assert any(hit.value == "Monaco Grand Prix" for hit in pack.value_hits)
    assert ("races", "circuitId", "circuits", "circuitId") in pack.join_edges
    assert ("results", "raceId", "races", "raceId") in pack.join_edges
    assert "results.fastestLapTime = '1:29.488'" in pack.context


def test_dgsl_falls_back_to_full_schema_when_nothing_links(tmp_path: Path) -> None:
    catalog = _make_formula_db(tmp_path / "formula.sqlite")
    pack = build_grounding_pack(
        catalog,
        decomposition=QuestionDecomposition(
            entity="zzzxxyy",
            raw_question="zzzxxyy",
        ),
        question="zzzxxyy",
        value_index=(),
    )

    assert pack.fallback_full_schema is True
    assert set(pack.selected_tables) == set(catalog.tables)


def test_gold_usage_and_grounding_metrics_are_offline_only(tmp_path: Path) -> None:
    catalog = _make_formula_db(tmp_path / "formula.sqlite")
    value_index = build_value_index(catalog, max_values_per_column=50)
    question = "What points were scored for the Monaco Grand Prix?"
    pack = build_grounding_pack(
        catalog,
        decomposition=QuestionDecomposition(
            entity="points",
            filters=("Monaco Grand Prix",),
            raw_question=question,
        ),
        question=question,
        value_index=value_index,
    )
    gold_sql = (
        "SELECT results.points "
        "FROM results "
        "JOIN races ON results.raceId = races.raceId "
        "WHERE races.name = 'Monaco Grand Prix'"
    )

    usage = extract_gold_schema_usage(gold_sql, catalog)
    assert usage.tables == frozenset({"results", "races"})
    assert "results.points" in usage.columns
    assert "races.name" in usage.columns
    assert ("results", "raceId", "races", "raceId") in usage.fk_edges
    assert "Monaco Grand Prix" in usage.string_literals

    metrics = score_grounding_pack(pack, gold_sql=gold_sql, catalog=catalog)
    assert metrics["table_recall"] == 1.0
    assert metrics["column_recall"] == 1.0
    assert metrics["fk_bridge_recall"] == 1.0
    assert metrics["value_grounding_recall"] == 1.0


def test_numeric_value_matching_respects_digit_boundaries() -> None:
    hits = match_indexed_values(
        (IndexedValue(table="races", column="raceId", value="10"),),
        question="Which races happened in 2010?",
    )

    assert hits == ()
