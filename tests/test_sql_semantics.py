from __future__ import annotations

from semantic_sql.sql_semantics import ColumnRef, parse_sql_semantics


def test_parse_sql_semantics_extracts_projection_aggregate_group_order_limit() -> None:
    sql = """
    SELECT c.name, SUM(r.points) AS total
    FROM results AS r
    JOIN constructors AS c ON r.constructorId = c.constructorId
    WHERE r.year >= 1980
    GROUP BY c.name
    ORDER BY SUM(r.points) DESC
    LIMIT 1
    """
    result = parse_sql_semantics(sql)
    assert result.status == "ok"
    assert result.semantics is not None
    semantics = result.semantics
    assert ColumnRef("constructors", "name") in semantics.projections
    assert semantics.aggregations[0].function == "SUM"
    assert semantics.aggregations[0].target == ColumnRef("results", "points")
    assert semantics.group_by == (ColumnRef("constructors", "name"),)
    assert semantics.order_by[0].direction == "DESC"
    assert "sum(results.points)" in semantics.order_by[0].expression
    assert semantics.limit == 1
    assert "1980" in semantics.literals


def test_parse_sql_semantics_detects_count_distinct() -> None:
    result = parse_sql_semantics(
        "SELECT COUNT(DISTINCT a.molecule_id) FROM atom AS a"
    )
    assert result.status == "ok"
    assert result.semantics is not None
    aggregation = result.semantics.aggregations[0]
    assert aggregation.function == "COUNT"
    assert aggregation.target == ColumnRef("atom", "molecule_id")
    assert aggregation.distinct is True


def test_aliases_resolve_to_physical_table_names() -> None:
    result = parse_sql_semantics(
        "SELECT c.name FROM constructors c JOIN results r "
        "ON r.constructorId = c.constructorId"
    )
    assert result.status == "ok"
    assert result.semantics is not None
    assert result.semantics.projections == (ColumnRef("constructors", "name"),)
    join = result.semantics.joins[0]
    assert {
        join.left,
        join.right,
    } == {
        ColumnRef("results", "constructorid"),
        ColumnRef("constructors", "constructorid"),
    }


def test_reverse_join_equality_normalizes_same_edge() -> None:
    left = parse_sql_semantics(
        "SELECT c.name FROM constructors c JOIN results r "
        "ON r.constructorId = c.constructorId"
    )
    right = parse_sql_semantics(
        "SELECT c.name FROM constructors c JOIN results r "
        "ON c.constructorId = r.constructorId"
    )
    assert left.status == right.status == "ok"
    assert left.semantics is not None and right.semantics is not None
    assert {left.semantics.joins[0].left, left.semantics.joins[0].right} == {
        right.semantics.joins[0].left,
        right.semantics.joins[0].right,
    }


def test_unqualified_column_with_multiple_sources_remains_unresolved() -> None:
    result = parse_sql_semantics(
        "SELECT name FROM constructors c JOIN teams t ON c.id = t.id"
    )
    assert result.status == "ok"
    assert result.semantics is not None
    assert result.semantics.projections == (ColumnRef(None, "name"),)


def test_parse_sql_semantics_extracts_filter_literals() -> None:
    result = parse_sql_semantics(
        "SELECT id FROM races WHERE name = 'Monaco Grand Prix' AND year BETWEEN 1980 AND 2010"
    )
    assert result.status == "ok"
    assert result.semantics is not None
    assert set(result.semantics.literals) >= {"Monaco Grand Prix", "1980", "2010"}


def test_parse_failure_returns_skipped_not_exception() -> None:
    result = parse_sql_semantics("SELECT FROM WHERE )")
    assert result.status == "skipped"
    assert result.semantics is None
    assert result.reason


def test_cte_query_does_not_crash_normalizer() -> None:
    result = parse_sql_semantics(
        "WITH ranked AS (SELECT id, points FROM results) "
        "SELECT id FROM ranked ORDER BY points DESC LIMIT 1"
    )
    assert result.status == "ok"
    assert result.semantics is not None
    assert result.semantics.limit == 1
