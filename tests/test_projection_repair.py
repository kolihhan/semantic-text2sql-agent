import pytest
from evaluation.projection_repair import eligible, repair


@pytest.mark.parametrize("sql,expected", [
    (
        "SELECT g.Name AS GenreName, AVG(t.Milliseconds) AS AverageDuration FROM Track t JOIN Genre g ON t.GenreId = g.GenreId GROUP BY g.Name ORDER BY AverageDuration DESC LIMIT 1;",
        "SELECT g.Name AS GenreName FROM Track t JOIN Genre g ON t.GenreId = g.GenreId GROUP BY g.Name ORDER BY AVG(t.Milliseconds) DESC LIMIT 1;"
    ),
    (
        "SELECT f.title, COUNT(r.rental_id) AS rental_count FROM film f JOIN rental r ON r.film_id=f.film_id GROUP BY f.film_id ORDER BY rental_count DESC LIMIT 1;",
        "SELECT f.title FROM film f JOIN rental r ON r.film_id=f.film_id GROUP BY f.film_id ORDER BY COUNT(r.rental_id) DESC LIMIT 1;"
    ),
])
def test_known_sql_shape_for_structural_correctness_only(sql, expected):
    output, reason = repair("Which genre/film has the most tracks or rentals?", sql)
    assert output == expected
    assert reason == "removed_unused_ranking_metrics"


def test_never_drop_requested_number():
    sql = "SELECT name, COUNT(*) AS n FROM jobs GROUP BY name ORDER BY n DESC LIMIT 1"
    assert repair("Which job has the most applications, and the count?", sql)[0] == sql


def test_never_drop_pure_aggregate_answer():
    sql = "SELECT MAX(score) FROM tournament"
    assert repair("What is the highest score in tournament?", sql)[0] == sql


def test_does_not_rewrite_having_or_cte():
    sql = "SELECT name, COUNT(*) AS n FROM x GROUP BY name HAVING n>3 ORDER BY n DESC LIMIT 1"
    assert repair("Which name has the most rows?", sql)[0] == sql


def test_no_ranking_metric_alias_reference_returns_original():
    sql = "SELECT name, SUM(x) AS n FROM table_a GROUP BY name ORDER BY name DESC LIMIT 1"
    assert repair("Which name has the highest sum?", sql)[0] == sql


def test_no_gold_sql_needed_and_aggregate_arguments_with_commas_parse():
    sql = "SELECT x.name, COUNT(COALESCE(y.b, 0)) AS ct FROM x JOIN y ON x.id=y.id GROUP BY x.name ORDER BY ct DESC LIMIT 1"
    fixed, _ = repair("Which name has the most items?", sql)
    assert "COUNT(COALESCE(y.b, 0)) DESC" in fixed and "AS ct FROM" not in fixed


def test_not_a_generic_reviewer():
    sql = "SELECT n, SUM(q) FROM tbl GROUP BY n"
    assert repair("Show each name and total quantity", sql)[0] == sql
    assert not eligible("How many customers exist?")
