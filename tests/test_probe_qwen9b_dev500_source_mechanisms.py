"""Synthetic source-only controls for the fixed Qwen DEV500 eligibility screen."""
from scripts.probe_qwen9b_dev500_source_mechanisms import classify


def test_growth_rate_split_is_a_candidate_not_a_correctness_claim():
    assert classify(
        "What is the growth rate of total loans from 1996 to 1997?",
        "SELECT SUM(a) AS numerator, SUM(b) AS denominator",
    ) == ("SCALAR_GROWTH_RATE_SPLIT_OUTPUT",)


def test_growth_rate_already_computed_must_not_trigger():
    assert classify(
        "What is the growth rate of total loans?",
        "SELECT (SUM(a) - SUM(b)) * 1.0 / SUM(b) AS growth_rate",
    ) == ()


def test_percentage_with_explicit_division_is_not_flagged():
    assert classify(
        "What is the percentage change in 2021?",
        "SELECT 100.0 * (x-y)/y FROM t",
    ) == ()


def test_ordinal_nth_limited_to_first_is_a_candidate():
    assert classify(
        "Which is the 7th highest math score?",
        "SELECT school FROM schools ORDER BY math DESC LIMIT 1",
    ) == ("NTH_RANK_LIMIT1_WITHOUT_OFFSET",)


def test_ordinal_with_offset_or_window_is_not_flagged():
    q = "Which is the seventh highest math score?"
    assert classify(q, "SELECT school FROM schools ORDER BY math DESC LIMIT 1 OFFSET 6") == ()
    assert classify(q, "SELECT school FROM (SELECT school, DENSE_RANK() OVER (ORDER BY math DESC) AS rnk FROM schools) WHERE rnk=7") == ()


def test_minimum_cardinality_does_not_imply_exact_topk():
    assert classify("List at least five superheroes.", "SELECT name FROM superhero LIMIT 5") == ()


def test_ordinary_first_name_is_not_rank():
    assert classify("List the first name of Jack", "SELECT first_name FROM people ORDER BY id DESC LIMIT 1") == ()


def test_two_highest_is_not_ordinal_second():
    assert classify("Show the two highest school scores", "SELECT name FROM schools ORDER BY score DESC LIMIT 1") == ()
