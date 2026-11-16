from __future__ import annotations

import pytest
from scripts.audit_birdtrain242_label_contracts import question_sql_flags, audit


def test_questions_asking_count_are_not_answered_by_listing_distinct_rows():
    assert question_sql_flags(
        "How many business ids have opening hours from 8AM to 6PM?",
        "SELECT DISTINCT business_id FROM Business_Hours WHERE opening_time = '8AM'"
    )==["asked_count_but_reference_lists_distinct_rows"]
    assert question_sql_flags("How many orders are there?", "SELECT COUNT(DISTINCT OrderID) FROM orders")==[]


def test_count_of_wins_is_not_win_percentage():
    assert question_sql_flags(
        "Which team has the highest winning percentage?",
        "SELECT AwayTeam FROM matchs WHERE ftr='A' GROUP BY AwayTeam ORDER BY COUNT(AwayTeam) DESC LIMIT 1"
    )==["asked_win_rate_but_reference_ranks_win_count"]
    assert question_sql_flags(
        "Who won the most games?",
        "SELECT team FROM matchs GROUP BY team ORDER BY COUNT(*) DESC LIMIT 1"
    )==[]


def test_audit_refuses_changed_reference_source(tmp_path):
    p=tmp_path/"changed.jsonl"
    p.write_text('{"source_index":2}\n')
    with pytest.raises(RuntimeError,match="SHA-256 drift"):
        audit(p)
