"""Small CPU guardrails for the one-shot Qwen SQL repair diagnostic."""
from evaluation.minimal_one_shot_qwen9b_repair6 import sql_only, HINTS, CONTROL_ID

def test_extract_plain_and_fenced_sql():
    assert sql_only(" SELECT 1; \n") == "SELECT 1;"
    fenced=chr(96)*3+"sql\nSELECT 1;\n"+chr(96)*3
    assert sql_only(fenced)=="SELECT 1;"

def test_only_three_hinted_cases_and_one_known_control():
    assert set(HINTS)=={"169","298","994"}
    assert CONTROL_ID=="755" and CONTROL_ID not in HINTS
