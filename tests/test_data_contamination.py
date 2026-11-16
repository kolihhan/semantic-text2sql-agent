from pathlib import Path
import json
import pytest
from scripts.check_data_contamination import audit, load_rows, normalize


def corpus(tmp_path: Path, filename: str, rows: list[dict]) -> Path:
    p = tmp_path / filename
    p.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return p


def test_exact_training_question_match_blocks(tmp_path):
    ref = corpus(tmp_path, "dev.json", [{"question": "Which customer bought the maximum number of books in 2021?", "SQL": "SELECT count(*) FROM sales"}])
    train = corpus(tmp_path, "train.json", [{"question": "Which customer bought the maximum number of books in 2021?", "SQL": "SELECT x FROM y"}])
    result = audit(train, {"bird-dev": ref}, {"bird-dev"})
    assert result["status"] == "CONTAMINATED"
    assert result["finding_count"] >= 1


def test_near_paraphrase_flagged(tmp_path):
    ref = corpus(tmp_path, "dev.json", [{"question": "How many people visited the museum during the year twenty twenty three?", "SQL": "SELECT count(*) FROM visitors"}])
    train = corpus(tmp_path, "train.json", [{"question": "How many people visited the museum during the year twenty twenty three!", "SQL": "SELECT x FROM y"}])
    assert audit(train, {"bird-dev": ref}, {"bird-dev"})["status"] == "CONTAMINATED"


def test_clean_supplied_reference_only(tmp_path):
    ref = corpus(tmp_path, "dev.json", [{"question": "How many visitors entered the old city museum?", "SQL": "SELECT count(*) FROM visitors"}])
    train = corpus(tmp_path, "train.json", [{"question": "How many whales migrate across Pacific oceans annually?", "SQL": "SELECT whales FROM ocean"}])
    result = audit(train, {"bird-dev": ref}, {"bird-dev"})
    assert result["status"] == "PASS_LEXICAL_SCREEN_ONLY"
    assert result["finding_count"] == 0


def test_fails_closed_without_required_reference(tmp_path):
    train = corpus(tmp_path, "train.json", [{"question": "Please read the linked article in full"}])
    with pytest.raises(ValueError, match="required reference roles"):
        audit(train, {}, {"gaia-all", "bird-dev"})


def test_fails_closed_on_missing_sql_label(tmp_path):
    ref = corpus(tmp_path, "dev.json", [{"question": "How many people visited?"}])
    train = corpus(tmp_path, "train.json", [{"question": "How many whales?"}])
    with pytest.raises(ValueError, match="question and SQL"):
        audit(train, {"bird-dev": ref}, {"bird-dev"})


def test_gaia_all_role_with_nested_trajectory(tmp_path):
    ref = corpus(tmp_path, "gaia.json", [{"Question": "Find the library that published the winning book", "Final answer": "West Library"}])
    train = corpus(tmp_path, "train.json", [{"messages": [{"role": "user", "content": "Find the library that published the winning book"}]}])
    result = audit(train, {"gaia-all": ref}, {"gaia-all"})
    assert result["status"] == "CONTAMINATED"
