import importlib.util
import json
from pathlib import Path


def load_module(name: str):
    root = Path(__file__).parents[1]
    path = root / "evaluation" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"p1_eval_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_bird_loader_keeps_gold_out_of_inference_case(tmp_path):
    loader = load_module("bird_loader")
    source = tmp_path / "bird.json"
    source.write_text(json.dumps([{
        "question_id": "x", "db_id": "db1", "question": "Q?",
        "evidence": "Use the badge field.", "SQL": "SELECT 1",
    }]), encoding="utf-8")
    cases, labels = loader.load_bird_json(source)
    assert cases[0].__dict__ == {"case_id": "x", "database_id": "db1", "question": "Q?", "evidence": "Use the badge field."}
    assert labels[0].__dict__ == {"case_id": "x", "sql": "SELECT 1"}
    assert "SQL" not in cases[0].__dict__


def test_paired_transition_api_is_the_only_metric_contract():
    metrics = load_module("metrics")
    assert metrics.paired_transition_counts([{
        "direct": {"official_ex": False, "execution_success": False},
        "guarded": {"official_ex": True, "execution_success": True},
    }]) == {
        "direct_wrong_to_guarded_correct": 1,
        "direct_correct_to_guarded_wrong": 0,
        "direct_execution_fail_to_guarded_success": 1,
        "direct_execution_success_to_guarded_fail": 0,
    }
