from pathlib import Path


def test_runtime_source_does_not_import_evaluation_or_define_gold_fields():
    src = Path(__file__).parents[1] / "src" / "semantic_sql"
    text = "\n".join(path.read_text(encoding="utf-8") for path in src.glob("*.py"))
    lowered = text.lower()
    assert "import evaluation" not in lowered
    assert "from evaluation" not in lowered
    for token in ("gold_sql", "reference_sql", "expected_sql"):
        assert token not in lowered
