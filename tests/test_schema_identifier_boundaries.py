from evaluation.schema_context import _tokens


def test_mixed_case_schema_identifier_preserves_word_boundaries() -> None:
    # A question about birth dates must match the real workerBirthDate field.
    assert {"worker", "birth", "date"} <= _tokens("workerBirthDate")
    assert {"birth", "date"} <= _tokens("birth_date")
