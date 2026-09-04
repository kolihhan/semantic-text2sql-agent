from semantic_sql.cli import main


def test_cli_demo_prints_current_runtime_stages(capsys):
    code = main(["demo"])
    out = capsys.readouterr().out
    assert code == 0
    for label in ("SQL", "VERIFY", "RESULT"):
        assert label in out
    assert "CZE" in out
    assert "SEMANTIC PLAN" not in out
    assert "GROUNDING" not in out


def test_cli_ask_reports_verification_refusal(capsys):
    code = main(["ask", "What is each customer's salary?"])
    out = capsys.readouterr().out.lower()
    assert code == 2
    assert "verification_failed" in out
    assert "salary" in out
