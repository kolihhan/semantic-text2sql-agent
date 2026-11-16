"""Guard against drift between the frozen Qwen9B workflow and its pinned CLI.

The exact-v5 protocol intentionally checks out BASELINE_COMMIT, not HEAD.
"""

from pathlib import Path

WORKFLOW = Path(".github/workflows/qwen9b-paired-frozen100.yml")


def test_qwen9b_workflow_uses_supported_pinned_cli():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "BASELINE_COMMIT: d3b687759cba2d9e7b4d8fb1d42bc4c33da575c4" in workflow
    assert "--max-repairs" not in workflow  # Pinned CLI hardcodes max_repairs=2.
    for flag in ("--source", "--database-root", "--manifest", "--model",
                 "--timeout 600", "--max-rows 50000", "--output"):
        assert flag in workflow


def test_qwen9b_workflow_rejects_incomplete_or_invalid_shards():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert 'payload["status"] == "complete"' in workflow
    assert 'assert len(cases) == 100' in workflow
    assert 'assert len(set(actual_ids)) == 100' in workflow
    assert 'assert len(digests) == 1' in workflow
    assert 'assert all(case.get("initial_sql_identical") is True for case in cases)' in workflow
    assert "if: ${{ always() && needs.preflight.result == 'success' && needs.benchmark.result == 'success' }}" in workflow
