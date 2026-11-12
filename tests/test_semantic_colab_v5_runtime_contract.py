"""Prevent pre-optimizer Transformers v5 failure and post-SFT artifact loss."""
from pathlib import Path
import ast


def test_hf_v5_training_uses_supported_scheduler_key():
    source = Path("scripts/colab_semantic_sql_remote.py").read_text()
    ast.parse(source)
    assert "warmup_ratio=" not in source
    assert "warmup_steps=0.05" in source
    assert 'SOURCE_SHA = "a25262e1f5483a35ab27bf7a5c15f6fff5f5fc6e"' in source
    assert 'BRANCH = "fix/semantic-sft-reserve-shard-holdout-20261112"' in source
    assert "reserve_semantic_audit_holdout(" in source


def test_adapter_saved_before_running_risky_dev_eval():
    source = Path("scripts/colab_semantic_sql_remote.py").read_text()
    assert 'COLAB_SQL_PRE_EVAL_ADAPTER_SAVED' in source
    assert source.index('COLAB_SQL_PRE_EVAL_ADAPTER_SAVED') < source.index('dev_rows,dev_schema,dbroot=download_bird_dev()')
    assert 'with zipfile.ZipFile(ARCHIVE,"a"' in source


def test_workflow_recovers_adapter_on_remote_failure_before_gpu_stop():
    content = Path(".github/workflows/colab-gpu-train.yml").read_text()
    assert "remote_rc=$?" in content
    assert content.index("remote_rc=$?") < content.index('colab --auth=adc download')
    assert content.index('colab --auth=adc download') < content.index('if [ "$remote_rc" -ne 0 ]')
    assert 'trap cleanup EXIT' in content
    assert "No verified adapter available for recovery" in content


def test_preflight_constructs_hf_v5_training_args_without_gpu():
    content = Path(".github/workflows/semantic-colab-v5-args-preflight.yml").read_text()
    assert "declared=set(inspect.signature(TrainingArguments).parameters)" in content
    assert '"warmup_steps" in requested' in content
    assert "inspect.signature(TrainingArguments)" in content
