"""Static headless Colab GPU training and API authorization contract."""
from __future__ import annotations
import ast
from pathlib import Path


def test_official_colab_action_does_not_allocate_before_auth_and_cleans_up_gpu():
    text = Path(".github/workflows/colab-gpu-train.yml").read_text(encoding="utf-8")
    assert "google-colab-cli==0.8.0" in text
    assert "GOOGLE_APPLICATION_CREDENTIALS=" in text
    assert "cred.refresh(Request())" in text
    assert "COLAB_AUTH_REJECTED" in text
    assert text.index("cred.refresh(Request())") < text.index("colab --auth=adc run")
    assert 'trap cleanup EXIT' in text
    assert 'colab --auth=adc stop -s "$session"' in text
    assert "colab --auth=adc download" in text
    assert "actions/upload-artifact@" in text
    assert "workflow_dispatch:" in text
    assert ".github/COLAB_GPU_START" in text
    assert 'if: always()' in text
    assert "colab drivemount" not in text
    assert "echo ${COLAB_TOKEN}" not in text


def test_noninteractive_remote_script_parses_and_only_claims_success_after_adapter():
    source = Path("scripts/colab_semantic_sql_remote.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert "XGenerationLab/XiYanSQL-QwenCoder-7B-2504" in source
    assert 'assert torch.cuda.is_available()' in source
    assert 'adapter_model.safetensors' in source
    assert 'COLAB_GPU' in source
    assert 'hashlib.sha256' in source
    assert 'google.colab' not in source
    assert "drive.mount(" not in source
    assert 'HF_TOKEN' not in source
    assert 'Authorization' not in source
    assert 'prerequisite_colab_secret' not in source
    assert 'compare_paired_scores(' in source
    assert 'run_inference(' in source
    assert 'prepare_semantic_sft_rows(' in source
    assert 'dev_db_ids=' in source
    assert 'BIRD DEV' in source

