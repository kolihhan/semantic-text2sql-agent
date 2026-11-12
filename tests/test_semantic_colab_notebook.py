"""Colab semantic SFT notebook must train 7B and verify actual SQL meaning."""
import ast
import json
from pathlib import Path


def test_semantic_colab_notebook_code_and_training_contract():
    doc = json.loads(Path("notebooks/semantic_xiyan7b_qlora_colab.ipynb").read_text(encoding="utf-8"))
    assert doc["nbformat"] == 4 and doc["metadata"]["accelerator"] == "GPU"
    cells = [c for c in doc["cells"] if c["cell_type"] == "code"]
    assert len(cells) >= 6
    source=[]
    for cell in cells:
        assert not cell.get("outputs")
        code = "".join(cell["source"])
        ast.parse(code)
        source.append(code)
    all_code = "\n".join(source)
    for frag in (
        "XGenerationLab/XiYanSQL-QwenCoder-7B-2504",
        "birdsql/bird23-train-filtered",
        "train-00000-of-00001.jsonl",
        "train_tables.json",
        "prepare_semantic_sft_rows(",
        "tokenize_semantic_pair(",
        "load_in_4bit=True",
        'bnb_4bit_quant_type="nf4"',
        "get_peft_model(",
        "trainer.train(",
        "run_inference(",
        "compare_paired_scores(",
        "save_pretrained(",
        "drive.mount(",
        "validation_db_fraction=0.15",
    ):
        assert frag in all_code, frag
    assert "PeftModel" in Path("experiments/semantic_sft_eval.py").read_text(encoding="utf-8")
    assert "small-agent-policy-v3" not in all_code
    assert "max_steps=64" not in all_code
    assert "train.json" not in all_code or "train-00000-of-00001.jsonl" in all_code
