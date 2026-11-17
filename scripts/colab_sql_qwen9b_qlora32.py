"""Physical fixed-Qwen3.5-9B QLoRA on original BIRD TRAIN source SQL (Colab T4).

Predeclared 32 optimization steps; database-separated TRAIN-only 242 source rows.
Publisher reference SQLs passed pinned SQLite preflight but do NOT constitute
independently verified semantic labels. No teacher models, no DEV or HOLDOUT.
Never call a train-loss reduction a correctness improvement.
"""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

SOURCE_SHA="b6851fb0f2b5027833aff45a0b5e01326359b245be1c812edc66986d0df1d402"
VALIDATION_SHA="cc112305bc9de33a03cea704f2edecfdfb45e62cc953ae498cea27fbed93b506"
MODEL="Qwen/Qwen3.5-9B"
MODEL_REV="c202236235762e1c871ad0ccb60c8ee5ba337b9a"
HELPER_COMMIT="6465698908c6883b0f91451785caa11ebfadb090"
STEPS=32
MAX_CONTEXT=4096
SEED=42
SOURCE=Path("/content/birdtrain-original-train242.jsonl")
REPORT=Path("/content/sql-qwen9b-qlora32-proof.json")
OUTPUT=Path("/content/sql-qwen9b-qlora32-checkpoint")
ARCHIVE=Path("/content/sql-qwen9b-qlora32-checkpoint.zip")
PROTOCOL="qwen35-9b-original-birdtrain-db16-qlora32-v1"

def sh(*args):
    subprocess.run([str(arg) for arg in args],check=True)

def verify_source():
    if sha256(SOURCE.read_bytes()).hexdigest()!=SOURCE_SHA:
        raise RuntimeError("Pinned original publisher SQL source SHA drift")
    rows=[json.loads(v) for v in SOURCE.read_text().splitlines() if v.strip()]
    if len(rows)!=242 or len({r["source_index"] for r in rows})!=242:
        raise RuntimeError("Expected 242 unique source SQL rows")
    dbs=Counter(r.get("source_db_id") for r in rows)
    if len(dbs)!=16:
        raise RuntimeError("Expected 16 source TRAIN databases")
    for r in rows:
        msgs=r.get("messages")
        if (not isinstance(msgs,list) or len(msgs)!=3
            or [m.get("role") for m in msgs]!=["system","user","assistant"]
            or sha256(msgs[1]["content"].encode()).hexdigest()!=r.get("input_sha256")
            or not msgs[2].get("content", "").strip()):
            raise RuntimeError("Original question/SQL mismatch")
    print("SQL_QLORA_ORIGINAL_TRAIN_SOURCE_CONFIRMED",json.dumps({
        "n":242,"db_count":16,"sha256":SOURCE_SHA,"validation_source_sha":VALIDATION_SHA,
        "source":"publisher SQL physically compile checked, not correctness-certified",
        "dev_used":False,"teacher_used":False},sort_keys=True),flush=True)
    return rows

def pack(status,error=None):
    if not OUTPUT.is_dir():
        return
    checkpoints=[p.name for p in sorted(OUTPUT.glob("checkpoint-*"))]
    has_final=(OUTPUT/"adapter_model.safetensors").exists()
    report={"status":status,"error_type":error,"protocol":PROTOCOL,"train_sha256":SOURCE_SHA,
            "base_revision":MODEL_REV,"optimizer_target_steps":STEPS,
            "checkpoints":checkpoints,"has_final":has_final,
            "independent_validation_performed":False,"semantic_EX_improvement":None,
            "not_promoted":True}
    REPORT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    with zipfile.ZipFile(ARCHIVE,"w",compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for f in sorted(OUTPUT.rglob("*")):
            if f.is_file():
                z.write(f,arcname="trainer/"+str(f.relative_to(OUTPUT)))
        z.write(REPORT,arcname="sql-qwen9b-qlora32-proof.json")
    print("SQL_QLORA_CHECKPOINT_PACKED",json.dumps({"status":status,
          "checkpoints":checkpoints,"has_final":has_final,
          "archive_bytes":ARCHIVE.stat().st_size}),flush=True)

def normalize_t4_trainable_precision(model, torch, *, capability):
    """Keep all T4 trainable LoRA gradients FP32 before FP16 GradScaler sees them.

    PEFT/TRL may construct trainable adapters in BF16 even if the quantized base
    uses FP16. PyTorch FP16 GradScaler cannot unscale BF16 gradients on T4.
    Never modify frozen base weights or 4-bit quantized tensors.
    """
    if capability >= (8, 0):
        return {"converted": 0, "checked": 0, "skipped_newer_gpu": True}
    converted = 0
    checked = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        checked += 1
        if param.__class__.__name__ == "Params4bit":
            raise RuntimeError(f"Trainable quantized parameter is unsafe: {name}")
        if param.dtype in (torch.float16, torch.bfloat16):
            param.data = param.data.to(torch.float32)
            converted += 1
        if param.dtype != torch.float32:
            raise RuntimeError(f"Unsupported T4 trainable gradient dtype: {name}: {param.dtype}")
    if checked == 0:
        raise RuntimeError("No trainable LoRA parameters found before QLoRA training")
    return {"converted": converted, "checked": checked, "skipped_newer_gpu": False}


def main():
    import torch
    if not torch.cuda.is_available():raise RuntimeError("Physical GPU required")
    gpu=torch.cuda.get_device_properties(0)
    if gpu.total_memory/2**30<14.5:raise RuntimeError("T4/greater VRAM required")
    if OUTPUT.exists() or ARCHIVE.exists():
        raise RuntimeError("Refusing to overwrite previous Colab run")
    rows=verify_source()
    helper=Path("/content/sql-qlora-qwen9b-pinned-helpers")
    if helper.exists():raise RuntimeError("Old helper checkout exists")
    sh("git","clone","--depth","10","--branch",
       "experiment/qlora-v8-durable-stage35-20261115",
       "https://github.com/kolihhan/small-agent-qlora.git",helper)
    sh("git","-C",helper,"checkout","--detach",HELPER_COMMIT)
    actual=subprocess.check_output(["git","-C",str(helper),"rev-parse","HEAD"],text=True).strip()
    if actual!=HELPER_COMMIT:raise RuntimeError("Trainer helper revision drift")
    sh(sys.executable,"-m","pip","install","-q","-e",str(helper)+"[train]",
       "jinja2>=3.1","sentencepiece>=0.2")
    sys.path.insert(0,str(helper/"src"))
    from gaia_small_agent.training.qlora import (
        _tokenize_completion_examples,
        _prepare_kbit_model_memory_safe,
        _configure_trl_legacy_addmm,
    )
    from datasets import Dataset
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
    from peft import LoraConfig,get_peft_model
    from trl import SFTTrainer,SFTConfig
    from trl.kernels import chunked_logprob

    path=snapshot_download(MODEL,revision=MODEL_REV)
    tok=AutoTokenizer.from_pretrained(path,trust_remote_code=True)
    examples=[]
    for r in rows:
        prompt=tok.apply_chat_template(r["messages"][:2],tokenize=False,
                                       add_generation_prompt=True,enable_thinking=False)
        full=tok.apply_chat_template(r["messages"],tokenize=False,
                                     add_generation_prompt=False,enable_thinking=False)
        if not full.startswith(prompt):
            raise RuntimeError("Cannot isolate source supervised SQL completion")
        completion=full[len(prompt):]
        if not completion.strip():raise RuntimeError("Publisher SQL target is empty")
        examples.append({"prompt":prompt,"completion":completion})
    # Do not left-trim source schema. Reject over-limit rows with explicit record.
    selected=[]; skipped=[]
    for i,e in enumerate(examples):
        p=tok(e["prompt"],add_special_tokens=False)["input_ids"]
        c=tok(e["completion"],add_special_tokens=False)["input_ids"]
        if len(p)+len(c)>MAX_CONTEXT:
            skipped.append({"source_index":rows[i]["source_index"],"total_tokens":len(p)+len(c)})
        else:
            selected.append(e)
    if len(selected)<180 or len(skipped)>62:
        raise RuntimeError("Too many original SQL contexts exceed 4096; do not silently truncate")
    tokenized=_tokenize_completion_examples(tok,selected,max_length=MAX_CONTEXT,
                                             require_full_prompt=True)
    print("SQL_QLORA_FULL_CONTEXT_PREFLIGHT_PASS",json.dumps({
        "kept":len(tokenized),"excluded_over_context":len(skipped),
        "max_context":MAX_CONTEXT,"no_left_trim":True,
        "base_revision":MODEL_REV,"gpu":gpu.name},sort_keys=True),flush=True)
    OUTPUT.mkdir(parents=True,exist_ok=False)
    (OUTPUT/"source-exclusions.json").write_text(json.dumps(skipped,indent=2))
    bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
                           bnb_4bit_use_double_quant=True,
                           bnb_4bit_compute_dtype=torch.float16)
    model=AutoModelForCausalLM.from_pretrained(
        path,quantization_config=bnb,device_map="auto",
        trust_remote_code=True,dtype=torch.float16)
    model=_prepare_kbit_model_memory_safe(model,torch,gradient_checkpointing=True)
    lora=LoraConfig(r=8,lora_alpha=16,lora_dropout=0.05,bias="none",
                    task_type="CAUSAL_LM",target_modules=["q_proj","v_proj"])
    model=get_peft_model(model,lora)
    precision_preflight=normalize_t4_trainable_precision(
        model,torch,capability=torch.cuda.get_device_capability(0))
    print("SQL_QLORA_T4_TRAINABLE_FP32_PREFLIGHT",
          json.dumps(precision_preflight,sort_keys=True),flush=True)
    if _configure_trl_legacy_addmm(chunked_logprob,capability=torch.cuda.get_device_capability(0)):
        print("SQL_QLORA_T4_TRL_MIXED_GEMM_FP32_FALLBACK",flush=True)
    settings=SFTConfig(
        output_dir=str(OUTPUT),num_train_epochs=1.0,max_steps=STEPS,
        learning_rate=5e-5,per_device_train_batch_size=1,
        gradient_accumulation_steps=1,max_length=MAX_CONTEXT,
        completion_only_loss=False,loss_type="nll",
        logging_steps=1,save_strategy="steps",save_steps=16,save_total_limit=2,
        report_to="none",fp16=True,bf16=False,seed=SEED,data_seed=SEED)
    (OUTPUT/"study-design.json").write_text(json.dumps({
        "model":MODEL,"base_revision":MODEL_REV,"train_source_sha256":SOURCE_SHA,
        "train_samples":len(tokenized),"source_total":242,"train_db_count":16,
        "validation_db_count":4,"validation_sha256":VALIDATION_SHA,
        "max_context":MAX_CONTEXT,"steps":STEPS,"lr":5e-5,"lora_r":8,
        "lora_alpha":16,"lora_modules":["q_proj","v_proj"],
        "source_label":"publisher reference SQL compiling on source SQLite; semantic label unproven",
        "no_teacher":True,"dev_holdout_unused":True,"no_dynamic_early_selection":True,
        "original_source_sql_not_generated_by_teacher":True},indent=2))
    trainer=SFTTrainer(model=model,args=settings,train_dataset=Dataset.from_list(tokenized),
                        processing_class=tok)
    # TRL may cast LoRA weights back to BF16 while wrapping the model.
    # The optimizer is created later by trainer.train(); restoring ONLY the
    # trainable LoRA parameters to FP32 here keeps FP16 GradScaler safe.
    # normalize_t4_trainable_precision rejects trainable Params4bit, unknown
    # dtypes and zero trainable parameters; frozen base weights are untouched.
    precision_after_trainer=normalize_t4_trainable_precision(
        trainer.model,torch,capability=torch.cuda.get_device_capability(0))
    if precision_after_trainer["converted"]:
        print("SQL_QLORA_TRL_RECAST_TRAINABLES_RESTORED_FP32",
              json.dumps(precision_after_trainer,sort_keys=True),flush=True)
    # A second pass MUST be idempotent; no trainable half/BF16 remains.
    verified_after_trainer=normalize_t4_trainable_precision(
        trainer.model,torch,capability=torch.cuda.get_device_capability(0))
    if verified_after_trainer["converted"] or verified_after_trainer["checked"] != precision_after_trainer["checked"]:
        raise RuntimeError("TRL post-init FP32 trainable dtype verification failed")
    print("SQL_QLORA_T4_TRAINER_FP32_VERIFIED",
          json.dumps(verified_after_trainer,sort_keys=True),flush=True)
    started=time.monotonic()
    try:
        trainer.train()
        trainer.save_model(str(OUTPUT))
        state_path=OUTPUT/"checkpoint-32"/"trainer_state.json"
        if not state_path.is_file():raise RuntimeError("Final step state missing")
        state=json.loads(state_path.read_text())
        if state.get("global_step")!=STEPS:raise RuntimeError("Not all optimizer steps finished")
        weight=OUTPUT/"adapter_model.safetensors"
        if not weight.is_file():raise RuntimeError("No trained final adapter")
        digest=sha256(weight.read_bytes()).hexdigest()
        proof={"actual_optimizer_steps":STEPS,"source_total":242,
               "kept_examples":len(tokenized),"excluded_examples":len(skipped),
               "adapter_sha256":digest,"adapter_bytes":weight.stat().st_size,
               "base_revision":MODEL_REV,"training_seconds":round(time.monotonic()-started),
               "no_validation_used_in_training":True,"no_holdout_scored":True,
               "semantic_EX_improvement":None}
        (OUTPUT/"physical-sql-training-proof.json").write_text(json.dumps(proof,indent=2))
        pack("complete_verified_32_steps")
        print("SQL_QWEN9B_QLORA32_PHYSICAL_OPTIMIZER_PROOF",json.dumps(proof,sort_keys=True),flush=True)
    except BaseException as exc:
        pack("partial_failure",type(exc).__name__)
        raise

if __name__=="__main__":
    main()
