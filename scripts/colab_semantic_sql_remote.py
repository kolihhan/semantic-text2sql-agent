"""Headless full BIRD TRAIN semantic XiYanSQL 7B QLoRA on Google Colab.

Authenticates via GitHub Actions + official Colab CLI, not notebook Drive prompts.
BIRD DEV labels enter only during paired independent evaluation AFTER SFT.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import zipfile


SOURCE_SHA = "a25262e1f5483a35ab27bf7a5c15f6fff5f5fc6e"
BRANCH = "fix/semantic-sft-reserve-shard-holdout-20261112"
MODEL = "XGenerationLab/XiYanSQL-QwenCoder-7B-2504"
FILTERED_REV = "4068469807b255fcfc0816bdd520946fe460d256"
SCHEMA_REV = "c4a98a1"
OUT = Path("/content/semantic-sql-full-remote")
ARCHIVE = Path("/content/semantic-sql-full-adapter.zip")


def run(*args: str) -> None:
    print("RUN",args[0],args[1] if len(args)>1 else "",flush=True)
    subprocess.run(list(args),check=True)


def download_bird_dev() -> tuple[list[dict],list[dict],Path]:
    url="https://huggingface.co/datasets/HAL-9001/bird-dev/resolve/main/dev.zip"
    expected="cdd6d19faeb45a23970b98d3ef6c40a87987c95459c2cf12076897a60cf5a630"
    zpath=Path("/content/bird-dev-for-independent-eval.zip")
    urllib.request.urlretrieve(url,zpath)
    if hashlib.sha256(zpath.read_bytes()).hexdigest()!=expected:
        raise ValueError("BIRD DEV download does not match pinned checksum")
    root=Path("/content/bird-dev-for-independent-eval")
    root.mkdir(exist_ok=False)
    with zipfile.ZipFile(zpath) as archive:
        archive.extractall(root)
    source=list(root.rglob("dev.json"))
    tables=list(root.rglob("dev_tables.json"))
    nested=list(root.rglob("dev_databases.zip"))
    assert len(source)==len(tables)==len(nested)==1,"BIRD DEV archive structure changed"
    dest=root/"database-files"
    dest.mkdir()
    with zipfile.ZipFile(nested[0]) as archive:
        archive.extractall(dest)
    dbroots=[p for p in dest.rglob("dev_databases") if p.is_dir() and any(p.rglob("*.sqlite"))]
    if not dbroots:
        dbroots=[dest]
    return json.loads(source[0].read_text()), json.loads(tables[0].read_text()), dbroots[0]


def main() -> None:
    import torch
    assert torch.cuda.is_available(),"Colab CLI did not provide GPU"
    vram=torch.cuda.get_device_properties(0).total_memory/(1024**3)
    assert vram>=14.5,"At least 15GB GPU needed for XiYan 7B NF4"
    max_length=1536 if vram<20 else 2048
    print("COLAB_GPU",json.dumps({
        "model":MODEL,"gpu":torch.cuda.get_device_name(0),
        "vram_gib":round(vram,2),"max_length":max_length,
    }),flush=True)
    repo=Path("/content/semantic-text2sql-agent")
    assert not repo.exists(),"unexpected Colab VM already contains repo"
    run("git","clone","--branch",BRANCH,"--depth","100",
        "https://github.com/kolihhan/semantic-text2sql-agent.git",str(repo))
    run("git","-C",str(repo),"checkout","--detach",SOURCE_SHA)
    sha=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    assert sha==SOURCE_SHA,"source checkout mismatch"
    run(sys.executable,"-m","pip","install","-q","-e",str(repo)+"[dev]",
        "transformers>=5.15,<6","peft>=0.20,<1","accelerate>=1.10,<2",
        "bitsandbytes>=0.50.1,<1","datasets>=4","huggingface_hub>=1",
        "jinja2>=3.1")
    sys.path.insert(0,str(repo))

    from huggingface_hub import hf_hub_download
    from datasets import Dataset
    from experiments.semantic_sft_data import prepare_semantic_sft_rows,tokenize_semantic_pair
    from experiments.semantic_sft_eval import run_inference,compare_paired_scores,reserve_semantic_audit_holdout
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
        TrainingArguments, Trainer, EarlyStoppingCallback,
    )
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training

    filtered=Path(hf_hub_download("birdsql/bird23-train-filtered",
        "data/train-00000-of-00001.jsonl",repo_type="dataset",revision=FILTERED_REV))
    train_tables=Path(hf_hub_download("Deema/BIRD-SQL","train_tables.json",
        repo_type="dataset",revision=SCHEMA_REV))
    dev_tables_only=Path(hf_hub_download("Deema/BIRD-SQL","dev_tables.json",
        repo_type="dataset",revision=SCHEMA_REV))
    original=[json.loads(x) for x in filtered.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(original)==6601
    schemas=json.loads(train_tables.read_text())
    dev_schemas=json.loads(dev_tables_only.read_text())
    train,val,report=prepare_semantic_sft_rows(
        original,schemas,dev_db_ids={x["db_id"] for x in dev_schemas},
        validation_db_fraction=0.15,min_train=1000,min_val=100,
    )
    assert len(train)==5742 and len(val)==857
    tokenizer=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side="right"
    def encode(rows: list[dict]):
        outputs=[]
        dropped=0
        for row in rows:
            try:
                outputs.append(tokenize_semantic_pair(tokenizer,row,max_length=max_length))
            except ValueError as exc:
                if "too_long" not in str(exc):
                    raise
                dropped+=1
        return outputs,dropped

    train_tokens,dropped_train=encode(train)
    val_tokens,dropped_val=encode(val)
    if len(train_tokens)<4000 or len(val_tokens)<100:
        raise ValueError("too few intact supervised semantic SQL examples")
    print("SEMANTIC_DATA",json.dumps({
        "raw_filtered":6601,"train":len(train_tokens),"validation":len(val_tokens),
        "skipped_train_long_schema":dropped_train,
        "skipped_validation_long_schema":dropped_val,
        "train_dev_database_overlap":0,
    }),flush=True)

    OUT.mkdir(exist_ok=False)
    report.update({
        "source_sha":SOURCE_SHA,"model":MODEL,"token_budget":max_length,
        "train_after_budget":len(train_tokens),"val_after_budget":len(val_tokens),
        "train_dropped_due_to_length":dropped_train,
        "val_dropped_due_to_length":dropped_val,
        "training_data_revision":FILTERED_REV,
        "schema_revision":SCHEMA_REV,
    })
    (OUT/"data-audit.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")

    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype)
    base=AutoModelForCausalLM.from_pretrained(
        MODEL,quantization_config=quant,device_map="auto",dtype=dtype,
        trust_remote_code=True)
    base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True)
    model=get_peft_model(base,LoraConfig(
        r=16,lora_alpha=32,lora_dropout=0.05,target_modules="all-linear",
        task_type="CAUSAL_LM",bias="none"))
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)>0
    assert all("lora_" in k for k,p in model.named_parameters() if p.requires_grad)
    model.config.use_cache=False

    def collate(batch):
        max_len=((max(len(x["input_ids"]) for x in batch)+7)//8)*8
        ids=torch.full((len(batch),max_len),tokenizer.pad_token_id,dtype=torch.long)
        mask=torch.zeros((len(batch),max_len),dtype=torch.long)
        labels=torch.full((len(batch),max_len),-100,dtype=torch.long)
        for i,x in enumerate(batch):
            n=len(x["input_ids"])
            ids[i,:n]=torch.tensor(x["input_ids"],dtype=torch.long)
            mask[i,:n]=1
            labels[i,:n]=torch.tensor(x["labels"],dtype=torch.long)
        return {"input_ids":ids,"attention_mask":mask,"labels":labels}

    train_ds=Dataset.from_list(train_tokens)
    val_ds=Dataset.from_list(val_tokens)
    args=TrainingArguments(
        output_dir=str(OUT/"checkpoints"),num_train_epochs=1.0,
        per_device_train_batch_size=1,per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,learning_rate=5e-5,
        lr_scheduler_type="cosine",warmup_steps=0.05,weight_decay=0,
        logging_steps=25,eval_strategy="steps",eval_steps=100,
        save_strategy="steps",save_steps=100,save_total_limit=2,
        load_best_model_at_end=True,metric_for_best_model="eval_loss",
        greater_is_better=False,gradient_checkpointing=True,
        bf16=(dtype==torch.bfloat16),fp16=(dtype==torch.float16),
        optim="paged_adamw_8bit",report_to="none",seed=42,data_seed=42,
        remove_unused_columns=False,
    )
    trainer=Trainer(model=model,args=args,train_dataset=train_ds,
        eval_dataset=val_ds.select(range(min(128,len(val_ds)))),
        data_collator=collate,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
    started=time.time()
    trained=trainer.train()
    assert trainer.state.global_step>64
    trainer.save_model(str(OUT))
    tokenizer.save_pretrained(str(OUT))
    trainer.save_state()
    weights=OUT/"adapter_model.safetensors"
    assert weights.exists() and weights.stat().st_size>1_000_000
    (OUT/"train-metrics.json").write_text(json.dumps({
        "steps":trainer.state.global_step,
        "training_metrics":trained.metrics,
        "elapsed_s":round(time.time()-started),
    },indent=2,default=str)+"\n")
    print("COLAB_SQL_ADAPTER_TRAINED",json.dumps({
        "steps":trainer.state.global_step,
        "adapter_sha256":hashlib.sha256(weights.read_bytes()).hexdigest(),
        "adapter_size":weights.stat().st_size,
        "semantic_gain":"unmeasured",
    }),flush=True)

    # Export the real adapter BEFORE trying DEV downloads or semantic evaluation.
    # If evaluation fails, the CI runner can still retrieve this validated adapter.
    if ARCHIVE.exists():
        raise FileExistsError("refuse to overwrite model archive")
    with zipfile.ZipFile(ARCHIVE,"w",compression=zipfile.ZIP_STORED) as z:
        for path in sorted(OUT.iterdir()):
            if path.is_file():
                z.write(path,arcname=path.name)
    print("COLAB_SQL_PRE_EVAL_ADAPTER_SAVED",json.dumps({
        "path":str(ARCHIVE), "bytes":ARCHIVE.stat().st_size,
        "sha256":hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        "semantic_correctness":"unmeasured",
    }),flush=True)

    # Release training model before loading independent evaluation workloads.
    del trainer,model,base,train_ds,val_ds
    gc.collect()
    torch.cuda.empty_cache()
    dev_rows,dev_schema,dbroot=download_bird_dev()
    frozen=json.loads((repo/"runs/bird-paired-dev-v4/paired.json").read_text())
    explored={str(x["case_id"]) for x in frozen["cases"]}
    assert len(explored)==100
    reserved=reserve_semantic_audit_holdout(
        dev_rows,historical_frozen_ids=explored,count=100,
    )
    assert len(reserved)==100 and not reserved.intersection(explored)
    excluded=explored | reserved
    assert len(excluded)==200
    print("SEMANTIC_EVAL_COHORT_ISOLATION",json.dumps({
        "historical_frozen_excluded":len(explored),
        "other_independent_audit_reserved":len(reserved),
        "total_excluded":len(excluded),
    }),flush=True)
    base_out=OUT/"independent100-base.jsonl"
    tuned_out=OUT/"independent100-tuned.jsonl"
    base_result=run_inference(
        model_id=MODEL,adapter_path=None,data=dev_rows,schemas=dev_schema,
        db_root=dbroot,excluded_ids=excluded,count=100,output=base_out)
    gc.collect()
    torch.cuda.empty_cache()
    tuned_result=run_inference(
        model_id=MODEL,adapter_path=OUT,data=dev_rows,schemas=dev_schema,
        db_root=dbroot,excluded_ids=excluded,count=100,output=tuned_out)
    old=[json.loads(x) for x in base_out.read_text().splitlines() if x.strip()]
    new=[json.loads(x) for x in tuned_out.read_text().splitlines() if x.strip()]
    paired=compare_paired_scores(old,new,min_gain=3)
    (OUT/"semantic-execution-match.json").write_text(json.dumps(paired,indent=2,sort_keys=True)+"\n")
    print("COLAB_SEMANTIC_VERIFIED_RESULT",json.dumps(paired,sort_keys=True),flush=True)

    # Append only evaluation outputs; keep the verified adapter untouched.
    with zipfile.ZipFile(ARCHIVE,"a",compression=zipfile.ZIP_STORED) as z:
        existing = set(z.namelist())
        for path in sorted(OUT.iterdir()):
            if path.is_file() and path.name not in existing:
                z.write(path,arcname=path.name)
    print("COLAB_SEMANTIC_ARTIFACT",json.dumps({
        "path":str(ARCHIVE),
        "size":ARCHIVE.stat().st_size,
        "sha256":hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        "paired_net_gain":paired["net_gain"],
        "gate_met":paired["gate_met"],
    }),flush=True)


if __name__=="__main__":
    main()
