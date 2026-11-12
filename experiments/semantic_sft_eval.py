"""Independent BIRD DEV cohort and reproducible paired execution-match evaluation."""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
import re


def choose_eval_cases(dev: list[dict], *, frozen_ids: set[str], mode: str,
                      count: int = 100, seed: str = "semantic-sft-heldout-v1") -> list[dict]:
    if mode not in {"independent", "frozen"} or count < 1:
        raise ValueError("invalid evaluation mode or count")
    ids = [str(x.get("question_id", x.get("case_id", ""))) for x in dev]
    if len(set(ids)) != len(ids) or not all(ids):
        raise ValueError("missing or duplicated case ids")
    eligible = [r for r in dev if (str(r.get("question_id",r.get("case_id"))) in frozen_ids) == (mode == "frozen")]
    if len(eligible) < count:
        raise ValueError("insufficient heldout cases")
    return sorted(
        eligible,
        key=lambda r: sha256((seed + ":" + str(r.get("question_id",r.get("case_id")))).encode()).hexdigest()
    )[:count]


def compare_paired_scores(base: list[dict], tuned: list[dict], *, min_gain: int=3) -> dict:
    aa = [str(r["case_id"]) for r in base]
    bb = [str(r["case_id"]) for r in tuned]
    if (len(aa) != len(bb) or len(set(aa)) != len(aa)
        or len(set(bb)) != len(bb) or set(aa) != set(bb)):
        raise ValueError("paired case IDs invalid or mismatched")
    if min_gain < 1:
        raise ValueError("positive gain required")
    a = {str(r["case_id"]):bool(r["correct"]) for r in base}
    b = {str(r["case_id"]):bool(r["correct"]) for r in tuned}
    gains = sum(not a[k] and b[k] for k in a)
    losses = sum(a[k] and not b[k] for k in a)
    return {
        "sample_size":len(a), "base_correct":sum(a.values()),
        "tuned_correct":sum(b.values()), "wrong_to_correct":gains,
        "correct_to_wrong":losses, "net_gain":gains-losses,
        "gate_met":gains-losses >= min_gain, "min_gain":min_gain,
        "metric":"BIRD SQLite project-local execution match (set equality)",
    }


def cleaned_sql(generated: str) -> str:
    from experiments.semantic_sft_data import sql_is_readonly
    raw = generated.strip()
    fence = chr(96) * 3
    if raw.startswith(fence):
        raw = raw.split("\n", 1)[-1].split(fence, 1)[0].strip()
    if not sql_is_readonly(raw):
        return ""
    return raw


def run_inference(
    *, model_id: str, adapter_path: Path | None, data: list[dict],
    schemas: list[dict], db_root: Path, excluded_ids: set[str],
    count: int, output: Path,
) -> dict:
    """Select independent cohort BEFORE loading any gold SQL."""
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from experiments.xiyan_dgsl_frozen100 import score_query
    from experiments.semantic_sft_data import make_semantic_prompt, render_bird_schema
    if output.exists():
        raise FileExistsError("refuse to overwrite evaluated model results")
    cases = choose_eval_cases(data, frozen_ids=excluded_ids, mode="independent", count=count)
    schema_by_id = {x["db_id"]:render_bird_schema(x) for x in schemas}
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    quant = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype)
    base = AutoModelForCausalLM.from_pretrained(model_id,quantization_config=quant,device_map="auto",dtype=dtype,trust_remote_code=True)
    model = PeftModel.from_pretrained(base, str(adapter_path)) if adapter_path else base
    model.eval()
    output.parent.mkdir(parents=True,exist_ok=True)
    scored=[]
    with output.open("w",encoding="utf-8") as fp:
        for i,row in enumerate(cases,1):
            cid = str(row["question_id"])
            db = str(row["db_id"])
            prompt = make_semantic_prompt(str(row["question"]),str(row.get("evidence") or ""),schema_by_id[db])
            inputs = tokenizer(prompt,return_tensors="pt",add_special_tokens=False)
            sql=""
            if inputs.input_ids.shape[1] <= 3072:
                inputs = {k:v.to(model.device) for k,v in inputs.items()}
                with torch.inference_mode():
                    tokens=model.generate(**inputs,max_new_tokens=256,do_sample=False,pad_token_id=tokenizer.eos_token_id)
                response=tokenizer.decode(tokens[0][inputs["input_ids"].shape[1]:],skip_special_tokens=True)
                sql=cleaned_sql(response)
            correct=False
            if sql:
                db_path=db_root/db/(db+".sqlite")
                if not db_path.is_file():
                    raise FileNotFoundError(f"missing BIRD database: {db}")
                correct=bool(score_query(db_path,sql,str(row["SQL"]))["frozen_execution_match"])
            result={"case_id":cid,"correct":correct,"predicted_sql":sql,"database_id":db}
            scored.append(result)
            fp.write(json.dumps(result,sort_keys=True,ensure_ascii=False)+"\n")
            fp.flush()
            print(f"SEMANTIC_EX {i}/{count} id={cid} correct={correct}",flush=True)
    summary={"count":len(scored),"correct":sum(x["correct"] for x in scored),"mode":"independent-dev-not-frozen100","adapter":bool(adapter_path)}
    (output.parent/(output.stem+"-summary.json")).write_text(json.dumps(summary,indent=2)+"\n")
    return summary
