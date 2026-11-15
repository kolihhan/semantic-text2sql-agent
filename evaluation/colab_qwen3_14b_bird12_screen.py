"""GPU-screen a DIFFERENT, larger Qwen model, not another SQL repair heuristic.

Qwen3-14B 4bit original untrained Instruct text model vs *stored* original
Qwen3.5-9B Guarded SQL from #43738683554, SAME 12 original BIRD questions
and physical SQLite DBs. Both use the SAME unchanged repo Guarded verifier,
schema selector, external evidence, max repairs. Reference SQL is read ONLY
after all model predictions are frozen. Training or data-dependent edits
are explicitly disallowed. Previously exposed DEV12; screen, not hidden OOD.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256,file_digest
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import monotonic
from urllib.request import urlretrieve
import zipfile

SOURCE_CODE_SHA="__CODE_SHA__"
COHORT_SHA="d889f818f2988d66e612e97e0d9762f797c300fa205d73addfb6b62df24d9c86"
BIRD_SHA="ffd8018378ddb1a8794753e0a31cfc81862ff7318a5184c22f3dc4ce03a03feb"
DB_ARCHIVE_SHA="cdd6d19faeb45a23970b98d3ef6c40a87987c95459c2cf12076897a60cf5a630"
BASE_QWEN9B_DIGEST="56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
MODEL="Qwen/Qwen3-14B"
PROTOCOL="qwen3-14b-4bit-original-guarded-bird12-same-original9b-source-v1"
COHORT=Path("/content/qwen-native-batched-fresh-bird12-cohort.json")
ORIGINAL=Path("/content/qwen-native-batched-bird12-final.json")
OUTPUT=Path("/content/qwen3-14b-guarded-original-bird12-comparison.json")
WORK=Path("/content/qwen3-14b-guarded-source-screen")
DATA_URL="https://huggingface.co/datasets/birdsql/bird_sql_dev_20251106/resolve/3c11fb193e5439b338e23677fa0aae11e8b85db9/data/dev_20251106-00000-of-00001.json?download=true"
DATABASE_URL="https://huggingface.co/datasets/HAL-9001/bird-dev/resolve/main/dev.zip"
MAX_ROWS=50_000
N=12


def sha_file(path:Path)->str:
    with path.open("rb") as fp:
        return file_digest(fp,"sha256").hexdigest()


def source_inputs():
    if SOURCE_CODE_SHA=="__CODE_SHA__" or len(SOURCE_CODE_SHA)!=40:
        raise RuntimeError("exact source-code revision not frozen")
    m=json.loads(COHORT.read_text())
    rows=m.get("cases")
    if (m.get("cases_sha")!=COHORT_SHA or
        m.get("n")!=12 or m.get("db_count")!=6 or
        m.get("gold_used_to_select") is not False or
        m.get("independent_question_ids_from_exposed_160") is not True or
        m.get("digest")!=BASE_QWEN9B_DIGEST or
        not isinstance(rows,list) or len(rows)!=N or
        sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()!=COHORT_SHA):
        raise RuntimeError("pinned source model-independent cohort changed")
    old=json.loads(ORIGINAL.read_text())
    original_rows=old.get("cases")
    if (old.get("protocol")!="qwen9b-native-batched-tool-observation-versus-guarded-bird-dev12-v2"
        or old.get("n")!=N or not isinstance(original_rows,list)
        or len(original_rows)!=N or
        old.get("arms",{}).get("original",{}).get("strict_set_ex")!=3 or
        old.get("arms",{}).get("original",{}).get("strict_bag_ex")!=3):
        raise RuntimeError("physically frozen original9B paired original SQL incomplete")
    by_id={str(x["case_id"]):x for x in original_rows}
    if len(by_id)!=N or set(by_id)!={x["case_id"] for x in rows}:
        raise RuntimeError("original9B baseline and frozen manifest identity mismatch")
    for row in rows:
        prior=by_id[row["case_id"]]
        if (prior["database_id"]!=row["database_id"] or
            prior["question"]!=row["question"] or
            prior["evidence"]!=row["evidence"]):
            raise RuntimeError("historical original9B question/schema/evidence drift")
    return rows,by_id


def download_data():
    WORK.mkdir(parents=True,exist_ok=False)
    raw=WORK/"dev.json"
    urlretrieve(DATA_URL,raw)
    if sha_file(raw)!=BIRD_SHA:
        raise RuntimeError("original publisher BIRD DEV JSON drift")
    dbzip=WORK/"dev.zip"
    urlretrieve(DATABASE_URL,dbzip)
    if sha_file(dbzip)!=DB_ARCHIVE_SHA:
        raise RuntimeError("original first-party BIRD DEV sqlite zip drift")
    with zipfile.ZipFile(dbzip) as z:
        entries=[n for n in z.namelist() if n.endswith("/dev_databases.zip") and "__MACOSX" not in n]
        if len(entries)!=1:
            raise RuntimeError("unexpected physical database zip layout")
        nested=WORK/"dev_databases.zip"
        nested.write_bytes(z.read(entries[0]))
    root=WORK/"databases"
    root.mkdir()
    with zipfile.ZipFile(nested) as z:
        z.extractall(root)
    return raw,root


class OriginalQwen3_14B:
    def __init__(self,model,tokenizer,torch):
        self.model=model
        self.tokenizer=tokenizer
        self.torch=torch
        self.calls=0
    def complete_text(self,*,system:str,user:str)->str:
        self.calls+=1
        device=next(self.model.parameters()).device
        messages=[{"role":"system","content":system},{"role":"user","content":user}]
        inputs=self.tokenizer.apply_chat_template(
            messages,tokenize=True,add_generation_prompt=True,
            enable_thinking=False,return_tensors="pt"
        ).to(device)
        with self.torch.inference_mode():
            generated=self.model.generate(
                input_ids=inputs,max_new_tokens=700,do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        out=self.tokenizer.decode(generated[0][inputs.shape[-1]:],
                                  skip_special_tokens=True).strip()
        del inputs,generated
        return out


def main():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("real Colab GPU mandatory, no fake CPU inference")
    props=torch.cuda.get_device_properties(0)
    if props.total_memory/2**30<14.5:
        raise RuntimeError("T4 >=14.5GiB required for quantized Qwen3-14B")
    rows,prior=source_inputs()
    if WORK.exists() or OUTPUT.exists():
        raise FileExistsError("refuse to overwrite physical source artifacts")
    repo=Path("/content/original-text2sql")
    subprocess.run(["git","clone","--branch","experiment/qwen3-14b-samebird12-colab-screen-20261116",
                    "--depth","10","https://github.com/kolihhan/semantic-text2sql-agent.git",str(repo)],check=True)
    subprocess.run(["git","-C",str(repo),"checkout","--detach",SOURCE_CODE_SHA],check=True)
    observed=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    if observed!=SOURCE_CODE_SHA:
        raise RuntimeError("tested SQL repo SHA drift")
    subprocess.run([sys.executable,"-m","pip","install","-q","-e",str(repo),
                    "transformers>=4.58","accelerate","bitsandbytes",
                    "huggingface_hub"],check=True)
    sys.path[:0]=[str(repo/"src"),str(repo)]
    from huggingface_hub import HfApi,snapshot_download
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
    from evaluation.run_bird import _resolve_database
    from evaluation.schema_context import select_schema_context
    from semantic_sql.catalog import DatabaseCatalog
    from semantic_sql.inference import run_guarded
    from semantic_sql.execution import execute_readonly,SQLExecutionError
    original_json,dbroot=download_data()
    all_source=json.loads(original_json.read_text(encoding="utf-8"))
    original_by_id={str(r.get("question_id") or r.get("id") or i):r
                    for i,r in enumerate(all_source)}
    info=HfApi().model_info(MODEL)
    model_sha=str(info.sha)
    if len(model_sha)!=40:
        raise RuntimeError("immutable Qwen3-14B revision unavailable")
    checkpoint=snapshot_download(MODEL,revision=model_sha)
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,trust_remote_code=False)
    quant=BitsAndBytesConfig(
        load_in_4bit=True,bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True
    )
    model=AutoModelForCausalLM.from_pretrained(
        checkpoint,quantization_config=quant,
        device_map={"":0},torch_dtype=torch.float16,
        trust_remote_code=False
    )
    model.eval()
    model_provider=OriginalQwen3_14B(model,tokenizer,torch)
    print("QWEN3_14B_REAL_GPU_ORIGINAL_MODEL_LOADED",json.dumps({
        "model":MODEL,"sha":model_sha,"gpu":props.name,
        "vram_gib":round(props.total_memory/2**30,2),
        "nf4":True,"training":False,
        "questions":N,"strict_baseline9b":3
    },sort_keys=True),flush=True)
    results=[]
    for record in rows:
        cid=record["case_id"];dbid=record["database_id"]
        raw=original_by_id[cid]
        if (raw["db_id"]!=dbid or
            " ".join(str(raw["question"]).split())!=record["question"]):
            raise RuntimeError("BIRD original publisher case identity changed")
        db=_resolve_database(dbroot,dbid)
        cat=DatabaseCatalog.from_sqlite(db)
        schema,_=select_schema_context(cat,dbroot,dbid,record["question"])
        start=monotonic()
        before_calls=model_provider.calls
        prediction=run_guarded(
            database=db,provider=model_provider,question=record["question"],
            schema_context=schema,evidence=record["evidence"] or None,
            max_repairs=2,max_rows=MAX_ROWS
        )
        previous=prior[cid]["arms"]["original"]
        predicted_rows=prediction.rows if prediction.status=="ok" and not prediction.truncated else None
        results.append({
            "case_id":cid,"database_id":dbid,
            "question":record["question"],
            "schema_sha256":sha256(schema.encode()).hexdigest(),
            "original9b":{
                "sql":previous["sql"],"status":previous["status"],
                "strict_set_ex":previous["set_ex"],
                "strict_bag_ex":previous["bag_ex"],
                "model_calls":previous["model_calls"],
            },
            "qwen3_14b":{
                "sql":prediction.candidate.sql if prediction.candidate else None,
                "status":prediction.status,"truncated":prediction.truncated,
                "model_calls":model_provider.calls-before_calls,
                "elapsed_s":round(monotonic()-start,2),
                "raw_rows":predicted_rows,
            }
        })
        print("QWEN3_14B_GENERATED",cid,dbid,"status",prediction.status,
              "calls",model_provider.calls-before_calls,flush=True)
    # All 12 source-grounded outputs frozen BEFORE first reference SQL execution.
    proof=WORK/"predictions-frozen-before-gold.json"
    proof.write_text(json.dumps(results,default=str,ensure_ascii=False,indent=2)+"\n")
    scorable=0
    for record in results:
        db=_resolve_database(dbroot,record["database_id"])
        raw=original_by_id[record["case_id"]]
        try:
            ref=execute_readonly(db,raw["SQL"],max_rows=MAX_ROWS)
            good=not ref.truncated
        except (SQLExecutionError,ValueError):
            ref,good=None,False
        scorable+=good
        output=record["qwen3_14b"]
        pred=output.pop("raw_rows")
        output["set_ex"]=bool(
            good and pred is not None and set(map(tuple,pred))==set(map(tuple,ref.rows))
        ) if good else None
        output["bag_ex"]=bool(
            good and pred is not None and Counter(map(tuple,pred))==Counter(map(tuple,ref.rows))
        ) if good else None
    if scorable!=N:
        raise RuntimeError(f"original BIRD12 reference source incomplete: {scorable}")
    base_set=sum(r["original9b"]["strict_set_ex"] for r in results)
    base_bag=sum(r["original9b"]["strict_bag_ex"] for r in results)
    new_set=sum(r["qwen3_14b"]["set_ex"] for r in results)
    new_bag=sum(r["qwen3_14b"]["bag_ex"] for r in results)
    rescued=[r for r in results if not r["original9b"]["strict_set_ex"] and r["qwen3_14b"]["set_ex"]]
    damaged=[r for r in results if r["original9b"]["strict_set_ex"] and not r["qwen3_14b"]["set_ex"]]
    report={
        "protocol":PROTOCOL,"source_sha":BIRD_SHA,
        "source_cohort_sha":COHORT_SHA,
        "old_original_qwen9b_digest":BASE_QWEN9B_DIGEST,
        "new_qwen3_14b_model":MODEL,"new_model_sha":model_sha,
        "n":N,"scorable":scorable,
        "strict_set_qwen35_9b":base_set,
        "strict_set_qwen3_14b":new_set,
        "strict_bag_qwen35_9b":base_bag,
        "strict_bag_qwen3_14b":new_bag,
        "rescues":[r["case_id"] for r in rescued],
        "regressions":[r["case_id"] for r in damaged],
        "qwen14b_model_calls":model_provider.calls,
        "qwen14b_elapsed_s":round(sum(r["qwen3_14b"]["elapsed_s"] for r in results),1),
        "positive_screen_only":bool(
            new_set-base_set>=3 and not damaged and
            new_bag>=base_bag and len({r["database_id"] for r in rescued})>=3
        ),
        "promoted":False,"no_training":True,
        "original_bird100_ex_unchanged":39,
        "source_caveat":"public previously evaluated BIRD DEV12; need independent fresh DB tests before accepting model swap",
        "cases":results
    }
    OUTPUT.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str)+"\n")
    print("QWEN3_14B_VS_ORIGINAL9B_BIRD12_FINAL",
          json.dumps({k:v for k,v in report.items() if k!="cases"},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
