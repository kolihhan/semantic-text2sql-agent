"""Frozen, diagnostic-only Qwen3.5-9B Q4_K_M vs Q8_0 CPU Text2SQL run.

Uses the unchanged BIRD v5 Direct/Guarded evaluator. Not a full benchmark.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time
import urllib.request

from evaluation.run_bird import run_paired_dev

SOURCE_HASH = "0310b7ca2dfb6247234e2073f93e7a75106ee41ba55231f75d06d2ef68efe2ec"
SPLIT_HASH = "b3b4ec070e63a7106d023917e0f0cc61ea75e2be2f2888b8c7bdaf2064fb3ae1"
FROZEN_IDS = ("38", "952", "943", "622", "1366", "1139")
SETTINGS = {"temperature": 0.0, "seed": 42, "num_predict": 512, "num_ctx": 4096, "num_thread": 4}
MODEL_MAP = {"q4": "qwen3.5:9b-q4_K_M", "q8": "qwen3.5:9b-q8_0"}
Q4_HISTORICAL_DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"


def _get_json(url: str, payload: dict | None = None, *, timeout: int = 300) -> dict:
    req = urllib.request.Request(
        url, data=(json.dumps(payload).encode("utf-8") if payload is not None else None),
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.load(resp)
    if not isinstance(result, dict):
        raise RuntimeError("Ollama returned non-dict JSON")
    return result


def _model_digest(model: str) -> str:
    tags = _get_json("http://127.0.0.1:11434/api/tags", timeout=15)
    items = tags.get("models", [])
    matched = [m for m in items if (m.get("name") or m.get("model")) == model]
    if len(matched) != 1:
        raise RuntimeError(f"Expected one locally installed model: {model}")
    digest = str(matched[0].get("digest", "")).lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise RuntimeError("Invalid model digest")
    return digest


class MeteredLocalOllama:
    def __init__(self, model: str, log: list[dict]) -> None:
        self.model = model
        self.log = log

    def complete_text(self, *, system: str, user: str) -> str:
        started = time.monotonic()
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "keep_alive": "20m",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": SETTINGS,
        }
        try:
            response = _get_json("http://127.0.0.1:11434/api/chat", payload, timeout=320)
        except Exception as exc:
            self.log.append({"error_type": type(exc).__name__, "elapsed_s": round(time.monotonic() - started, 2)})
            raise
        message = response.get("message", {})
        answer = message.get("content") if isinstance(message, dict) else None
        done = response.get("done_reason")
        entry = {
            "elapsed_s": round(time.monotonic() - started, 2),
            "done_reason": done,
            "prompt_eval_count": response.get("prompt_eval_count"),
            "eval_count": response.get("eval_count"),
            "load_duration_ns": response.get("load_duration"),
            "eval_duration_ns": response.get("eval_duration"),
            "prompt_eval_duration_ns": response.get("prompt_eval_duration"),
            "output_chars": len(answer) if isinstance(answer, str) else None,
        }
        self.log.append(entry)
        if done != "stop" or not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("Ollama incomplete/no answer; reject for Strict EX")
        return answer


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = MODEL_MAP[args.quant]
    observed_digest = _model_digest(model)
    reference = json.loads(Path("runs/bird-paired-dev-v5/paired.json").read_text("utf-8"))
    if reference.get("status") != "complete" or len(reference.get("cases", [])) != 100:
        raise RuntimeError("Historical BIRD v5 reference is not complete 100 cases")
    if reference["source_sha256"] != SOURCE_HASH or reference["split_manifest_sha256"] != SPLIT_HASH:
        raise RuntimeError("Frozen BIRD v5 identity has changed")
    if sha256(Path(args.source).read_bytes()).hexdigest() != SOURCE_HASH:
        raise RuntimeError("BIRD source hash mismatch")
    first_six = reference["cases"][:6]
    if tuple(str(x["case_id"]) for x in first_six) != FROZEN_IDS:
        raise RuntimeError("Frozen six IDs changed")
    source_rows = json.loads(Path(args.source).read_text("utf-8"))
    by_id = {str(x.get("question_id") or x.get("id") or i): x for i, x in enumerate(source_rows)}
    for x in first_six:
        item = by_id[str(x["case_id"])]
        if str(item["db_id"]) != str(x["database_id"]) or str(item["question"]) != str(x["question"]):
            raise RuntimeError("Frozen question / db identity drift")

    if args.quant == "q4" and observed_digest != Q4_HISTORICAL_DIGEST:
        raise RuntimeError("Q4 historical Ollama digest drift; cannot compare original model")
    manifest = {
        "protocol_version": 1, "source_sha256": SOURCE_HASH,
        "seed": "q4-vs-q8-cpu-exposed-v5-six-v1",
        "dev_case_ids": list(FROZEN_IDS), "final_case_ids": [],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    calls: list[dict] = []
    initial = {
        "study": "Qwen3.5-9B quantization ablation (diagnostic, exposed BIRD DEV)",
        "quant": args.quant, "model": model,
        "digest": observed_digest, "historical_q4_digest": Q4_HISTORICAL_DIGEST,
        "settings": SETTINGS, "think": False,
        "case_ids": list(FROZEN_IDS), "source_sha256": SOURCE_HASH,
        "split_sha256": SPLIT_HASH,
        "status": "incomplete", "cases_completed": 0,
        "calls": [], "error_type": None,
    }
    report_path = output_dir / "report.json"
    paired_path = output_dir / "paired.json"
    report_path.write_text(json.dumps(initial, indent=2))
    print("FROZEN_SIX_VERIFIED model=" + model + " sha256=" + observed_digest, flush=True)
    status = "incomplete"
    failure = None
    try:
        result = run_paired_dev(
            source_path=Path(args.source), database_root=Path(args.database_root),
            manifest_path=output_dir / "manifest.json",
            output_path=paired_path, model=model,
            provider_factory=lambda: MeteredLocalOllama(model, calls),
            timeout_s=320, max_repairs=2, max_rows=50000,
        )
        status = result["status"]
    except Exception as exc:
        failure = type(exc).__name__
        print("PILOT_INCOMPLETE error_type=" + failure, flush=True)

    result_record = json.loads(paired_path.read_text("utf-8")) if paired_path.exists() else {}
    records = result_record.get("cases", [])
    report = {**initial, "status": status, "error_type": failure,
              "cases_completed": len(records), "calls": calls,
              "metrics": result_record.get("metrics") if status == "complete" else None,
              "cases": [
                  {"case_id": x["case_id"], "database_id": x["database_id"],
                   "direct_ex": x["direct"]["official_ex"],
                   "guarded_ex": x["guarded"]["official_ex"],
                   "execution_success_direct": x["direct"]["execution_success"],
                   "execution_success_guarded": x["guarded"]["execution_success"],
                   "model_calls": x["guarded"]["model_calls"]}
                  for x in records
              ]}
    report_path.write_text(json.dumps(report, indent=2))
    print("QUANT_PILOT_REPORT " + json.dumps({
        "model": model, "status": status, "completed": len(records),
        "direct_ex": sum(c["direct_ex"] for c in report["cases"]),
        "guarded_ex": sum(c["guarded_ex"] for c in report["cases"]),
        "inference_calls": len(calls), "error_type": failure,
    }), flush=True)
    return 0 if status == "complete" and len(records) == len(FROZEN_IDS) else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant", choices=list(MODEL_MAP), required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--database-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
