"""Same pinned Qwen3.5-9B: verify Ollama native thinking is actually supported.

Pure capability gate, NOT a BIRD accuracy claim. No SQL training, no other
model; reasoning text is never saved in CI artifacts, only its length.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from urllib.request import Request, urlopen

from evaluation.run_bird import _ollama_model_digest

MODEL = "qwen3.5:9b-q4_K_M"
DIGEST = "56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90"
QUESTION = (
    "SQLite tables:\n"
    "accounts(account_id, name)\n"
    "payments(payment_id, account_id, amount, paid_on)\n"
    "Question: Which account IDs have a total sum of payments in the year 2024 "
    "above 100? Return account_id and total amount, ordered from largest total.\n"
    "Return one read-only SQLite SELECT or WITH query only, no explanation."
)
SYSTEM = "Generate one correct read-only SQLite SQL query. Output SQL only."


def payload(*, think: bool) -> dict:
    return {
        "model": MODEL, "stream": False, "think": think,
        "options": {"temperature": 0, "seed": 42,
                    "num_predict": 4096 if think else 1024},
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": QUESTION}],
    }


def ask(*, think: bool) -> dict:
    request = Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(payload(think=think)).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=1600) as response:
        obj = json.load(response)
    message = obj.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("Ollama response lacks message")
    content = str(message.get("content") or "")
    thinking = str(message.get("thinking") or "")
    # Reasoning trace itself is deliberately discarded.
    return {
        "think_requested": think,
        "sql": content,
        "thinking_present": bool(thinking.strip()),
        "thinking_chars": len(thinking),
        "content_chars": len(content),
        "reasoning_trace_saved": False,
        "done_reason": obj.get("done_reason"),
        "eval_count": obj.get("eval_count"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite preflight")
    digest = _ollama_model_digest("http://127.0.0.1:11434", MODEL)
    if digest != DIGEST:
        raise ValueError("Qwen9B model digest drift")
    direct = ask(think=False)
    reasoned = ask(think=True)
    report = {
        "model": MODEL, "digest": digest,
        "question_sha256": sha256(QUESTION.encode()).hexdigest(),
        "no_training": True, "no_other_model": True,
        "arms": [direct, reasoned],
        "supported": bool(reasoned["thinking_present"] and reasoned["sql"].strip()),
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("QWEN_NATIVE_THINK_GATE", json.dumps({
        k:v for k,v in report.items() if k!="arms"
    }), "arms", [{"think":r["think_requested"],
                  "thinking_chars":r["thinking_chars"],
                  "content_chars":r["content_chars"],
                  "done_reason":r["done_reason"]} for r in report["arms"]],flush=True)
    if not report["supported"]:
        raise RuntimeError("Native thinking did not provide both reasoning and final SQL")


if __name__ == "__main__":
    main()
